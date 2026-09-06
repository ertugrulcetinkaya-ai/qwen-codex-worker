#!/usr/bin/env python3
"""Sandbox filesystem agent for the local Qwen worker."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from qwen_worker_server import (
    DEFAULT_MODEL,
    HTTP_TIMEOUT,
    MAX_OUTPUT_TOKENS,
    PRESENCE_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    _elapsed_ms,
    _metadata,
    _configured_base_url,
    _preflight,
    _request_with_connect_retries,
    _stream_text,
)


MAX_AGENT_ACTIONS = 30
MAX_PROTOCOL_ERRORS = 3
MAX_READ_BYTES = 64_000
MAX_READ_CHARS = 32_000
MAX_GREP_RESULTS = 100
MAX_GREP_CHARS = 20_000
MAX_WRITE_BYTES = 1_000_000
MAX_ACTION_TEXT_CHARS = 20_000
MAX_TASK_BYTES = 480_000
MAX_TASK_CHARS = 120_000
MAX_COMMAND_OUTPUT_CHARS = 8_000
AGENT_SYSTEM_PROMPT = """You are a constrained code implementation worker operating only inside the
explicit sandbox supplied by the caller.

Use exactly one JSON action per turn. Never emit a unified diff, markdown
fences, shell script, or implementation essay. The worker executes only
validated actions.

Available actions:
{"action":"read","path":"relative/path","start_line":1,"end_line":80}
{"action":"grep","query":"text","paths":["relative/path-or-directory"]}
{"action":"replace","path":"relative/file","old":"exact text","new":"replacement text"}
{"action":"replace","path":"relative/file","old":"exact text","new":"replacement text","count":2}
{"action":"write","path":"relative/new-file","content":"full file content"}
{"action":"run","command":"one explicitly allowed command"}
{"action":"done","summary":"short completion summary"}

Read and write only paths allowed by the caller. Do not access secrets,
credentials, .env files, or paths outside the sandbox. Use replace for small
edits. Read an existing file before rewriting it. Stop with done when the
filesystem state satisfies the task.
"""

FORBIDDEN_BASENAMES = frozenset(
    {
        "credentials",
        "credentials.json",
        "passwords",
        "secrets",
        "secrets.json",
        "tokens",
        "tokens.json",
    }
)
FORBIDDEN_COMMANDS = frozenset(
    {
        "bash",
        "cmd",
        "curl",
        "dd",
        "fish",
        "nc",
        "netcat",
        "ncat",
        "powershell",
        "rm",
        "rmdir",
        "scp",
        "sftp",
        "sh",
        "ssh",
        "telnet",
        "wget",
        "zsh",
    }
)
FORBIDDEN_GIT_COMMANDS = frozenset(
    {"apply", "checkout", "clean", "commit", "push", "reset", "restore", "stash"}
)
FORBIDDEN_INSTALL_COMMANDS = frozenset(
    {"add", "install", "publish", "uninstall", "upgrade"}
)


class AgentError(ValueError):
    """A deterministic sandbox or action-protocol failure."""


@dataclass(frozen=True)
class AgentGeneration:
    text: str
    completion_tokens: int | None
    reasoning_chars: int
    elapsed_ms: int
    response_chars: int = 0
    finish_reason: str | None = None
    http_status: int | None = None


@dataclass(frozen=True)
class AgentResult:
    status: str
    actions: int
    files_changed: list[str]
    model_calls: int
    completion_tokens: int
    reasoning_chars: int
    elapsed_ms: int
    read_bytes: int
    write_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "actions": self.actions,
            "files_changed": self.files_changed,
            "model_calls": self.model_calls,
            "completion_tokens": self.completion_tokens,
            "reasoning_chars": self.reasoning_chars,
            "elapsed_ms": self.elapsed_ms,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
        }


def _relative_path(value: Any, *, field: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AgentError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AgentError(f"{field} must stay inside the sandbox")
    normalized = path.as_posix()
    if normalized in {"", "."}:
        raise AgentError(f"{field} must name a sandbox path")
    return normalized


def _within(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _is_forbidden_path(relative: str) -> bool:
    for part in Path(relative).parts:
        lowered = part.lower()
        if lowered.startswith(".env") or lowered in FORBIDDEN_BASENAMES:
            return True
    return False


def _declared_scope(root: Path, value: str) -> tuple[str, bool]:
    relative = _relative_path(value)
    if _is_forbidden_path(relative):
        raise AgentError(f"forbidden path: {relative}")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        is_directory = resolved.is_dir()
    except FileNotFoundError:
        is_directory = False
        parent = candidate.parent.resolve(strict=True)
        resolved = parent / candidate.name
    except RuntimeError as exc:
        raise AgentError(f"invalid symlink path: {relative}") from exc
    if not _within(root, resolved):
        raise AgentError(f"path escapes sandbox: {relative}")
    return relative, is_directory


class SandboxAgent:
    """Execute validated agent actions inside an explicit sandbox."""

    def __init__(
        self,
        sandbox: str | Path,
        allow: Iterable[str],
        *,
        read_only: Iterable[str] = (),
        allowed_commands: Iterable[str] = (),
    ) -> None:
        root = Path(sandbox).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise AgentError("sandbox must resolve to a directory")
        self.root = root
        self.writable_scopes = tuple(
            _declared_scope(root, value) for value in allow
        )
        self.readable_scopes = self.writable_scopes + tuple(
            _declared_scope(root, value) for value in read_only
        )
        if not self.writable_scopes:
            raise AgentError("at least one --allow path is required")
        self.allowed_commands = tuple(
            self._parse_allowed_command(command) for command in allowed_commands
        )
        self.read_paths: set[str] = set()
        self.changed_paths: set[str] = set()
        self.read_bytes = 0
        self.write_bytes = 0

    @staticmethod
    def _parse_allowed_command(command: str) -> tuple[str, ...]:
        if not isinstance(command, str) or not command.strip():
            raise AgentError("allowed commands must be non-empty strings")
        return _parse_command(command)

    def _scope_allows(
        self,
        relative: str,
        scopes: tuple[tuple[str, bool], ...],
    ) -> bool:
        candidate = Path(relative).parts
        for scope, is_directory in scopes:
            scope_parts = Path(scope).parts
            if relative == scope:
                return True
            if is_directory and candidate[: len(scope_parts)] == scope_parts:
                return True
        return False

    def _resolve(
        self,
        relative_value: Any,
        *,
        writable: bool = False,
        readable: bool = False,
        must_exist: bool = False,
    ) -> tuple[str, Path]:
        relative = _relative_path(relative_value)
        if _is_forbidden_path(relative):
            raise AgentError(f"forbidden path: {relative}")
        scopes = self.writable_scopes if writable else self.readable_scopes
        if not self._scope_allows(relative, scopes):
            scope_name = "writable" if writable else "readable"
            raise AgentError(f"path is not in the {scope_name} allowlist: {relative}")
        candidate = self.root / relative
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (FileNotFoundError, RuntimeError) as exc:
            raise AgentError(f"invalid sandbox path: {relative}") from exc
        if not _within(self.root, resolved):
            raise AgentError(f"path escapes sandbox: {relative}")
        if must_exist and not resolved.exists():
            raise AgentError(f"path does not exist: {relative}")
        return relative, resolved

    def _read_text(self, relative: str, path: Path) -> str:
        if not path.is_file():
            raise AgentError(f"path is not a regular file: {relative}")
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            raise AgentError(f"file exceeds read limit: {relative}")
        try:
            data = path.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AgentError(f"could not read UTF-8 file: {relative}") from exc
        if b"\x00" in data:
            raise AgentError(f"binary file rejected: {relative}")
        self.read_paths.add(relative)
        self.read_bytes += len(data)
        return text

    def _atomic_write(self, relative: str, path: Path, content: str) -> None:
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise AgentError(f"write exceeds byte limit: {relative}")
        if not content:
            raise AgentError("empty writes are rejected")
        parent = path.parent
        if not parent.is_dir():
            raise AgentError(f"parent directory does not exist: {relative}")
        old_mode: int | None = None
        if path.exists():
            if not path.is_file():
                raise AgentError(f"path is not a regular file: {relative}")
            old_mode = stat.S_IMODE(path.stat().st_mode)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=parent,
                prefix=f".{path.name}.qwen-",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            if old_mode is not None:
                os.chmod(temporary_name, old_mode)
            os.replace(temporary_name, path)
        except OSError as exc:
            raise AgentError(f"atomic write failed: {relative}") from exc
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink(missing_ok=True)
                except OSError:
                    pass
        self.changed_paths.add(relative)
        self.write_bytes += len(data)

    def _read_action(self, action: dict[str, Any]) -> dict[str, Any]:
        relative, path = self._resolve(
            action.get("path"), readable=True, must_exist=True
        )
        text = self._read_text(relative, path)
        start = action.get("start_line", 1)
        end = action.get("end_line")
        if isinstance(start, bool) or not isinstance(start, int) or start < 1:
            raise AgentError("start_line must be a positive integer")
        if end is not None and (
            isinstance(end, bool) or not isinstance(end, int) or end < start
        ):
            raise AgentError("end_line must be an integer after start_line")
        lines = text.splitlines()
        selected = lines[start - 1 : end]
        content = "\n".join(selected)
        if selected and text.endswith("\n") and end is None:
            content += "\n"
        if len(content.encode("utf-8")) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + "\n[truncated]"
        return {
            "ok": True,
            "action": "read",
            "path": relative,
            "content": content,
        }

    def _grep_action(self, action: dict[str, Any]) -> dict[str, Any]:
        query = action.get("query")
        if not isinstance(query, str) or not query:
            raise AgentError("grep query must be a non-empty string")
        requested_paths = action.get(
            "paths", [scope[0] for scope in self.readable_scopes]
        )
        if not isinstance(requested_paths, list) or any(
            not isinstance(value, str) for value in requested_paths
        ):
            raise AgentError("grep paths must be an array of strings")

        files: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for requested in requested_paths:
            relative, base = self._resolve(requested, readable=True, must_exist=True)
            if base.is_file():
                files.append((relative, base))
                continue
            if not base.is_dir():
                raise AgentError(f"grep path is not a file or directory: {relative}")
            for candidate in base.rglob("*"):
                if not candidate.is_file():
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except (FileNotFoundError, RuntimeError):
                    continue
                if not _within(self.root, resolved):
                    raise AgentError(f"grep encountered symlink escape: {candidate}")
                normalized = resolved.relative_to(self.root).as_posix()
                if not self._scope_allows(normalized, self.readable_scopes):
                    continue
                if _is_forbidden_path(normalized) or resolved in seen:
                    continue
                seen.add(resolved)
                files.append((normalized, resolved))
                if len(files) >= MAX_GREP_RESULTS:
                    break
            if len(files) >= MAX_GREP_RESULTS:
                break

        matches: list[str] = []
        total_chars = 0
        for relative, path in files:
            text = self._read_text(relative, path)
            for line_number, line in enumerate(text.splitlines(), 1):
                if query not in line:
                    continue
                entry = f"{relative}:{line_number}:{line}"
                if total_chars + len(entry) + 1 > MAX_GREP_CHARS:
                    return {
                        "ok": True,
                        "action": "grep",
                        "matches": matches,
                        "truncated": True,
                    }
                matches.append(entry)
                total_chars += len(entry) + 1
                if len(matches) >= MAX_GREP_RESULTS:
                    return {
                        "ok": True,
                        "action": "grep",
                        "matches": matches,
                        "truncated": True,
                    }
        return {
            "ok": True,
            "action": "grep",
            "matches": matches,
            "truncated": False,
        }

    def _replace_action(self, action: dict[str, Any]) -> dict[str, Any]:
        old = action.get("old")
        new = action.get("new")
        if not isinstance(old, str) or not old:
            raise AgentError("replace old must be a non-empty string")
        if not isinstance(new, str):
            raise AgentError("replace new must be a string")
        relative, path = self._resolve(
            action.get("path"), writable=True, must_exist=True
        )
        text = self._read_text(relative, path)
        matches = text.count(old)
        count = action.get("count")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise AgentError("replace count must be a positive integer")
        if matches == 0:
            raise AgentError(f"replace text was not found: {relative}")
        if count is None and matches != 1:
            raise AgentError(
                f"replace text must match exactly once; found {matches}: {relative}"
            )
        if count is not None and count > matches:
            raise AgentError(
                f"replace count {count} exceeds matches {matches}: {relative}"
            )
        replacement_count = 1 if count is None else count
        updated = text.replace(old, new, replacement_count)
        self._atomic_write(relative, path, updated)
        return {
            "ok": True,
            "action": "replace",
            "path": relative,
            "matches": matches,
            "replaced": replacement_count,
        }

    def _write_action(self, action: dict[str, Any]) -> dict[str, Any]:
        content = action.get("content")
        if not isinstance(content, str):
            raise AgentError("write content must be a string")
        relative, path = self._resolve(action.get("path"), writable=True)
        exists = path.exists()
        if exists and not path.is_file():
            raise AgentError(f"path is not a regular file: {relative}")
        if exists and path.stat().st_size > 0 and relative not in self.read_paths:
            raise AgentError(f"read existing file before rewriting: {relative}")
        self._atomic_write(relative, path, content)
        return {
            "ok": True,
            "action": "write",
            "path": relative,
            "bytes": len(content.encode("utf-8")),
        }

    def _run_action(self, action: dict[str, Any]) -> dict[str, Any]:
        command = action.get("command")
        argv = _parse_command(command)
        if not _command_allowed(argv, self.allowed_commands):
            raise AgentError(f"command is not allowlisted: {command}")
        try:
            completed = subprocess.run(
                argv,
                cwd=self.root,
                env=_safe_environment(),
                shell=False,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentError(f"command failed to run: {command}") from exc
        return {
            "ok": completed.returncode == 0,
            "action": "run",
            "command": " ".join(argv),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-MAX_COMMAND_OUTPUT_CHARS:],
            "stderr": completed.stderr[-MAX_COMMAND_OUTPUT_CHARS:],
        }

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise AgentError("action must be a JSON object")
        name = action.get("action")
        if name == "read":
            return self._read_action(action)
        if name == "grep":
            return self._grep_action(action)
        if name == "replace":
            return self._replace_action(action)
        if name == "write":
            return self._write_action(action)
        if name == "run":
            return self._run_action(action)
        if name == "done":
            summary = action.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise AgentError("done summary must be a non-empty string")
            if len(summary) > 500 or summary.count("\n") > 4:
                raise AgentError("done summary is too long")
            return {"ok": True, "action": "done", "summary": summary.strip()}
        raise AgentError(f"unknown action: {name!r}")


def _parse_command(command: Any) -> tuple[str, ...]:
    if not isinstance(command, str) or not command.strip():
        raise AgentError("command must be a non-empty string")
    if any(symbol in command for symbol in (";", "|", "&", ">", "<", "$", chr(96))):
        raise AgentError("shell operators are not allowed")
    try:
        argv = tuple(shlex.split(command))
    except ValueError as exc:
        raise AgentError("command has invalid shell quoting") from exc
    if not argv:
        raise AgentError("command must not be empty")
    executable = Path(argv[0]).name.lower()
    if executable in FORBIDDEN_COMMANDS:
        raise AgentError(f"network command is forbidden: {executable}")
    if executable == "git" and len(argv) > 1 and argv[1] in FORBIDDEN_GIT_COMMANDS:
        raise AgentError(f"git {argv[1]} is forbidden")
    if executable in {"npm", "pnpm", "yarn", "pip", "pip3"} and len(argv) > 1:
        if argv[1] in FORBIDDEN_INSTALL_COMMANDS:
            raise AgentError(f"package command is forbidden: {executable} {argv[1]}")
    return argv


def _command_allowed(
    argv: tuple[str, ...], allowed_commands: tuple[tuple[str, ...], ...]
) -> bool:
    return argv in allowed_commands


def _safe_environment() -> dict[str, str]:
    path = os.environ.get("PATH", "")
    return {"PATH": path, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


def _task_id() -> str:
    return uuid.uuid4().hex[:12]


def _sandbox_basename(sandbox: str | Path) -> str:
    name = Path(sandbox).name
    return name or "sandbox"


def _safe_metadata(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())[:160]
    return str(value)


def _diagnostic(
    task_id: str,
    sandbox_name: str,
    stage: str,
    status: str,
    **metadata: Any,
) -> None:
    fields = [
        f"task_id={_safe_metadata(task_id)}",
        f"sandbox={_safe_metadata(sandbox_name)}",
        f"stage={stage}",
        f"status={status}",
    ]
    for name, value in metadata.items():
        if value is not None:
            fields.append(f"{name}={_safe_metadata(value)}")
    print("[qwen-agent] " + " ".join(fields), file=sys.stderr, flush=True)


def _error_kind(exc: BaseException, *, stage: str) -> str:
    message = str(exc).lower()
    if "endpoint_not_configured" in message:
        return "endpoint_not_configured"
    if "timeout" in message:
        return "timeout"
    if "http_error" in message or " status=" in message:
        return "http_error"
    if "connect" in message or "unreachable" in message:
        return "endpoint_unreachable"
    if "not valid json" in message:
        return "invalid_json"
    if "action limit" in message:
        return "action_limit"
    if stage == "ACTION_VALIDATE":
        return "action_invalid"
    return type(exc).__name__.lower()


def _validate_task_text(task: str) -> str:
    if not isinstance(task, str) or not task.strip():
        raise AgentError("task must not be empty")
    encoded = task.encode("utf-8")
    if len(encoded) > MAX_TASK_BYTES or len(task) > MAX_TASK_CHARS:
        raise AgentError("task exceeds the bounded task size")
    return task


def parse_action(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise AgentError("model returned an empty action")
    if len(text) > MAX_ACTION_TEXT_CHARS:
        raise AgentError("model action exceeds size limit")
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"}:
            raise AgentError("model action fence is not valid JSON")
        if lines[-1].strip() != "```":
            raise AgentError("model action fence is not closed")
        candidate = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AgentError("model action is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AgentError("model action must be a JSON object")
    if not isinstance(value.get("action"), str):
        raise AgentError("model action is missing an action name")
    return value


async def _parse_agent_stream(
    response: httpx.Response,
    *,
    started: float,
    attempts: int,
    http_status: int,
) -> AgentGeneration:
    final_parts: list[str] = []
    reasoning_chars = 0
    completion_tokens: int | None = None
    finish_reason: Any = None
    event_count = 0
    try:
        async for line in response.aiter_lines():
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            event_count += 1
            try:
                payload = httpx.Response(200, text=data).json()
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    "AGENT_GENERATION_MALFORMED: invalid SSE JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    "AGENT_GENERATION_MALFORMED: event is not an object"
                )
            usage = payload.get("usage")
            if isinstance(usage, dict) and isinstance(
                usage.get("completion_tokens"), int
            ):
                completion_tokens = usage["completion_tokens"]
            choices = payload.get("choices")
            if choices == []:
                continue
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
            ):
                raise RuntimeError(
                    "AGENT_GENERATION_MALFORMED: missing streamed choice"
                )
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise RuntimeError("AGENT_GENERATION_MALFORMED: missing delta")
            if "content" in delta:
                final_parts.append(_stream_text(delta["content"], "delta.content"))
            for alternate in ("text", "output_text"):
                if alternate in delta:
                    final_parts.append(
                        _stream_text(delta[alternate], f"delta.{alternate}")
                    )
            if "reasoning_content" in delta:
                reasoning_chars += len(
                    _stream_text(delta["reasoning_content"], "delta.reasoning_content")
                )
    except httpx.ReadTimeout as exc:
        raise RuntimeError("AGENT_GENERATION_TIMEOUT") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"AGENT_GENERATION_STREAM: {type(exc).__name__}: {exc}"
        ) from exc

    elapsed_ms = _elapsed_ms(started)
    stats = _metadata(
        attempts=attempts,
        events=event_count,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        reasoning_content_chars=reasoning_chars,
        elapsed_ms=elapsed_ms,
    )
    if finish_reason == "length":
        raise RuntimeError("GENERATION_LENGTH " + stats)
    final_text = "".join(final_parts).strip()
    if not final_text:
        raise RuntimeError("AGENT_GENERATION_EMPTY_FINAL " + stats)
    return AgentGeneration(
        final_text,
        completion_tokens,
        reasoning_chars,
        elapsed_ms,
        response_chars=len(final_text),
        finish_reason=finish_reason,
        http_status=http_status,
    )


async def _generate_agent_turn(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    sleep: Any = asyncio.sleep,
) -> AgentGeneration:
    started = time.monotonic()
    response, attempts = await _request_with_connect_retries(
        client,
        "POST",
        f"{base_url}/chat/completions",
        stage="AGENT_GENERATION_CONNECT",
        stream=True,
        json={
            "model": model,
            "messages": messages,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "presence_penalty": PRESENCE_PENALTY,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        sleep=sleep,
    )
    try:
        if not 200 <= response.status_code < 300:
            try:
                await response.aread()
            except httpx.HTTPError:
                pass
            raise RuntimeError(
                f"AGENT_GENERATION_HTTP attempts={attempts} "
                f"status={response.status_code}"
            )
        return await _parse_agent_stream(
            response,
            started=started,
            attempts=attempts,
            http_status=response.status_code,
        )
    finally:
        await response.aclose()


def _protocol_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


async def run_qwen_agent(
    task: str,
    sandbox: str | Path,
    allow: list[str],
    *,
    read_only: list[str] | None = None,
    allowed_commands: list[str] | None = None,
    validation_failure: str | None = None,
    max_actions: int = MAX_AGENT_ACTIONS,
) -> AgentResult:
    task_id = _task_id()
    sandbox_name = _sandbox_basename(sandbox)
    started = time.monotonic()
    current_stage = "PRECHECK"
    model_calls = 0
    completion_tokens = 0
    reasoning_chars = 0
    protocol_errors = 0
    agent: SandboxAgent | None = None

    _diagnostic(task_id, sandbox_name, "PRECHECK", "start")
    try:
        _validate_task_text(task)
        if isinstance(max_actions, bool) or not isinstance(max_actions, int):
            raise AgentError("max_actions must be an integer")
        if not 1 <= max_actions <= 100:
            raise AgentError("max_actions must be between 1 and 100")

        agent = SandboxAgent(
            sandbox,
            allow,
            read_only=read_only or (),
            allowed_commands=allowed_commands or (),
        )
        _diagnostic(
            task_id,
            sandbox_name,
            "PRECHECK",
            "ok",
            writable_scopes=len(allow),
            read_only_scopes=len(read_only or ()),
        )

        current_stage = "ENDPOINT_CONNECT"
        _diagnostic(task_id, sandbox_name, current_stage, "start")
        base_url = _configured_base_url()
        model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
        prompt = (
            "TASK:\n"
            + task
            + "\n\nSANDBOX:\n"
            + str(agent.root)
            + "\n\nWRITABLE ALLOWLIST:\n"
            + "\n".join(f"- {value}" for value in allow)
        )
        if read_only:
            prompt += "\n\nREAD-ONLY CONTEXT:\n" + "\n".join(
                f"- {value}" for value in read_only
            )
        if allowed_commands:
            prompt += "\n\nALLOWED COMMANDS:\n" + "\n".join(
                f"- {value}" for value in allowed_commands
            )
        if validation_failure:
            prompt += "\n\nCURRENT VALIDATION FAILURE:\n" + validation_failure.strip()
        prompt += "\n\nPerform the task with JSON actions. Do not output a diff."

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            preflight_attempts = await _preflight(client, base_url, model)
            _diagnostic(
                task_id,
                sandbox_name,
                current_stage,
                "ok",
                attempts=preflight_attempts,
            )
            for action_number in range(1, max_actions + 1):
                model_calls += 1
                current_stage = "MODEL_REQUEST"
                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "start",
                    model_call=model_calls,
                )
                generation = await _generate_agent_turn(
                    client, base_url, model, messages
                )
                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "ok",
                    model_call=model_calls,
                    elapsed_ms=generation.elapsed_ms,
                )
                completion_tokens += generation.completion_tokens or 0
                reasoning_chars += generation.reasoning_chars
                current_stage = "MODEL_RESPONSE"
                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "ok",
                    model_call=model_calls,
                    completion_tokens=generation.completion_tokens,
                    finish_reason=generation.finish_reason,
                    response_chars=generation.response_chars,
                    http_status=generation.http_status,
                )

                current_stage = "ACTION_PARSE"
                try:
                    action = parse_action(generation.text)
                except AgentError as exc:
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        current_stage,
                        "error",
                        model_call=model_calls,
                        kind=_error_kind(exc, stage=current_stage),
                        response_chars=len(generation.text),
                    )
                    protocol_errors += 1
                    if protocol_errors > MAX_PROTOCOL_ERRORS:
                        raise RuntimeError(f"AGENT_PROTOCOL_FAILED: {exc}") from exc
                    result = {"ok": False, "error": str(exc)}
                    messages.extend(
                        [
                            {"role": "assistant", "content": generation.text},
                            {"role": "user", "content": _protocol_result(result)},
                        ]
                    )
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        "MODEL_FEEDBACK",
                        "sent",
                        model_call=model_calls,
                        feedback_chars=len(_protocol_result(result)),
                    )
                    continue

                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "ok",
                    model_call=model_calls,
                    action_type=action.get("action"),
                )
                current_stage = "ACTION_VALIDATE"
                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "start",
                    model_call=model_calls,
                    action_type=action.get("action"),
                )
                try:
                    result = agent.execute(action)
                except AgentError as exc:
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        current_stage,
                        "error",
                        model_call=model_calls,
                        kind=_error_kind(exc, stage=current_stage),
                    )
                    protocol_errors += 1
                    if protocol_errors > MAX_PROTOCOL_ERRORS:
                        raise RuntimeError(f"AGENT_ACTION_FAILED: {exc}") from exc
                    result = {"ok": False, "error": str(exc)}
                else:
                    protocol_errors = 0
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        current_stage,
                        "ok",
                        model_call=model_calls,
                        action_type=action.get("action"),
                    )

                current_stage = "ACTION_EXECUTE"
                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "ok" if result.get("ok") is True else "error",
                    model_call=model_calls,
                    action_type=result.get("action", action.get("action")),
                )
                if result.get("action") == "done" and result.get("ok") is True:
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        "DONE",
                        "ok",
                        actions=action_number,
                        model_calls=model_calls,
                        completion_tokens=completion_tokens,
                        reasoning_chars=reasoning_chars,
                        read_bytes=agent.read_bytes,
                        write_bytes=agent.write_bytes,
                        elapsed_ms=_elapsed_ms(started),
                    )
                    return AgentResult(
                        status="done",
                        actions=action_number,
                        files_changed=sorted(agent.changed_paths),
                        model_calls=model_calls,
                        completion_tokens=completion_tokens,
                        reasoning_chars=reasoning_chars,
                        elapsed_ms=_elapsed_ms(started),
                        read_bytes=agent.read_bytes,
                        write_bytes=agent.write_bytes,
                    )
                messages.extend(
                    [
                        {"role": "assistant", "content": generation.text},
                        {"role": "user", "content": _protocol_result(result)},
                    ]
                )
                _diagnostic(
                    task_id,
                    sandbox_name,
                    "MODEL_FEEDBACK",
                    "sent",
                    model_call=model_calls,
                    feedback_chars=len(_protocol_result(result)),
                )
        raise RuntimeError(f"AGENT_ACTION_LIMIT: {max_actions}")
    except Exception as exc:
        kind = _error_kind(exc, stage=current_stage)
        _diagnostic(
            task_id,
            sandbox_name,
            current_stage,
            "error",
            kind=kind,
        )
        _diagnostic(
            task_id,
            sandbox_name,
            "DONE",
            "error",
            failed_stage=current_stage,
            kind=kind,
            model_calls=model_calls,
            completion_tokens=completion_tokens,
            reasoning_chars=reasoning_chars,
            read_bytes=agent.read_bytes if agent is not None else 0,
            write_bytes=agent.write_bytes if agent is not None else 0,
            elapsed_ms=_elapsed_ms(started),
        )
        raise
