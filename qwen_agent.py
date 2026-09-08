#!/usr/bin/env python3
"""Sandbox filesystem agent for the local Qwen worker."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, cast

import httpx

from qwen_input_limits import (
    MAX_ALLOW_SCOPES,
    MAX_ALLOWED_COMMANDS,
    MAX_COMMAND_BYTES,
    MAX_COMMAND_CHARS,
    MAX_MODEL_IDENTIFIER_BYTES,
    MAX_MODEL_IDENTIFIER_CHARS,
    MAX_PATH_BYTES,
    MAX_PATH_CHARS,
    MAX_READ_ONLY_SCOPES,
    MAX_TASK_BYTES,
    MAX_TASK_CHARS,
    MAX_VALIDATION_COMMANDS,
    MAX_VALIDATION_FAILURE_BYTES,
    MAX_VALIDATION_FAILURE_CHARS,
    InputLimitError,
    ensure_model_input,
    ensure_request_budget,
    read_bounded_file,
    validate_string_list,
    validate_text,
)
from qwen_worker_server import (
    DEFAULT_MODEL,
    HTTP_TIMEOUT,
    MAX_AGENT_FINAL_CONTENT_BYTES,
    MAX_AGENT_FINAL_CONTENT_CHARS,
    MAX_GENERATION_SECONDS,
    MAX_OUTPUT_TOKENS,
    MAX_REASONING_CONTENT_BYTES,
    MAX_REASONING_CONTENT_CHARS,
    MAX_SSE_EVENT_BYTES,
    MAX_STREAM_BYTES,
    MAX_STREAM_EVENTS,
    PRESENCE_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    BoundedSSEParser,
    StreamLimitError,
    StreamLimits,
    _bounded_stream_text,
    _configured_base_url,
    _elapsed_ms,
    _metadata,
    _preflight,
    _read_bounded_response_excerpt,
    _retry_telemetry,
    _start_generation_request,
    _transport_error_kind,
)

MAX_AGENT_ACTIONS = 30
MAX_PROTOCOL_ERRORS = 3
MAX_READ_BYTES = 64_000
MAX_READ_CHARS = 32_000
MAX_GREP_FILES = 500
MAX_GREP_MATCHES = 100
MAX_GREP_RESPONSE_CHARS = 20_000
MAX_GREP_RESPONSE_BYTES = 80_000
MAX_GREP_FILE_BYTES = 64_000
MAX_WRITE_BYTES = 1_000_000
MAX_ACTION_TEXT_CHARS = MAX_AGENT_FINAL_CONTENT_CHARS
MAX_ACTION_TEXT_BYTES = MAX_AGENT_FINAL_CONTENT_BYTES
MAX_COMMAND_OUTPUT_CHARS = 8_000
COMMAND_TIMEOUT_SECONDS = 120
MAX_COMMAND_TRACKED_FILES = 1_024
MAX_COMMAND_TRACKED_BYTES = 8_000_000
# These limits count Python characters in the model messages, including the
# system prompt, stable base context, compact state, and recent interactions.
MAX_AGENT_HISTORY_CHARS = 200_000
MAX_AGENT_BASE_CONTEXT_CHARS = 144_000
MAX_AGENT_STATE_SUMMARY_CHARS = 8_000
MAX_AGENT_RECENT_CHARS = 44_000
MAX_AGENT_RECENT_INTERACTIONS = 6
MAX_AGENT_STATE_EVENTS = 100
MAX_AGENT_VALIDATION_FAILURE_CHARS = 8_000
MAX_AGENT_CONTEXT_LIST_CHARS = 4_000
MAX_VALIDATION_ROUNDS = 3
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
edits. Write is primarily for creating a new file; an existing non-empty file
may be rewritten only after that exact file has had a complete explicit read
and has not changed since that read. A grep never counts as a complete read.
A replace.old string must be copied exactly from previously read file content.
Successful replace and write actions invalidate prior full-read proof, so read
the file again before a later full overwrite. After an action-validation error,
use the returned validation error to correct the next action; do not repeat the
same invalid action unchanged. The done action must always include a non-empty
summary. Stop with done when the filesystem state satisfies the task. Done
only means that you have finished editing; the host may run required validation
commands afterward, and a model-run command is not proof that the task is
validated.
"""

AGENT_RESPONSE_FORMAT = {
    "type": "json_object",
    "schema": {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read"]},
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["action", "path"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["grep"]},
                    "query": {"type": "string"},
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["action", "query"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["replace"]},
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["action", "path", "old", "new"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["write"]},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["action", "path", "content"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["done"]},
                    "summary": {"type": "string"},
                },
                "required": ["action", "summary"],
                "additionalProperties": False,
            },
        ],
    },
}

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
    stream_bytes: int = 0
    event_count: int = 0
    final_content_bytes: int = 0
    reasoning_content_bytes: int = 0


@dataclass(frozen=True)
class AgentInteraction:
    assistant_content: str
    result_content: str
    metadata: str

    @property
    def chars(self) -> int:
        return len(self.assistant_content) + len(self.result_content)


@dataclass(frozen=True)
class _FileReadSnapshot:
    text: str
    data: bytes
    sha256: str


def _truncate_context(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    marker = "\n[context truncated]\n"
    if limit <= len(marker):
        return marker[:limit]
    remaining = limit - len(marker)
    left = (remaining + 1) // 2
    right = remaining // 2
    suffix = value[-right:] if right else ""
    return value[:left] + marker + suffix


def _bounded_context_list(values: Iterable[str], limit: int) -> str:
    rendered = "\n".join(f"- {value}" for value in values) or "(none)"
    return _truncate_context(rendered, limit)


def _full_context_list(values: Iterable[str]) -> str:
    return "\n".join(f"- {value}" for value in values) or "(none)"


def _context_scalar(value: Any, limit: int = 240) -> str:
    if value is None:
        return "<none>"
    return _truncate_context(" ".join(str(value).split()), limit)


def _build_agent_base_prompt(
    task: str,
    allow: Iterable[str],
    read_only: Iterable[str],
    allowed_commands: Iterable[str],
    validation_failure: str | None,
    validation_commands: Iterable[str] = (),
) -> str:
    task = _validate_task_text(task)
    allow = validate_string_list(
        list(allow),
        field="allow",
        max_items=MAX_ALLOW_SCOPES,
        count_code="TOO_MANY_ALLOW_SCOPES",
        item_kind="path",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
    )
    read_only = validate_string_list(
        list(read_only),
        field="read_only",
        max_items=MAX_READ_ONLY_SCOPES,
        count_code="TOO_MANY_READ_ONLY_SCOPES",
        item_kind="path",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
    )
    allowed_commands = validate_string_list(
        list(allowed_commands),
        field="allowed_commands",
        max_items=MAX_ALLOWED_COMMANDS,
        count_code="TOO_MANY_COMMANDS",
        item_kind="command",
        max_bytes=MAX_COMMAND_BYTES,
        max_chars=MAX_COMMAND_CHARS,
    )
    validation_commands = validate_string_list(
        list(validation_commands),
        field="validation_commands",
        max_items=MAX_VALIDATION_COMMANDS,
        count_code="TOO_MANY_VALIDATION_COMMANDS",
        item_kind="command",
        max_bytes=MAX_COMMAND_BYTES,
        max_chars=MAX_COMMAND_CHARS,
    )
    if validation_failure is not None:
        validation_failure = validate_text(
            validation_failure,
            field="validation_failure",
            max_bytes=MAX_VALIDATION_FAILURE_BYTES,
            max_chars=MAX_VALIDATION_FAILURE_CHARS,
            code="VALIDATION_FAILURE_TOO_LARGE",
        )
    sections = [
        "TASK:",
        task,
        "",
        "SANDBOX: isolated workspace; all paths are sandbox-relative",
        "",
        "WRITABLE ALLOWLIST:",
        _full_context_list(sorted(allow)),
    ]
    for title, values in (
        ("READ-ONLY CONTEXT:", sorted(read_only)),
        ("ALLOWED COMMANDS:", sorted(allowed_commands)),
        ("REQUIRED HOST VALIDATION COMMANDS:", sorted(validation_commands)),
    ):
        if values:
            sections.extend(
                ["", title, _full_context_list(values)]
            )
    if validation_failure:
        sections.extend(
            [
                "",
                "INITIAL EXTERNAL VALIDATION FAILURE (CURRENT VALIDATION FAILURE INPUT):",
                _truncate_context(
                    validation_failure.strip(), MAX_AGENT_VALIDATION_FAILURE_CHARS
                ),
            ]
        )
    sections.extend(["", "Perform the task with JSON actions. Do not output a diff."])
    prompt = "\n".join(sections)
    if len(prompt) > MAX_AGENT_BASE_CONTEXT_CHARS:
        raise InputLimitError(
            "MODEL_INPUT_TOO_LARGE",
            "agent_base_context",
            len(prompt),
            MAX_AGENT_BASE_CONTEXT_CHARS,
            "chars",
        )
    return ensure_model_input(prompt, field="agent_base_context")


def _interaction_metadata(
    action: dict[str, Any] | None, result: dict[str, Any]
) -> str:
    if action is None:
        return f"protocol_error error={_context_scalar(result.get('error'))}"

    action_name = action.get("action", "unknown")
    path = result.get("path", action.get("path"))
    if action_name == "read":
        content = result.get("content")
        return f"read path={_context_scalar(path)} chars={len(content) if isinstance(content, str) else 0}"
    if action_name == "grep":
        matches = result.get("matches")
        files_scanned = result.get("files_scanned")
        files_skipped = result.get("files_skipped")
        truncation_reason = result.get("truncation_reason")
        return (
            f"grep paths={_context_scalar(action.get('paths', []))} query_chars="
            f"{len(action.get('query', '')) if isinstance(action.get('query'), str) else 0} "
            f"matches={len(matches) if isinstance(matches, list) else 0} "
            f"files_scanned={files_scanned} files_skipped={files_skipped} "
            f"truncated={bool(result.get('truncated'))} "
            f"truncation_reason={_context_scalar(truncation_reason)}"
        )
    if action_name == "run":
        return (
            f"run command={_context_scalar(action.get('command'))} "
            f"returncode={result.get('returncode')} "
            f"stdout_chars={len(result.get('stdout', '')) if isinstance(result.get('stdout'), str) else 0} "
            f"stderr_chars={len(result.get('stderr', '')) if isinstance(result.get('stderr'), str) else 0}"
        )
    if action_name == "validation":
        return (
            f"validation command={_context_scalar(result.get('command'))} "
            f"returncode={result.get('returncode')} "
            f"stdout_chars={len(result.get('stdout', '')) if isinstance(result.get('stdout'), str) else 0} "
            f"stderr_chars={len(result.get('stderr', '')) if isinstance(result.get('stderr'), str) else 0}"
        )
    if action_name == "replace":
        return f"replace path={_context_scalar(path)} ok={bool(result.get('ok'))} replaced={result.get('replaced', 0)}"
    if action_name == "write":
        return f"write path={_context_scalar(path)} ok={bool(result.get('ok'))} bytes={result.get('bytes', 0)}"
    return (
        f"action={_context_scalar(action_name)} ok={bool(result.get('ok'))} "
        f"error={_context_scalar(result.get('error'))}"
    )


def _compact_assistant_action(content: str) -> str:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return '{"action":"invalid"}'
    if not isinstance(value, dict):
        return '{"action":"invalid"}'
    return json.dumps(
        {"action": value.get("action", "unknown")},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _compact_result_json(content: str, limit: int) -> str:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return json.dumps(
            {"ok": False, "context_truncated": True},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if not isinstance(value, dict):
        value = {"ok": False, "context_truncated": True}
    compact = dict(value)
    compact["context_truncated"] = True
    large_fields = ("content", "matches", "stdout", "stderr", "error")
    for field in large_fields:
        field_value = compact.get(field)
        if isinstance(field_value, str):
            compact[field] = _truncate_context(field_value, max(0, limit // 2))
        elif isinstance(field_value, list):
            compact[field] = field_value[: max(0, limit // 200)]
    encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= limit:
        return encoded
    for field in large_fields:
        compact.pop(field, None)
        encoded = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= limit:
            return encoded
    return json.dumps(
        {
            key: compact[key]
            for key in ("ok", "action", "path", "returncode", "context_truncated")
            if key in compact
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class AgentConversation:
    """Bounded model context; authorization remains in SandboxAgent."""

    def __init__(
        self,
        task: str,
        allow: Iterable[str],
        read_only: Iterable[str],
        allowed_commands: Iterable[str],
        validation_failure: str | None,
        validation_commands: Iterable[str] = (),
    ) -> None:
        self.base_prompt = _build_agent_base_prompt(
            task,
            allow,
            read_only,
            allowed_commands,
            validation_failure,
            validation_commands,
        )
        self._recent: list[AgentInteraction] = []
        self._state_events: list[str] = []
        self._files_read: tuple[str, ...] = ()
        self._files_changed: tuple[str, ...] = ()
        self._actions_completed = 0
        self._protocol_errors = 0
        self._last_command = "(none)"
        self._dropped_interactions = 0
        self._history_compacted = False

    def record(
        self,
        assistant_content: str,
        action: dict[str, Any] | None,
        result: dict[str, Any],
        *,
        files_read: Iterable[str],
        files_changed: Iterable[str],
        actions_completed: int,
        protocol_errors: int,
    ) -> None:
        result_content = _protocol_result(result)
        metadata = _interaction_metadata(action, result)
        self._recent.append(
            AgentInteraction(
                assistant_content,
                result_content,
                metadata,
            )
        )
        self._state_events.append(metadata)
        if action is not None and action.get("action") == "run":
            self._last_command = metadata
        self._files_read = tuple(sorted(set(files_read)))
        self._files_changed = tuple(sorted(set(files_changed)))
        self._actions_completed = actions_completed
        self._protocol_errors = protocol_errors
        self._trim_state_events()
        self._trim_recent()

    def _trim_state_events(self) -> None:
        while (
            len(self._state_events) > MAX_AGENT_STATE_EVENTS
            or len("\n".join(self._state_events)) > MAX_AGENT_STATE_SUMMARY_CHARS
        ):
            self._state_events.pop(0)

    def _fit_last_interaction(self, interaction: AgentInteraction) -> AgentInteraction:
        assistant = _compact_assistant_action(interaction.assistant_content)
        available = max(1, MAX_AGENT_RECENT_CHARS - len(assistant))
        result = _compact_result_json(interaction.result_content, available)
        return AgentInteraction(assistant, result, interaction.metadata)

    def _trim_recent(self) -> None:
        while len(self._recent) > MAX_AGENT_RECENT_INTERACTIONS:
            self._recent.pop(0)
            self._dropped_interactions += 1
            self._history_compacted = True
        while sum(item.chars for item in self._recent) > MAX_AGENT_RECENT_CHARS:
            if len(self._recent) > 1:
                self._recent.pop(0)
                self._dropped_interactions += 1
                self._history_compacted = True
                continue
            fitted = self._fit_last_interaction(self._recent[0])
            self._recent[0] = fitted
            self._history_compacted = True
            break

    def _state_prompt(self) -> str:
        return _truncate_context(
            "\n".join(
                [
                    "COMPACT STATE (metadata only; authorization remains host-side):",
                    f"actions_completed={self._actions_completed}",
                    f"protocol_errors={self._protocol_errors}",
                    f"last_command={self._last_command}",
                    "last_interaction:",
                    self._state_events[-1] if self._state_events else "(none)",
                    "fully_observed_files:",
                    _bounded_context_list(
                        self._files_read, MAX_AGENT_CONTEXT_LIST_CHARS
                    ),
                    "files_changed:",
                    _bounded_context_list(
                        self._files_changed, MAX_AGENT_CONTEXT_LIST_CHARS
                    ),
                    "older_interaction_metadata:",
                    _bounded_context_list(
                        self._state_events, MAX_AGENT_STATE_SUMMARY_CHARS
                    ),
                ]
            ),
            MAX_AGENT_STATE_SUMMARY_CHARS,
        )

    def snapshot(self) -> tuple[list[dict[str, str]], dict[str, int | bool]]:
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.base_prompt},
            {"role": "user", "content": self._state_prompt()},
        ]
        for interaction in self._recent:
            messages.extend(
                [
                    {"role": "assistant", "content": interaction.assistant_content},
                    {"role": "user", "content": interaction.result_content},
                ]
            )
        history_chars = sum(len(message["content"]) for message in messages)
        if history_chars > MAX_AGENT_HISTORY_CHARS:
            raise AgentError("agent history budget enforcement failed")
        ensure_model_input(
            "\n".join(message["content"] for message in messages),
            field="agent_model_input",
        )
        return messages, {
            "history_interactions": len(self._recent),
            "history_chars": history_chars,
            "history_compacted": self._history_compacted,
            "history_dropped_interactions": self._dropped_interactions,
        }


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CommandPathScope:
    """A canonical project path granted to a command sandbox."""

    path: Path
    is_directory: bool


@dataclass(frozen=True)
class CommandAccessPolicy:
    """Explicit project and scratch authority for one command invocation."""

    readable_paths: tuple[CommandPathScope, ...]
    writable_paths: tuple[CommandPathScope, ...]
    scratch_path: Path
    validation: bool


@dataclass(frozen=True)
class _CommandPathSnapshot:
    kind: str
    digest: str | None = None


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
    validated: bool = False
    validation_commands: int = 0
    validation_failures: int = 0
    validation_rounds: int = 0

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
            "validated": self.validated,
            "validation_commands": self.validation_commands,
            "validation_failures": self.validation_failures,
            "validation_rounds": self.validation_rounds,
        }


def _relative_path(value: Any, *, field: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AgentError(f"{field} must be a non-empty relative path")
    try:
        validate_text(
            value,
            field=field,
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
            code="PATH_TOO_LONG",
        )
    except InputLimitError as exc:
        raise AgentError(str(exc)) from exc
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


def _canonical_relative(
    root: Path,
    resolved: Path,
    *,
    original: str,
    reject_forbidden: bool = True,
) -> str:
    if not _within(root, resolved):
        raise AgentError(f"path escapes sandbox: {original}")
    canonical = resolved.relative_to(root).as_posix()
    if reject_forbidden and _is_forbidden_path(canonical):
        raise AgentError(f"forbidden path: {original}")
    return canonical


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
    canonical = _canonical_relative(root, resolved, original=relative)
    # Scopes are canonical so a lexical path inside an allowed directory
    # cannot use an intra-sandbox symlink to reach another directory.
    return canonical, is_directory


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
        self._writable_command_scopes = self._initial_command_scope_paths(
            self.writable_scopes
        )
        self._readable_command_scopes = self._initial_command_scope_paths(
            self.readable_scopes
        )
        self.allowed_commands = tuple(
            self._parse_allowed_command(command) for command in allowed_commands
        )
        # This is authorization state, not conversation state.  The digest is
        # of the raw bytes returned by a complete explicit read.
        self.fully_observed_files: dict[str, str] = {}
        self.changed_paths: set[str] = set()
        self.read_bytes = 0
        self.write_bytes = 0
        self.mutation_generation = 0

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
        candidate = self.root / relative
        try:
            if must_exist:
                resolved = candidate.resolve(strict=True)
            else:
                try:
                    resolved = candidate.resolve(strict=True)
                except FileNotFoundError:
                    resolved = candidate.parent.resolve(strict=True) / candidate.name
        except (FileNotFoundError, RuntimeError) as exc:
            raise AgentError(f"invalid sandbox path: {relative}") from exc
        canonical = _canonical_relative(
            self.root, resolved, original=relative
        )
        if not self._scope_allows(canonical, scopes):
            scope_name = "writable" if writable else "readable"
            raise AgentError(
                f"resolved path is not in the {scope_name} allowlist: {relative}"
            )
        if must_exist and not resolved.exists():
            raise AgentError(f"path does not exist: {relative}")
        return relative, resolved

    def _canonical_identity(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise AgentError(f"path escapes sandbox: {path}") from exc

    def _read_file(
        self,
        relative: str,
        path: Path,
        *,
        max_bytes: int = MAX_READ_BYTES,
        limit_label: str = "read",
    ) -> _FileReadSnapshot:
        if not path.is_file():
            raise AgentError(f"path is not a regular file: {relative}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise AgentError(f"could not stat file: {relative}") from exc
        if size > max_bytes:
            raise AgentError(f"file exceeds {limit_label} limit: {relative}")
        try:
            data = read_bounded_file(
                path,
                max_bytes,
                field="sandbox file",
                code="FILE_TOO_LARGE",
            )
        except InputLimitError as exc:
            raise AgentError(str(exc)) from exc
        except OSError as exc:
            raise AgentError(f"could not read UTF-8 file: {relative}") from exc
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AgentError(f"could not read UTF-8 file: {relative}") from exc
        if b"\x00" in data:
            raise AgentError(f"binary file rejected: {relative}")
        self.read_bytes += len(data)
        return _FileReadSnapshot(
            text=text,
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def _read_text(self, relative: str, path: Path) -> str:
        """Read text without changing full-observation authorization state."""
        return self._read_file(relative, path).text

    def _invalidate_all_observations(self) -> None:
        # Option A: every successful mutation invalidates every observation.
        # This is deliberately stricter than refreshing the edited file only.
        self.fully_observed_files.clear()

    def _current_sha256(self, relative: str, path: Path) -> str:
        try:
            if not path.is_file():
                raise OSError("not a regular file")
            if path.stat().st_size > MAX_READ_BYTES:
                raise OSError("file exceeds observation size")
            data = read_bounded_file(
                path,
                MAX_READ_BYTES,
                field="sandbox observation",
                code="FILE_TOO_LARGE",
            )
        except InputLimitError as exc:
            raise OSError("file exceeds observation size") from exc
        except OSError as exc:
            raise AgentError(f"file changed since it was read: {relative}") from exc
        return hashlib.sha256(data).hexdigest()

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
        self._invalidate_all_observations()
        self.changed_paths.add(relative)
        self.write_bytes += len(data)
        self.mutation_generation += 1

    def _read_action(self, action: dict[str, Any]) -> dict[str, Any]:
        relative, path = self._resolve(
            action.get("path"), readable=True, must_exist=True
        )
        canonical = self._canonical_identity(path)
        snapshot = self._read_file(relative, path)
        text = snapshot.text
        start = action.get("start_line", 1)
        end = action.get("end_line")
        if isinstance(start, bool) or not isinstance(start, int) or start < 1:
            raise AgentError("start_line must be a positive integer")
        if end is not None and (
            isinstance(end, bool) or not isinstance(end, int) or end < start
        ):
            raise AgentError("end_line must be an integer after start_line")
        if start == 1 and end is None:
            content = text
        else:
            lines = text.splitlines()
            selected = lines[start - 1 : end]
            content = "\n".join(selected)
            if selected and text.endswith("\n") and end is None:
                content += "\n"
        complete = True
        if len(content.encode("utf-8")) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS] + "\n[truncated]"
            complete = False
        if complete and start == 1 and end is None and content == text:
            self.fully_observed_files[canonical] = snapshot.sha256
        else:
            self.fully_observed_files.pop(canonical, None)
        return {
            "ok": True,
            "action": "read",
            "path": relative,
            "content": content,
        }

    @staticmethod
    def _grep_skip_reason(error: AgentError) -> str:
        message = str(error)
        if "exceeds grep file limit" in message:
            return "file_too_large"
        if "binary file rejected" in message or "could not read UTF-8" in message:
            return "binary_or_non_utf8"
        return "unreadable"

    @staticmethod
    def _grep_result(
        matches: list[str],
        files_scanned: int,
        files_skipped: int,
        *,
        truncated: bool,
        truncation_reason: str | None = None,
        skipped_reasons: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": True,
            "action": "grep",
            "matches": matches,
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
            "truncated": truncated,
            "truncation_reason": truncation_reason,
        }
        if skipped_reasons:
            result["skipped_reasons"] = {
                key: skipped_reasons[key] for key in sorted(skipped_reasons)
            }
        return result

    def _iter_grep_directory(self, directory: Path) -> Iterable[Path]:
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise AgentError(f"grep directory could not be scanned: {directory}") from exc
        for candidate in children:
            try:
                if candidate.is_symlink():
                    yield candidate
                    continue
                if candidate.is_dir():
                    yield from self._iter_grep_directory(candidate)
                elif candidate.is_file():
                    yield candidate
            except OSError as exc:
                raise AgentError(
                    f"grep directory entry could not be inspected: {candidate}"
                ) from exc

    def _grep_candidates(
        self, requested_paths: list[str]
    ) -> tuple[list[tuple[str, Path]], bool, dict[str, int]]:
        files: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        skipped_reasons: dict[str, int] = {}
        discovered = 0
        file_limit_hit = False

        def skip(reason: str) -> None:
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        for requested in requested_paths:
            relative, base = self._resolve(requested, readable=True, must_exist=True)
            if base.is_file():
                candidates: Iterable[Path] = (base,)
            elif base.is_dir():
                candidates = self._iter_grep_directory(base)
            else:
                raise AgentError(f"grep path is not a file or directory: {relative}")

            for candidate in candidates:
                discovered += 1
                if discovered > MAX_GREP_FILES:
                    file_limit_hit = True
                    break
                try:
                    resolved = candidate.resolve(strict=True)
                except (FileNotFoundError, RuntimeError):
                    skip("invalid_symlink")
                    continue
                if not _within(self.root, resolved):
                    raise AgentError(f"grep encountered symlink escape: {candidate}")
                normalized = _canonical_relative(
                    self.root,
                    resolved,
                    original=str(candidate),
                    reject_forbidden=False,
                )
                if not self._scope_allows(normalized, self.readable_scopes):
                    skip("out_of_scope")
                    continue
                if _is_forbidden_path(normalized):
                    skip("forbidden")
                    continue
                if resolved in seen:
                    skip("duplicate")
                    continue
                seen.add(resolved)
                files.append((normalized, resolved))
            if file_limit_hit:
                break
        return files, file_limit_hit, skipped_reasons

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
        requested_paths = sorted(requested_paths)

        files, file_limit_hit, skipped_reasons = self._grep_candidates(requested_paths)
        matches: list[str] = []
        total_chars = 0
        total_bytes = 0
        files_scanned = 0
        for relative, path in files:
            try:
                snapshot = self._read_file(
                    relative,
                    path,
                    max_bytes=MAX_GREP_FILE_BYTES,
                    limit_label="grep file",
                )
            except AgentError as exc:
                reason = self._grep_skip_reason(exc)
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue
            files_scanned += 1
            for line_number, line in enumerate(snapshot.text.splitlines(), 1):
                if query not in line:
                    continue
                entry = f"{relative}:{line_number}:{line}"
                entry_chars = len(entry) + 1
                entry_bytes = len(entry.encode("utf-8")) + 1
                if (
                    total_chars + entry_chars > MAX_GREP_RESPONSE_CHARS
                    or total_bytes + entry_bytes > MAX_GREP_RESPONSE_BYTES
                ):
                    return self._grep_result(
                        matches,
                        files_scanned,
                        sum(skipped_reasons.values()),
                        truncated=True,
                        truncation_reason="response_limit",
                        skipped_reasons=skipped_reasons,
                    )
                matches.append(entry)
                total_chars += entry_chars
                total_bytes += entry_bytes
                if len(matches) >= MAX_GREP_MATCHES:
                    return self._grep_result(
                        matches,
                        files_scanned,
                        sum(skipped_reasons.values()),
                        truncated=True,
                        truncation_reason="match_limit",
                        skipped_reasons=skipped_reasons,
                    )
        if file_limit_hit:
            skipped_reasons["file_limit"] = skipped_reasons.get("file_limit", 0) + 1
        incomplete = bool(skipped_reasons)
        truncation_reason = (
            "file_limit" if file_limit_hit else ("file_skip" if incomplete else None)
        )
        return self._grep_result(
            matches,
            files_scanned,
            sum(skipped_reasons.values()),
            truncated=incomplete,
            truncation_reason=truncation_reason,
            skipped_reasons=skipped_reasons,
        )

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
        canonical = self._canonical_identity(path)
        exists = path.exists()
        if exists and not path.is_file():
            raise AgentError(f"path is not a regular file: {relative}")
        if exists:
            observed_sha256 = self.fully_observed_files.get(canonical)
            if observed_sha256 is not None:
                current_sha256 = self._current_sha256(relative, path)
                if current_sha256 != observed_sha256:
                    raise AgentError(f"file changed since it was read: {relative}")
            elif path.stat().st_size > 0:
                raise AgentError(f"read existing file before rewriting: {relative}")
        self._atomic_write(relative, path, content)
        return {
            "ok": True,
            "action": "write",
            "path": relative,
            "bytes": len(content.encode("utf-8")),
        }

    def _initial_command_scope_paths(
        self, scopes: tuple[tuple[str, bool], ...]
    ) -> tuple[CommandPathScope, ...]:
        command_scopes: list[CommandPathScope] = []
        seen: set[tuple[Path, bool]] = set()
        for relative, is_directory in scopes:
            try:
                resolved = (self.root / relative).resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise AgentError("command scope could not be resolved") from exc
            if not _within(self.root, resolved):
                raise AgentError("command scope escapes sandbox")
            scope = CommandPathScope(resolved, is_directory)
            key = (scope.path, scope.is_directory)
            if key not in seen:
                seen.add(key)
                command_scopes.append(scope)
        return tuple(command_scopes)

    @staticmethod
    def _command_scope_paths(
        scopes: tuple[CommandPathScope, ...]
    ) -> tuple[CommandPathScope, ...]:
        for scope in scopes:
            try:
                current = scope.path.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise AgentError("command scope could not be resolved") from exc
            if current != scope.path:
                raise AgentError("command scope changed since authorization")
        return scopes

    def _command_access_policy(
        self, scratch_path: Path, *, validation: bool
    ) -> CommandAccessPolicy:
        return CommandAccessPolicy(
            readable_paths=self._command_scope_paths(
                self._readable_command_scopes
            ),
            writable_paths=(
                ()
                if validation
                else self._command_scope_paths(self._writable_command_scopes)
            ),
            scratch_path=scratch_path,
            validation=validation,
        )

    def _snapshot_writable_state(self) -> dict[str, _CommandPathSnapshot]:
        snapshots: dict[str, _CommandPathSnapshot] = {}
        tracked_bytes = 0

        def record(relative: str, path: Path, *, recurse: bool) -> None:
            nonlocal tracked_bytes
            if relative in snapshots:
                return
            if len(snapshots) >= MAX_COMMAND_TRACKED_FILES:
                raise AgentError("command mutation snapshot exceeds file limit")
            try:
                observed = path.lstat()
            except FileNotFoundError:
                snapshots[relative] = _CommandPathSnapshot("missing")
                return
            except OSError as exc:
                raise AgentError("command mutation snapshot could not inspect path") from exc

            mode = stat.S_IFMT(observed.st_mode)
            permissions = stat.S_IMODE(observed.st_mode)
            if stat.S_ISREG(mode):
                if observed.st_size > MAX_COMMAND_TRACKED_BYTES - tracked_bytes:
                    raise AgentError("command mutation snapshot exceeds byte limit")
                try:
                    data = read_bounded_file(
                        path,
                        MAX_COMMAND_TRACKED_BYTES - tracked_bytes,
                        field="command mutation snapshot",
                        code="COMMAND_SNAPSHOT_TOO_LARGE",
                    )
                except (InputLimitError, OSError) as exc:
                    raise AgentError(
                        "command mutation snapshot exceeds byte limit"
                    ) from exc
                tracked_bytes += len(data)
                snapshots[relative] = _CommandPathSnapshot(
                    f"file:{permissions:o}", hashlib.sha256(data).hexdigest()
                )
                return
            if stat.S_ISLNK(mode):
                try:
                    target = os.readlink(path)
                except OSError as exc:
                    raise AgentError(
                        "command mutation snapshot could not inspect symlink"
                    ) from exc
                snapshots[relative] = _CommandPathSnapshot(
                    "symlink", hashlib.sha256(os.fsencode(target)).hexdigest()
                )
                return
            if stat.S_ISDIR(mode):
                snapshots[relative] = _CommandPathSnapshot(
                    f"directory:{permissions:o}"
                )
                if not recurse:
                    return
                try:
                    children = sorted(path.iterdir(), key=lambda child: child.name)
                except OSError as exc:
                    raise AgentError(
                        "command mutation snapshot could not scan directory"
                ) from exc
                for child in children:
                    record(f"{relative}/{child.name}", child, recurse=True)
                return
            snapshots[relative] = _CommandPathSnapshot(f"special:{mode:o}")

        for relative, is_directory in sorted(self.writable_scopes):
            record(relative, self.root / relative, recurse=is_directory)
        return snapshots

    def _record_command_mutations(
        self,
        before: dict[str, _CommandPathSnapshot],
        after: dict[str, _CommandPathSnapshot],
    ) -> None:
        changed = {
            relative
            for relative in set(before) | set(after)
            if before.get(relative) != after.get(relative)
        }
        if changed:
            self._invalidate_all_observations()
            self.changed_paths.update(changed)
            self.mutation_generation += 1

    def _run_command_argv(
        self, argv: tuple[str, ...], *, validation: bool
    ) -> dict[str, Any]:
        executable = _resolve_command_executable(argv[0], self.root)
        sandbox = _discover_command_sandbox()
        before = None if validation else self._snapshot_writable_state()
        try:
            with tempfile.TemporaryDirectory(prefix="qwen-command-") as directory:
                scratch_path = Path(directory).resolve(strict=True)
                access_policy = self._command_access_policy(
                    scratch_path, validation=validation
                )
                completed = sandbox.run(
                    (str(executable), *argv[1:]),
                    sandbox_root=self.root,
                    access_policy=access_policy,
                    environment=_safe_environment(scratch_path),
                    timeout=COMMAND_TIMEOUT_SECONDS,
                )
        finally:
            if before is not None:
                try:
                    self._record_command_mutations(
                        before, self._snapshot_writable_state()
                    )
                except AgentError:
                    # A post-command tracking failure must invalidate validation
                    # state instead of allowing an unobserved project mutation.
                    self._invalidate_all_observations()
                    self.mutation_generation += 1
                    raise
        return {
            "ok": completed.returncode == 0,
            "action": "run",
            "command": " ".join(argv),
            "returncode": completed.returncode,
            "stdout": _redact_command_scratch(
                completed.stdout, scratch_path
            )[-MAX_COMMAND_OUTPUT_CHARS:],
            "stderr": _redact_command_scratch(
                completed.stderr, scratch_path
            )[-MAX_COMMAND_OUTPUT_CHARS:],
        }

    def _run_action(self, action: dict[str, Any]) -> dict[str, Any]:
        command = action.get("command")
        argv = _parse_command(command)
        if not _command_allowed(argv, self.allowed_commands):
            raise AgentError(f"command is not allowlisted: {command}")
        return self._run_command_argv(argv, validation=False)

    def run_validation_command(self, argv: tuple[str, ...]) -> dict[str, Any]:
        if not _command_allowed(argv, self.allowed_commands):
            raise AgentError("validation command is not allowlisted")
        return self._run_command_argv(argv, validation=True)

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


class ValidationRunner:
    """Run required host validation through SandboxAgent's command backend."""

    def __init__(self, agent: SandboxAgent, policy: ValidationPolicy) -> None:
        self.agent = agent
        self.policy = policy

    def run(
        self,
        *,
        on_result: Callable[[int, ValidationResult], None] | None = None,
    ) -> tuple[ValidationResult, ...]:
        results: list[ValidationResult] = []
        for index, argv in enumerate(self.policy.commands, 1):
            command = shlex.join(argv)
            try:
                raw = self.agent.run_validation_command(argv)
                returncode = raw.get("returncode")
                if isinstance(returncode, bool) or not isinstance(returncode, int):
                    returncode = 1
                result = ValidationResult(
                    command,
                    returncode,
                    raw.get("stdout", ""),
                    raw.get("stderr", ""),
                )
            except AgentError as exc:
                result = ValidationResult(
                    command,
                    127,
                    "",
                    _truncate_context(
                        f"validation command execution failed: {exc}",
                        MAX_COMMAND_OUTPUT_CHARS,
                    ),
                )
            results.append(result)
            if on_result is not None:
                on_result(index, result)
            if not result.passed:
                break
        return tuple(results)


def _parse_command(command: Any) -> tuple[str, ...]:
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise AgentError("command must be a non-empty string")
    try:
        validate_text(
            command,
            field="command",
            max_bytes=MAX_COMMAND_BYTES,
            max_chars=MAX_COMMAND_CHARS,
            code="COMMAND_TOO_LONG",
        )
    except InputLimitError as exc:
        raise AgentError(str(exc)) from exc
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


class ValidationPhase(str, Enum):
    EDITING = "EDITING"
    MODEL_DONE = "MODEL_DONE"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ValidationResult:
    command: str
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stdout",
            _truncate_context(self.stdout if isinstance(self.stdout, str) else "", MAX_COMMAND_OUTPUT_CHARS),
        )
        object.__setattr__(
            self,
            "stderr",
            _truncate_context(self.stderr if isinstance(self.stderr, str) else "", MAX_COMMAND_OUTPUT_CHARS),
        )

    @property
    def passed(self) -> bool:
        return self.returncode == 0

    def feedback(self, validation_round: int) -> dict[str, Any]:
        return {
            "ok": False,
            "action": "validation",
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "validation_round": validation_round,
        }


@dataclass
class ValidationState:
    phase: ValidationPhase = ValidationPhase.EDITING
    validation_rounds: int = 0
    last_results: tuple[ValidationResult, ...] = ()
    mutation_generation: int = 0
    validated_generation: int | None = None

    def mark_mutation(self, generation: int) -> None:
        self.mutation_generation = generation
        self.phase = ValidationPhase.EDITING
        self.validated_generation = None

    def mark_model_done(self) -> None:
        self.phase = ValidationPhase.MODEL_DONE

    def prepare_retry(self) -> None:
        if self.phase == ValidationPhase.FAILED:
            self.phase = ValidationPhase.EDITING

    def start_round(self, limit: int) -> int:
        if self.validation_rounds >= limit:
            raise RuntimeError(f"AGENT_VALIDATION_LIMIT: {limit}")
        self.validation_rounds += 1
        self.phase = ValidationPhase.VALIDATING
        return self.validation_rounds

    def finish_round(
        self,
        results: Iterable[ValidationResult],
        generation: int,
    ) -> bool:
        self.last_results = tuple(results)
        self.mutation_generation = generation
        if self.last_results and all(result.passed for result in self.last_results):
            self.phase = ValidationPhase.VALIDATED
            self.validated_generation = generation
            return True
        self.phase = ValidationPhase.FAILED
        self.validated_generation = None
        return False

    @property
    def failure_count(self) -> int:
        return sum(not result.passed for result in self.last_results)


@dataclass(frozen=True)
class ValidationPolicy:
    commands: tuple[tuple[str, ...], ...]

    @classmethod
    def from_commands(
        cls,
        commands: Iterable[str] | None,
        allowed_commands: tuple[tuple[str, ...], ...],
    ) -> "ValidationPolicy":
        if commands is None:
            return cls(())
        if isinstance(commands, (str, bytes)):
            raise AgentError("validation_commands must be an array of strings")
        try:
            requested = tuple(commands)
        except TypeError as exc:
            raise AgentError("validation_commands must be an array of strings") from exc
        if len(requested) > MAX_VALIDATION_COMMANDS:
            raise InputLimitError(
                "TOO_MANY_VALIDATION_COMMANDS",
                "validation_commands",
                len(requested),
                MAX_VALIDATION_COMMANDS,
                "count",
            )
        allowed = set(allowed_commands)
        parsed: list[tuple[str, ...]] = []
        for command in requested:
            argv = _parse_command(command)
            if argv not in allowed:
                raise AgentError(
                    "validation command is not in the allowed command allowlist: "
                    + shlex.join(argv)
                )
            parsed.append(argv)
        return cls(tuple(parsed))

    @property
    def count(self) -> int:
        return len(self.commands)

    @property
    def display_commands(self) -> tuple[str, ...]:
        return tuple(shlex.join(argv) for argv in self.commands)


def _trusted_command_paths() -> tuple[Path, ...]:
    candidates: tuple[Path, ...] = (
        Path(sys.prefix) / "bin",
        Path("/usr/bin"),
        Path("/bin"),
        Path("/usr/sbin"),
        Path("/sbin"),
    )
    if sys.platform == "darwin":
        candidates += (Path("/opt/homebrew/bin"),)
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if not resolved.is_dir() or resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return tuple(paths)


def _safe_environment(scratch_path: Path | None = None) -> dict[str, str]:
    environment = {
        "PATH": os.pathsep.join(str(path) for path in _trusted_command_paths()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if scratch_path is not None:
        environment.update(
            {
                "HOME": str(scratch_path),
                "TMPDIR": str(scratch_path),
                "PYTHONPYCACHEPREFIX": str(scratch_path / "pycache"),
                "PYTHONNOUSERSITE": "1",
            }
        )
    return environment


def _redact_command_scratch(output: str, scratch_path: Path) -> str:
    return output.replace(str(scratch_path), "[command scratch]")


def _resolve_command_executable(argv0: str, sandbox_root: Path) -> Path:
    if "/" in argv0 or os.sep in argv0:
        candidate = Path(argv0)
        if not candidate.is_absolute():
            candidate = sandbox_root / candidate
    else:
        selected = shutil.which(
            argv0, path=_safe_environment()["PATH"]
        )
        if selected is None:
            raise AgentError("command executable was not found")
        candidate = Path(selected)
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise AgentError("command executable was not found") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise AgentError("command executable is not runnable")
    return resolved


def _sbpl_parameter(name: str) -> str:
    return f'(param "{name}")'


def _sbpl_regex(pattern: str) -> str:
    return "#" + json.dumps(pattern)


def _forbidden_command_path_regex(sandbox_root: Path) -> str:
    def insensitive(value: str) -> str:
        parts: list[str] = []
        for character in value:
            if character.isascii() and character.isalpha():
                parts.append(f"[{character.lower()}{character.upper()}]")
            else:
                parts.append(re.escape(character))
        return "".join(parts)

    root = str(sandbox_root).rstrip("/") or "/"
    separator = "" if root == "/" else "/"
    components = "|".join(
        [r"\.[eE][nN][vV][^/]*"]
        + [insensitive(name) for name in sorted(FORBIDDEN_BASENAMES)]
    )
    # The root is escaped before it becomes a profile regular expression; all
    # filename matching is static rather than derived from caller path input.
    return (
        f"^{re.escape(root)}{separator}([^/]+/)*({components})(/.*)?$"
    )


def _command_scope_predicate(
    parameter: str, scope: CommandPathScope
) -> str:
    operation = "subpath" if scope.is_directory else "literal"
    return f"({operation} {parameter})"


def _macos_runtime_paths(executable: Path) -> tuple[Path, ...]:
    candidates = [
        Path("/System"),
        Path("/usr/bin"),
        Path("/usr/sbin"),
        Path("/bin"),
        Path("/sbin"),
        Path("/usr/lib"),
        Path("/usr/share"),
        Path("/Library/Apple"),
        Path("/Library/Frameworks"),
        Path("/Library/PrivateFrameworks"),
        Path(sys.prefix),
    ]
    homebrew = Path("/opt/homebrew")
    worker_executable = Path(sys.executable).resolve(strict=False)
    if _within(homebrew, executable) or _within(homebrew, worker_executable):
        candidates.append(homebrew)
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if not resolved.is_dir() or resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return tuple(paths)


def _macos_sandbox_profile(
    sandbox_root: Path,
    executable: Path,
    access_policy: CommandAccessPolicy,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    parameters: list[tuple[str, str]] = [
        ("SANDBOX_ROOT", str(sandbox_root)),
        ("EXECUTABLE", str(executable)),
        ("SCRATCH", str(access_policy.scratch_path)),
    ]
    forbidden = _sbpl_regex(_forbidden_command_path_regex(sandbox_root))
    rules = [
        "(version 1)",
        "(deny default)",
        # Keep system.sb for runtime bootstrap only; do not import its optional
        # system-network rules so network access remains denied by default.
        '(import "system.sb")',
        "(allow process-fork)",
        "(allow signal)",
        f"(allow file-read-metadata file-test-existence (path-ancestors {_sbpl_parameter('SANDBOX_ROOT')}))",
        f"(allow file-read* file-map-executable (literal {_sbpl_parameter('EXECUTABLE')}))",
        f"(allow process-exec (literal {_sbpl_parameter('EXECUTABLE')}))",
        f"(allow file-read-metadata file-test-existence (path-ancestors {_sbpl_parameter('EXECUTABLE')}))",
        f"(allow file-read* file-map-executable (subpath {_sbpl_parameter('SCRATCH')}))",
        f"(allow file-write* (subpath {_sbpl_parameter('SCRATCH')}))",
        f"(allow file-read-metadata file-test-existence (path-ancestors {_sbpl_parameter('SCRATCH')}))",
    ]
    for access, scopes in (
        ("READABLE", access_policy.readable_paths),
        ("WRITABLE", access_policy.writable_paths),
    ):
        for index, scope in enumerate(scopes):
            parameter_name = f"{access}_{index}"
            parameters.append((parameter_name, str(scope.path)))
            parameter = _sbpl_parameter(parameter_name)
            predicate = _command_scope_predicate(parameter, scope)
            if access == "READABLE":
                rules.append(
                    f"(allow file-read* file-map-executable {predicate})"
                )
            else:
                rules.append(f"(allow file-write* {predicate})")
            rules.append(
                f"(allow file-read-metadata file-test-existence (path-ancestors {parameter}))"
            )
    for index, path in enumerate(_macos_runtime_paths(executable)):
        parameter_name = f"RUNTIME_{index}"
        parameters.append((parameter_name, str(path)))
        parameter = _sbpl_parameter(parameter_name)
        rules.extend(
            [
                f"(allow file-read* file-map-executable (subpath {parameter}))",
                f"(allow process-exec (subpath {parameter}))",
                f"(allow process-exec-interpreter (subpath {parameter}))",
                f"(allow file-read-metadata file-test-existence (path-ancestors {parameter}))",
            ]
        )
    # Seatbelt resolves overlapping rules in declaration order, so these
    # explicit secret denials must follow directory-level scope allowances.
    rules.extend(
        [
            f"(deny file-read* {forbidden})",
            f"(deny file-write* {forbidden})",
        ]
    )
    return "\n".join(rules), tuple(parameters)


def _tail_buffer(buffer: bytearray, data: bytes) -> None:
    buffer.extend(data)
    if len(buffer) > MAX_COMMAND_OUTPUT_CHARS:
        del buffer[:-MAX_COMMAND_OUTPUT_CHARS]


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _collect_process_output(
    process: subprocess.Popen[bytes], timeout: float
) -> CommandResult:
    streams = {
        stream: name
        for stream, name in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
        )
        if stream is not None
    }
    selector = selectors.DefaultSelector()
    for stream in streams:
        selector.register(stream, selectors.EVENT_READ, streams[stream])
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        while streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            for key, _ in selector.select(min(remaining, 0.1)):
                stream = cast(BinaryIO, key.fileobj)
                data = os.read(stream.fileno(), 8192)
                if not data:
                    selector.unregister(stream)
                    streams.pop(stream, None)
                    stream.close()
                    continue
                _tail_buffer(buffers[key.data], data)
        try:
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError from exc
    finally:
        selector.close()
    return CommandResult(
        returncode=returncode,
        stdout=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
    )


class CommandSandbox:
    """OS-backed command isolation interface."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        sandbox_root: Path,
        access_policy: CommandAccessPolicy,
        environment: dict[str, str],
        timeout: float,
    ) -> CommandResult:
        raise NotImplementedError


class MacOSCommandSandbox(CommandSandbox):
    """Run commands inside a deny-by-default macOS sandbox profile."""

    def __init__(self, sandbox_exec: Path) -> None:
        self.sandbox_exec = sandbox_exec

    @classmethod
    def discover(cls) -> "MacOSCommandSandbox":
        if sys.platform != "darwin":
            raise AgentError("command sandbox is unavailable")
        sandbox_exec = shutil.which("sandbox-exec", path=os.defpath)
        if sandbox_exec is None:
            raise AgentError("command sandbox is unavailable")
        try:
            resolved = Path(sandbox_exec).resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise AgentError("command sandbox is unavailable") from exc
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise AgentError("command sandbox is unavailable")
        true_executable = Path("/usr/bin/true")
        if not true_executable.is_file() or not os.access(true_executable, os.X_OK):
            raise AgentError("command sandbox is unavailable")
        with tempfile.TemporaryDirectory(prefix="qwen-sandbox-probe-") as directory:
            probe_root = Path(directory).resolve(strict=True)
            probe_policy = CommandAccessPolicy(
                readable_paths=(),
                writable_paths=(),
                scratch_path=probe_root,
                validation=False,
            )
            profile, parameters = _macos_sandbox_profile(
                probe_root, true_executable, probe_policy
            )
            command = [str(resolved)]
            for name, value in parameters:
                command.extend(["-D", f"{name}={value}"])
            command.extend(["-p", profile, str(true_executable)])
            try:
                probe = subprocess.run(
                    command,
                    cwd=probe_root,
                    env=_safe_environment(probe_root),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    shell=False,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise AgentError("command sandbox is unavailable") from exc
        if probe.returncode != 0:
            raise AgentError("command sandbox is unavailable")
        return cls(resolved)

    def run(
        self,
        argv: tuple[str, ...],
        *,
        sandbox_root: Path,
        access_policy: CommandAccessPolicy,
        environment: dict[str, str],
        timeout: float,
    ) -> CommandResult:
        executable = Path(argv[0])
        profile, parameters = _macos_sandbox_profile(
            sandbox_root, executable, access_policy
        )
        command = [str(self.sandbox_exec)]
        for name, value in parameters:
            command.extend(["-D", f"{name}={value}"])
        command.extend(["-p", profile, *argv])
        try:
            process = subprocess.Popen(
                command,
                cwd=sandbox_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise AgentError("command sandbox is unavailable") from exc
        except OSError as exc:
            raise AgentError("command sandbox setup failed") from exc
        try:
            result = _collect_process_output(process, timeout)
            if result.returncode == 71 and result.stderr.startswith("sandbox-exec:"):
                raise AgentError("command sandbox setup failed")
            return result
        except TimeoutError as exc:
            _terminate_process_group(process)
            _close_process_pipes(process)
            raise AgentError("command timed out") from exc
        except (OSError, subprocess.TimeoutExpired) as exc:
            _terminate_process_group(process)
            _close_process_pipes(process)
            raise AgentError("command sandbox execution failed") from exc


def _discover_command_sandbox() -> CommandSandbox:
    return MacOSCommandSandbox.discover()


def _command_sandbox_available() -> bool:
    try:
        _discover_command_sandbox()
    except AgentError:
        return False
    return True


def _task_id() -> str:
    return uuid.uuid4().hex[:12]


def _sandbox_basename(sandbox: str | Path) -> str:
    name = Path(sandbox).name
    return name or "sandbox"


def _request_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise AgentError(f"{field} must be an array of strings")
    return value


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validate_agent_request_inputs(
    task: str,
    sandbox: str | Path,
    allow: Any,
    read_only: Any,
    allowed_commands: Any,
    validation_commands: Any,
    validation_failure: Any,
) -> tuple[str, str, list[str], list[str], list[str], list[str], str | None]:
    task = _validate_task_text(task)
    sandbox_value = validate_text(
        str(sandbox) if isinstance(sandbox, Path) else sandbox,
        field="sandbox",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
        code="PATH_TOO_LONG",
    )
    allow_values = validate_string_list(
        _request_list(allow, "allow"),
        field="allow",
        max_items=MAX_ALLOW_SCOPES,
        count_code="TOO_MANY_ALLOW_SCOPES",
        item_kind="path",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
    )
    read_only_values = validate_string_list(
        _request_list(read_only if read_only is not None else [], "read_only"),
        field="read_only",
        max_items=MAX_READ_ONLY_SCOPES,
        count_code="TOO_MANY_READ_ONLY_SCOPES",
        item_kind="path",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
    )
    allowed_command_values = validate_string_list(
        _request_list(
            allowed_commands if allowed_commands is not None else [],
            "allowed_commands",
        ),
        field="allowed_commands",
        max_items=MAX_ALLOWED_COMMANDS,
        count_code="TOO_MANY_COMMANDS",
        item_kind="command",
        max_bytes=MAX_COMMAND_BYTES,
        max_chars=MAX_COMMAND_CHARS,
    )
    validation_command_values = validate_string_list(
        _request_list(
            validation_commands if validation_commands is not None else [],
            "validation_commands",
        ),
        field="validation_commands",
        max_items=MAX_VALIDATION_COMMANDS,
        count_code="TOO_MANY_VALIDATION_COMMANDS",
        item_kind="command",
        max_bytes=MAX_COMMAND_BYTES,
        max_chars=MAX_COMMAND_CHARS,
    )
    if validation_failure is not None:
        validation_failure = validate_text(
            validation_failure,
            field="validation_failure",
            max_bytes=MAX_VALIDATION_FAILURE_BYTES,
            max_chars=MAX_VALIDATION_FAILURE_CHARS,
            code="VALIDATION_FAILURE_TOO_LARGE",
        )
    return (
        task,
        sandbox_value,
        _dedupe_preserving_order(allow_values),
        _dedupe_preserving_order(read_only_values),
        _dedupe_preserving_order(allowed_command_values),
        validation_command_values,
        validation_failure,
    )


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
    if isinstance(exc, InputLimitError):
        return exc.code.lower()
    message = str(exc).lower()
    for endpoint_kind in (
        "endpoint_not_configured",
        "endpoint_invalid_url",
        "endpoint_invalid_scheme",
        "endpoint_credentials_not_allowed",
        "endpoint_query_fragment_not_allowed",
        "endpoint_remote_not_allowed",
    ):
        if endpoint_kind in message:
            return endpoint_kind
    for stream_kind in (
        "generation_deadline",
        "agent_generation_deadline",
        "generation_connect_timeout",
        "generation_pool_timeout",
        "generation_connect_error",
        "generation_read_timeout",
        "generation_write_timeout",
        "generation_read_error",
        "generation_protocol_error",
        "agent_generation_connect_timeout",
        "agent_generation_pool_timeout",
        "agent_generation_connect_error",
        "agent_generation_read_timeout",
        "agent_generation_write_timeout",
        "agent_generation_read_error",
        "agent_generation_protocol_error",
        "generation_stream_limit",
        "generation_event_limit",
        "generation_event_size_limit",
        "generation_output_limit",
        "generation_reasoning_limit",
        "generation_malformed",
        "agent_generation_stream",
    ):
        if stream_kind in message:
            return stream_kind
    if "timeout" in message:
        return "timeout"
    if "http_error" in message or " status=" in message:
        return "http_error"
    if "connect" in message or "unreachable" in message:
        return "endpoint_unreachable"
    if "not valid json" in message:
        return "invalid_json"
    if "action limit" in message or "action_limit" in message:
        return "action_limit"
    if "validation limit" in message or "validation_limit" in message:
        return "validation_limit"
    if stage == "ACTION_VALIDATE":
        return "action_invalid"
    return type(exc).__name__.lower()


def _validate_task_text(task: str) -> str:
    task = validate_text(
        task,
        field="task",
        max_bytes=MAX_TASK_BYTES,
        max_chars=MAX_TASK_CHARS,
        code="TASK_TOO_LARGE",
    )
    if not task.strip():
        raise AgentError("task must not be empty")
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
    limits: StreamLimits | None = None,
) -> AgentGeneration:
    limits = limits or StreamLimits(
        max_stream_bytes=MAX_STREAM_BYTES,
        max_events=MAX_STREAM_EVENTS,
        max_event_bytes=MAX_SSE_EVENT_BYTES,
        max_final_chars=MAX_ACTION_TEXT_CHARS,
        max_final_bytes=MAX_ACTION_TEXT_BYTES,
        max_reasoning_chars=MAX_REASONING_CONTENT_CHARS,
        max_reasoning_bytes=MAX_REASONING_CONTENT_BYTES,
    )
    parser = BoundedSSEParser(limits)
    final_parts: list[str] = []
    final_content_chars = 0
    final_content_bytes = 0
    reasoning_chars = 0
    reasoning_content_bytes = 0
    completion_tokens: int | None = None
    finish_reason: Any = None
    try:
        async for data in parser.iter_data(response):
            if data.strip() == "[DONE]":
                break
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
                text, final_content_chars, final_content_bytes = _bounded_stream_text(
                    delta["content"],
                    "delta.content",
                    current_chars=final_content_chars,
                    current_bytes=final_content_bytes,
                    max_chars=limits.max_final_chars,
                    max_bytes=limits.max_final_bytes,
                    code="GENERATION_OUTPUT_LIMIT",
                )
                final_parts.append(text)
            for alternate in ("text", "output_text"):
                if alternate in delta:
                    text, final_content_chars, final_content_bytes = _bounded_stream_text(
                        delta[alternate],
                        f"delta.{alternate}",
                        current_chars=final_content_chars,
                        current_bytes=final_content_bytes,
                        max_chars=limits.max_final_chars,
                        max_bytes=limits.max_final_bytes,
                        code="GENERATION_OUTPUT_LIMIT",
                    )
                    final_parts.append(text)
            if "reasoning_content" in delta:
                _text, reasoning_chars, reasoning_content_bytes = _bounded_stream_text(
                    delta["reasoning_content"],
                    "delta.reasoning_content",
                    current_chars=reasoning_chars,
                    current_bytes=reasoning_content_bytes,
                    max_chars=limits.max_reasoning_chars,
                    max_bytes=limits.max_reasoning_bytes,
                    code="GENERATION_REASONING_LIMIT",
                )
    except StreamLimitError as exc:
        _retry_telemetry(
            operation="generation",
            method="POST",
            attempt=attempts,
            error_kind=exc.code.lower(),
            retry=False,
        )
        raise
    except RuntimeError as exc:
        if str(exc).startswith("GENERATION_MALFORMED"):
            _retry_telemetry(
                operation="generation",
                method="POST",
                attempt=attempts,
                error_kind="malformed_sse",
                retry=False,
            )
            raise RuntimeError("AGENT_GENERATION_MALFORMED: invalid SSE stream") from exc
        raise
    except httpx.ReadTimeout as exc:
        _retry_telemetry(
            operation="generation",
            method="POST",
            attempt=attempts,
            error_kind="read_timeout",
            retry=False,
        )
        raise RuntimeError(
            f"AGENT_GENERATION_READ_TIMEOUT attempts={attempts} "
            f"events={parser.event_count}"
        ) from exc
    except httpx.HTTPError as exc:
        error_kind = _transport_error_kind(exc)
        _retry_telemetry(
            operation="generation",
            method="POST",
            attempt=attempts,
            error_kind=error_kind,
            retry=False,
        )
        raise RuntimeError(
            f"AGENT_GENERATION_{error_kind.upper()} attempts={attempts} "
            f"events={parser.event_count}"
        ) from exc

    elapsed_ms = _elapsed_ms(started)
    stats = _metadata(
        attempts=attempts,
        events=parser.event_count,
        stream_bytes=parser.stream_bytes,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        reasoning_content_chars=reasoning_chars,
        reasoning_content_bytes=reasoning_content_bytes,
        final_content_chars=final_content_chars,
        final_content_bytes=final_content_bytes,
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
        stream_bytes=parser.stream_bytes,
        event_count=parser.event_count,
        final_content_bytes=final_content_bytes,
        reasoning_content_bytes=reasoning_content_bytes,
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
    try:
        async with asyncio.timeout(MAX_GENERATION_SECONDS):
            response, attempts = await _start_generation_request(
                client,
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": MAX_OUTPUT_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "presence_penalty": PRESENCE_PENALTY,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": AGENT_RESPONSE_FORMAT,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                error_prefix="AGENT_GENERATION",
                sleep=sleep,
            )
            try:
                if not 200 <= response.status_code < 300:
                    _retry_telemetry(
                        operation="generation",
                        method="POST",
                        attempt=attempts,
                        error_kind="http_status",
                        retry=False,
                    )
                    excerpt = await _read_bounded_response_excerpt(response)
                    raise RuntimeError(
                        f"AGENT_GENERATION_HTTP attempts={attempts} "
                        f"status={response.status_code}: {excerpt}"
                    )
                return await _parse_agent_stream(
                    response,
                    started=started,
                    attempts=attempts,
                    http_status=response.status_code,
                )
            finally:
                await response.aclose()
    except TimeoutError as exc:
        _retry_telemetry(
            operation="generation",
            method="POST",
            attempt=1,
            error_kind="generation_deadline",
            retry=False,
        )
        raise RuntimeError(
            f"AGENT_GENERATION_DEADLINE limit_seconds={MAX_GENERATION_SECONDS} "
            f"elapsed_ms={_elapsed_ms(started)}"
        ) from exc


def _protocol_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


async def run_qwen_agent(
    task: str,
    sandbox: str | Path,
    allow: list[str],
    *,
    read_only: list[str] | None = None,
    allowed_commands: list[str] | None = None,
    validation_commands: list[str] | None = None,
    validation_failure: str | None = None,
    max_actions: int = MAX_AGENT_ACTIONS,
) -> AgentResult:
    task_id = _task_id()
    sandbox_name = "sandbox"
    started = time.monotonic()
    current_stage = "PRECHECK"
    model_calls = 0
    completion_tokens = 0
    reasoning_chars = 0
    protocol_errors = 0
    agent: SandboxAgent | None = None

    try:
        (
            task,
            sandbox,
            allow,
            read_only_values,
            allowed_command_values,
            validation_command_values,
            validation_failure,
        ) = _validate_agent_request_inputs(
            task,
            sandbox,
            allow,
            read_only if read_only is not None else [],
            allowed_commands if allowed_commands is not None else [],
            validation_commands if validation_commands is not None else [],
            validation_failure,
        )
        if isinstance(max_actions, bool) or not isinstance(max_actions, int):
            raise AgentError("max_actions must be an integer")
        if not 1 <= max_actions <= 100:
            raise AgentError("max_actions must be between 1 and 100")
        ensure_request_budget(
            {
                "task": task,
                "sandbox": sandbox,
                "allow": allow,
                "read_only": read_only_values,
                "allowed_commands": allowed_command_values,
                "validation_commands": validation_command_values,
                "validation_failure": validation_failure,
                "max_actions": max_actions,
            }
        )
        sandbox_name = _sandbox_basename(sandbox)
        _diagnostic(task_id, sandbox_name, "PRECHECK", "start")

        agent = SandboxAgent(
            sandbox,
            allow,
            read_only=read_only_values,
            allowed_commands=allowed_command_values,
        )
        _diagnostic(
            task_id,
            sandbox_name,
            "PRECHECK",
            "ok",
            writable_scopes=len(allow),
            read_only_scopes=len(read_only_values),
        )
        validation_policy = ValidationPolicy.from_commands(
            validation_command_values, agent.allowed_commands
        )
        conversation = AgentConversation(
            task,
            allow,
            read_only_values,
            allowed_command_values,
            validation_failure,
            validation_policy.display_commands,
        )
        validation_state = ValidationState()
        validation_runner = ValidationRunner(agent, validation_policy)
        _diagnostic(
            task_id,
            sandbox_name,
            "PRECHECK",
            "ok",
            validation_commands=validation_policy.count,
        )

        current_stage = "ENDPOINT_CONNECT"
        _diagnostic(task_id, sandbox_name, current_stage, "start")
        base_url = _configured_base_url()
        model = validate_text(
            os.environ.get("QWEN_MODEL", DEFAULT_MODEL),
            field="model",
            max_bytes=MAX_MODEL_IDENTIFIER_BYTES,
            max_chars=MAX_MODEL_IDENTIFIER_CHARS,
            code="MODEL_IDENTIFIER_TOO_LONG",
        )
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT,
            trust_env=False,
            follow_redirects=False,
        ) as client:
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
                messages, history_stats = conversation.snapshot()
                _diagnostic(
                    task_id,
                    sandbox_name,
                    current_stage,
                    "start",
                    model_call=model_calls,
                    **history_stats,
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
                    stream_bytes=generation.stream_bytes,
                    event_count=generation.event_count,
                    final_content_bytes=generation.final_content_bytes,
                    reasoning_content_bytes=generation.reasoning_content_bytes,
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
                    result_content = _protocol_result(result)
                    conversation.record(
                        generation.text,
                        None,
                        result,
                        files_read=agent.fully_observed_files,
                        files_changed=agent.changed_paths,
                        actions_completed=action_number,
                        protocol_errors=protocol_errors,
                    )
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        "MODEL_FEEDBACK",
                        "sent",
                        model_call=model_calls,
                        feedback_chars=len(result_content),
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
                mutation_generation_before = agent.mutation_generation
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

                if agent.mutation_generation != mutation_generation_before:
                    validation_state.mark_mutation(agent.mutation_generation)

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
                    validation_state.mark_model_done()
                    current_stage = "MODEL_DONE"
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        current_stage,
                        "ok",
                        actions=action_number,
                        model_calls=model_calls,
                    )
                    if not validation_policy.commands:
                        _diagnostic(
                            task_id,
                            sandbox_name,
                            "DONE",
                            "ok",
                            result_status="done",
                            validated=False,
                            actions=action_number,
                            model_calls=model_calls,
                            completion_tokens=completion_tokens,
                            reasoning_chars=reasoning_chars,
                            read_bytes=agent.read_bytes,
                            write_bytes=agent.write_bytes,
                            validation_commands=0,
                            validation_rounds=0,
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
                            validated=False,
                            validation_commands=0,
                            validation_failures=0,
                            validation_rounds=0,
                        )

                    current_stage = "VALIDATION_START"
                    validation_round = validation_state.start_round(
                        MAX_VALIDATION_ROUNDS
                    )
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        current_stage,
                        "start",
                        validation_round=validation_round,
                        validation_commands=validation_policy.count,
                    )

                    def report_validation_command(
                        index: int, validation_result: ValidationResult
                    ) -> None:
                        _diagnostic(
                            task_id,
                            sandbox_name,
                            "VALIDATION_COMMAND",
                            "ok" if validation_result.passed else "error",
                            validation_round=validation_round,
                            validation_command_index=index,
                            returncode=validation_result.returncode,
                        )

                    validation_results = validation_runner.run(
                        on_result=report_validation_command
                    )
                    if validation_state.finish_round(
                        validation_results, agent.mutation_generation
                    ):
                        _diagnostic(
                            task_id,
                            sandbox_name,
                            "VALIDATION_PASSED",
                            "ok",
                            validation_round=validation_round,
                            validation_commands=validation_policy.count,
                        )
                        _diagnostic(
                            task_id,
                            sandbox_name,
                            "DONE",
                            "ok",
                            result_status="validated",
                            validated=True,
                            actions=action_number,
                            model_calls=model_calls,
                            completion_tokens=completion_tokens,
                            reasoning_chars=reasoning_chars,
                            read_bytes=agent.read_bytes,
                            write_bytes=agent.write_bytes,
                            validation_commands=validation_policy.count,
                            validation_failures=validation_state.failure_count,
                            validation_rounds=validation_state.validation_rounds,
                            elapsed_ms=_elapsed_ms(started),
                        )
                        return AgentResult(
                            status="validated",
                            actions=action_number,
                            files_changed=sorted(agent.changed_paths),
                            model_calls=model_calls,
                            completion_tokens=completion_tokens,
                            reasoning_chars=reasoning_chars,
                            elapsed_ms=_elapsed_ms(started),
                            read_bytes=agent.read_bytes,
                            write_bytes=agent.write_bytes,
                            validated=True,
                            validation_commands=validation_policy.count,
                            validation_failures=validation_state.failure_count,
                            validation_rounds=validation_state.validation_rounds,
                        )

                    failed_result = next(
                        validation_result
                        for validation_result in validation_results
                        if not validation_result.passed
                    )
                    current_stage = "VALIDATION_FAILED"
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        current_stage,
                        "error",
                        validation_round=validation_round,
                        validation_command_index=validation_results.index(
                            failed_result
                        )
                        + 1,
                        returncode=failed_result.returncode,
                    )
                    if validation_state.validation_rounds >= MAX_VALIDATION_ROUNDS:
                        raise RuntimeError(
                            f"AGENT_VALIDATION_LIMIT: {MAX_VALIDATION_ROUNDS}"
                        )
                    feedback = failed_result.feedback(validation_round)
                    conversation.record(
                        generation.text,
                        {"action": "validation"},
                        feedback,
                        files_read=agent.fully_observed_files,
                        files_changed=agent.changed_paths,
                        actions_completed=action_number,
                        protocol_errors=protocol_errors,
                    )
                    validation_state.prepare_retry()
                    _diagnostic(
                        task_id,
                        sandbox_name,
                        "MODEL_FEEDBACK",
                        "sent",
                        model_call=model_calls,
                        feedback_chars=len(_protocol_result(feedback)),
                    )
                    continue
                result_content = _protocol_result(result)
                conversation.record(
                    generation.text,
                    action,
                    result,
                    files_read=agent.fully_observed_files,
                    files_changed=agent.changed_paths,
                    actions_completed=action_number,
                    protocol_errors=protocol_errors,
                )
                _diagnostic(
                    task_id,
                    sandbox_name,
                    "MODEL_FEEDBACK",
                    "sent",
                    model_call=model_calls,
                    feedback_chars=len(result_content),
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
