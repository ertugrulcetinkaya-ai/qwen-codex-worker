#!/usr/bin/env python3
"""Read-only MCP bridge from Codex to a local OpenAI-compatible Qwen server."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from qwen_input_limits import (
    MAX_ENDPOINT_BYTES,
    MAX_ENDPOINT_CHARS,
    MAX_MODEL_IDENTIFIER_BYTES,
    MAX_MODEL_IDENTIFIER_CHARS,
    MAX_PATH_BYTES,
    MAX_PATH_CHARS,
    MAX_SOURCE_FILES,
    MAX_TASK_BYTES,
    MAX_TASK_CHARS,
    MAX_TEST_OUTPUT_BYTES,
    MAX_TEST_OUTPUT_CHARS,
    InputLimitError,
    ensure_model_input,
    ensure_request_budget,
    read_bounded_file,
    validate_string_list,
    validate_text,
)

DEFAULT_BASE_URL: str | None = None
DEFAULT_MODEL = (
    "/Users/ertugrulcetinkaya/Models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
)
MAX_FILES = MAX_SOURCE_FILES
MAX_SOURCE_CHARS = 260_000
MAX_SOURCE_BYTES = 1_100_000
MAX_OUTPUT_TOKENS = 4_096
MAX_STREAM_BYTES = 2_000_000
MAX_STREAM_EVENTS = 10_000
MAX_GENERATION_SECONDS = 300.0
MAX_PATCH_FINAL_CONTENT_CHARS = 512_000
MAX_PATCH_FINAL_CONTENT_BYTES = 2_000_000
MAX_AGENT_FINAL_CONTENT_CHARS = 20_000
MAX_AGENT_FINAL_CONTENT_BYTES = 80_000
MAX_REASONING_CONTENT_CHARS = 200_000
MAX_REASONING_CONTENT_BYTES = 800_000
MAX_SSE_EVENT_BYTES = 256_000
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5
MAX_ERROR_BODY_CHARS = 1_000
MAX_ERROR_BODY_BYTES = 4_096
MAX_PREFLIGHT_BODY_BYTES = 64_000
MAX_CONNECT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
PREFLIGHT_MAX_CONNECT_ATTEMPTS = 5
PREFLIGHT_RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0)
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, pool=10.0, write=30.0, read=1_800.0)
PREFLIGHT_TIMEOUT = httpx.Timeout(connect=3.0, pool=3.0, write=3.0, read=5.0)
# The preflight endpoint is an idempotent model-discovery GET.  Its retry
# policy is intentionally separate from generation, whose POST may have
# started inference before the client observes a transport failure.
IDEMPOTENT_PREFLIGHT_RETRYABLE_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadTimeout,
    httpx.ReadError,
)
# These are the only generation failures that are known, for the current
# httpx/httpcore transport, to happen before request bytes can be sent.
# ConnectError is deliberately excluded: its semantics are not narrow enough
# to justify risking a duplicate inference.
GENERATION_SAFE_PRE_SEND_ERRORS = (httpx.ConnectTimeout, httpx.PoolTimeout)


@dataclass(frozen=True)
class StreamLimits:
    """Hard client-side limits for one OpenAI-compatible SSE generation."""

    max_stream_bytes: int = MAX_STREAM_BYTES
    max_events: int = MAX_STREAM_EVENTS
    max_event_bytes: int = MAX_SSE_EVENT_BYTES
    max_final_chars: int = MAX_PATCH_FINAL_CONTENT_CHARS
    max_final_bytes: int = MAX_PATCH_FINAL_CONTENT_BYTES
    max_reasoning_chars: int = MAX_REASONING_CONTENT_CHARS
    max_reasoning_bytes: int = MAX_REASONING_CONTENT_BYTES


class StreamLimitError(RuntimeError):
    """A bounded generation exceeded a client-enforced stream or output limit."""

    def __init__(self, code: str, observed: int, limit: int) -> None:
        self.code = code
        self.observed = observed
        self.limit = limit
        super().__init__(f"{code} observed={observed} limit={limit}")


def _default_patch_stream_limits() -> StreamLimits:
    """Build patch limits from current module constants (including test overrides)."""
    return StreamLimits(
        max_stream_bytes=MAX_STREAM_BYTES,
        max_events=MAX_STREAM_EVENTS,
        max_event_bytes=MAX_SSE_EVENT_BYTES,
        max_final_chars=MAX_PATCH_FINAL_CONTENT_CHARS,
        max_final_bytes=MAX_PATCH_FINAL_CONTENT_BYTES,
        max_reasoning_chars=MAX_REASONING_CONTENT_CHARS,
        max_reasoning_bytes=MAX_REASONING_CONTENT_BYTES,
    )


class BoundedSSEParser:
    """Incrementally parse SSE bytes without buffering an unbounded response."""

    def __init__(self, limits: StreamLimits) -> None:
        self.limits = limits
        self.stream_bytes = 0
        self.event_count = 0
        self._event_bytes = 0
        self._line = bytearray()
        self._data_fields: list[str] = []
        self._event_started = False
        self._ignore_lf = False

    def _count_byte(self) -> None:
        self.stream_bytes += 1
        if self.stream_bytes > self.limits.max_stream_bytes:
            raise StreamLimitError(
                "GENERATION_STREAM_LIMIT", self.stream_bytes, self.limits.max_stream_bytes
            )
        self._event_bytes += 1
        if self._event_bytes > self.limits.max_event_bytes:
            raise StreamLimitError(
                "GENERATION_EVENT_SIZE_LIMIT",
                self._event_bytes,
                self.limits.max_event_bytes,
            )

    def _reset_event(self) -> None:
        self._event_bytes = 0
        self._data_fields.clear()
        self._event_started = False

    def _dispatch(self) -> Iterator[str]:
        if not self._event_started:
            return
        self.event_count += 1
        if self.event_count > self.limits.max_events:
            raise StreamLimitError(
                "GENERATION_EVENT_LIMIT", self.event_count, self.limits.max_events
            )
        data = "\n".join(self._data_fields) if self._data_fields else None
        self._reset_event()
        if data is not None:
            yield data

    def _process_line(self, raw_line: bytes) -> Iterator[str]:
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("GENERATION_MALFORMED: invalid UTF-8 in SSE stream") from exc

        if not line:
            yield from self._dispatch()
            return

        self._event_started = True
        if line.startswith(":"):
            return
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field, value = line, ""
        if field == "data":
            self._data_fields.append(value)

    def feed(self, chunk: bytes) -> Iterator[str]:
        """Consume one raw byte chunk and yield complete SSE data fields."""
        if not isinstance(chunk, bytes):
            raise RuntimeError("GENERATION_MALFORMED: transport returned non-bytes")
        for byte in chunk:
            self._count_byte()
            if self._ignore_lf:
                self._ignore_lf = False
                if byte == 0x0A:
                    continue
            if byte == 0x0D:
                raw_line = bytes(self._line)
                self._line.clear()
                self._ignore_lf = True
                yield from self._process_line(raw_line)
            elif byte == 0x0A:
                raw_line = bytes(self._line)
                self._line.clear()
                yield from self._process_line(raw_line)
            else:
                self._line.append(byte)

    def finish(self) -> Iterator[str]:
        """Flush a final line/event when a server closes without a blank line."""
        if self._line:
            raw_line = bytes(self._line)
            self._line.clear()
            yield from self._process_line(raw_line)
        yield from self._dispatch()

    async def iter_data(self, response: httpx.Response) -> AsyncIterator[str]:
        async for chunk in response.aiter_bytes():
            for data in self.feed(chunk):
                yield data
                if data.strip() == "[DONE]":
                    return
        for data in self.finish():
            yield data
            if data.strip() == "[DONE]":
                return

WORKER_ROLE = """You are a code implementation worker. You do not manage the repository.
Implement only the requested task using the supplied source files.
Return a valid unified diff and nothing else.
Do not include markdown fences.
Do not make unrelated changes.
If the supplied files are insufficient, do not invent unseen code.
Instead return:
NEED_FILES:
path/to/file1
path/to/file2"""

mcp = FastMCP(
    "qwen_worker",
    instructions=(
        "Read-only implementation worker. qwen_patch reads only caller-selected repository "
        "files and returns model-generated text. It never changes or executes repository content."
    ),
)


def _read_supplied_files(repo_root: str, files: list[str]) -> tuple[Path, list[tuple[str, str]]]:
    files = validate_string_list(
        files,
        field="files",
        max_items=MAX_FILES,
        count_code="TOO_MANY_FILES",
        item_kind="path",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
    )
    if not files:
        raise ValueError("files must contain at least one repository-relative path")

    repo_root = validate_text(
        repo_root,
        field="repo_root",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
        code="PATH_TOO_LONG",
    )
    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repo_root must resolve to a directory")

    total_chars = 0
    total_bytes = 0
    supplied: list[tuple[str, str]] = []
    seen: set[Path] = set()

    for relative_name in files:
        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise ValueError(f"file path must be repository-relative: {relative_name!r}")

        resolved = (root / relative_path).resolve(strict=True)
        try:
            normalized = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"file path escapes repo_root: {relative_name!r}") from exc

        if resolved in seen:
            raise ValueError(f"duplicate file path: {relative_name!r}")
        seen.add(resolved)

        if not resolved.is_file():
            raise ValueError(f"path is not a regular file: {relative_name!r}")

        remaining_bytes = MAX_SOURCE_BYTES - total_bytes
        file_size = resolved.stat().st_size
        if file_size > remaining_bytes:
            raise ValueError("supplied files exceed the safety byte limit")

        data = read_bounded_file(
            resolved,
            remaining_bytes,
            field="source file",
            code="SOURCE_TOO_LARGE",
        )
        total_bytes += len(data)
        if b"\x00" in data:
            raise ValueError(f"binary file rejected: {relative_name!r}")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"non-UTF-8 or binary file rejected: {relative_name!r}") from exc

        total_chars += len(text)
        if total_chars > MAX_SOURCE_CHARS:
            raise ValueError(
                f"supplied source exceeds the {MAX_SOURCE_CHARS:,}-character limit"
            )
        supplied.append((normalized, text))

    return root, supplied


def _build_prompt(
    task: str,
    root: Path,
    supplied: list[tuple[str, str]],
    test_output: str | None,
) -> str:
    task = validate_text(
        task,
        field="task",
        max_bytes=MAX_TASK_BYTES,
        max_chars=MAX_TASK_CHARS,
        code="TASK_TOO_LARGE",
    )
    if not task.strip():
        raise ValueError("task must not be empty")
    if test_output is not None:
        test_output = validate_text(
            test_output,
            field="test_output",
            max_bytes=MAX_TEST_OUTPUT_BYTES,
            max_chars=MAX_TEST_OUTPUT_CHARS,
            code="TEST_OUTPUT_TOO_LARGE",
        )
    sections = [
        "IMPLEMENTATION TASK:",
        task,
        "",
        "REPOSITORY: supplied files only; all paths are repository-relative",
        "",
        "SUPPLIED SOURCE FILES:",
    ]
    for path, source in supplied:
        sections.extend(
            [
                f"===== BEGIN FILE: {path} =====",
                source,
                f"===== END FILE: {path} =====",
                "",
            ]
        )

    if test_output is not None:
        sections.extend(["TEST OUTPUT:", test_output, ""])

    sections.append(
        "Only modify files explicitly supplied in this request. Do not create, delete, "
        "rename, or target any other file. Produce only a unified diff for the requested "
        "implementation, or the specified NEED_FILES response if these files are "
        "insufficient."
    )
    return ensure_model_input("\n".join(sections), field="patch_model_input")


def _response_excerpt(response: httpx.Response) -> str:
    """Return a bounded, single-line response excerpt for error diagnostics."""
    text = " ".join(response.text.split())
    if not text:
        return "<empty response body>"
    if len(text) > MAX_ERROR_BODY_CHARS:
        return f"{text[:MAX_ERROR_BODY_CHARS]}..."
    return text


async def _read_bounded_response_excerpt(response: httpx.Response) -> str:
    """Read only a bounded prefix of a streamed HTTP error body."""
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            remaining = MAX_ERROR_BODY_BYTES - len(body)
            if remaining <= 0:
                break
            body.extend(chunk[:remaining])
            if len(body) >= MAX_ERROR_BODY_BYTES:
                break
    except httpx.HTTPError:
        pass
    text = " ".join(bytes(body).decode("utf-8", errors="replace").split())
    if not text:
        return "<empty response body>"
    if len(text) > MAX_ERROR_BODY_CHARS:
        return f"{text[:MAX_ERROR_BODY_CHARS]}..."
    return text


async def _read_bounded_response_bytes(
    response: httpx.Response, *, max_bytes: int
) -> bytes:
    """Read a bounded successful response without relying on ``aread``."""
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = max_bytes - len(body)
        if len(chunk) > remaining:
            raise RuntimeError(
                f"PREFLIGHT_BODY_LIMIT observed={len(body) + len(chunk)} limit={max_bytes}"
            )
        body.extend(chunk)
    return bytes(body)


def _text_field(value: Any, field_name: str) -> str | None:
    """Extract text from a string or OpenAI-style content-part array."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for index, part in enumerate(value):
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, Mapping):
                raise RuntimeError(
                    f"Qwen returned malformed {field_name}[{index}] content"
                )
            text = part.get("text")
            if not isinstance(text, str):
                raise RuntimeError(
                    f"Qwen returned malformed {field_name}[{index}].text"
                )
            parts.append(text)
        return "".join(parts)
    raise RuntimeError(f"Qwen returned non-text {field_name}")


def _parse_chat_completion(payload: Any) -> str:
    """Validate an OpenAI-compatible chat completion and return its final text."""
    if not isinstance(payload, Mapping):
        raise RuntimeError("Qwen returned malformed JSON: top level must be an object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Qwen response is missing a non-empty choices array")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise RuntimeError("Qwen response choices[0] must be an object")

    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise RuntimeError("Qwen generation exceeded the output token limit")

    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("Qwen response choices[0] is missing a message object")

    # llama.cpp uses message.content for the final answer and keeps private model
    # reasoning in message.reasoning_content. Some compatible servers represent
    # final content as typed parts or expose it as text/output_text on the message.
    final_text: str | None = None
    final_field = "message.content"
    if "content" in message:
        final_text = _text_field(message["content"], final_field)
    for alternate in ("text", "output_text"):
        if final_text is not None and final_text.strip():
            break
        if alternate in message:
            final_field = f"message.{alternate}"
            final_text = _text_field(message[alternate], final_field)

    if final_text is not None and final_text.strip():
        return final_text

    reasoning = _text_field(message.get("reasoning_content"), "message.reasoning_content")
    usage = payload.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    details = [f"finish_reason={finish_reason!r}"]
    if completion_tokens is not None:
        details.append(f"completion_tokens={completion_tokens!r}")
    if reasoning:
        details.append(f"reasoning_content_chars={len(reasoning)}")
    raise RuntimeError(
        "Qwen returned no final answer text (" + ", ".join(details) + ")"
    )


def _parse_chat_completion_response(response: httpx.Response) -> str:
    """Surface HTTP/JSON errors, then parse a chat completion response."""
    if not 200 <= response.status_code < 300:
        reason = f" {response.reason_phrase}" if response.reason_phrase else ""
        raise RuntimeError(
            f"Qwen HTTP error {response.status_code}{reason}: {_response_excerpt(response)}"
        )
    try:
        payload = response.json()
    except (JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Qwen returned malformed JSON (HTTP {response.status_code}): "
            f"{_response_excerpt(response)}"
        ) from exc
    return _parse_chat_completion(payload)


def _elapsed_ms(started: float) -> int:
    return round((time.monotonic() - started) * 1_000)


def _metadata(**values: Any) -> str:
    return " ".join(
        f"{name}={value!r}" for name, value in values.items() if value is not None
    )


def _debug(message: str) -> None:
    if os.environ.get("QWEN_DEBUG"):
        print(f"qwen-patch: {message}", file=sys.stderr)


def _normalize_diff_target(raw_path: str, *, side: str) -> str:
    """Return one canonical repository-relative diff target or fail closed."""
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_INVALID")
    if raw_path == "/dev/null":
        return raw_path
    if raw_path.startswith('"') or raw_path.endswith('"'):
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_QUOTED")
    if "\x00" in raw_path:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_INVALID")
    if (
        raw_path.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", raw_path) is not None
        or "\\" in raw_path
    ):
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_ABSOLUTE")

    path = raw_path
    for prefix in (f"{side}/", "a/", "b/"):
        if path.startswith(prefix):
            path = path[len(prefix) :]
            break
    parts = path.split("/")
    if any(part == ".." for part in parts):
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_TRAVERSAL")
    if not path or any(part == "" for part in parts):
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_INVALID")
    canonical = PurePosixPath(path).as_posix()
    if canonical in {"", "."}:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_INVALID")
    return canonical


def _diff_header_path(line: str) -> str:
    """Parse the documented header path, allowing a GNU tab timestamp suffix."""
    raw_path = line[4:]
    if "\t" in raw_path:
        raw_path = raw_path.split("\t", 1)[0]
    if not raw_path:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_INVALID")
    return raw_path


def _validate_diff_targets(
    old_raw: str,
    new_raw: str,
    *,
    allowed_files: frozenset[str],
    seen_targets: set[str],
) -> None:
    old_path = _normalize_diff_target(old_raw, side="old")
    new_path = _normalize_diff_target(new_raw, side="new")
    if old_path == "/dev/null" and new_path == "/dev/null":
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_INVALID_NULL_TARGET")
    if old_path == "/dev/null":
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_NEW_FILE_NOT_ALLOWED")
    if new_path == "/dev/null":
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_DELETE_NOT_ALLOWED")
    if old_path != new_path:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_RENAME_NOT_ALLOWED")
    if old_path not in allowed_files:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_TARGET_NOT_ALLOWED")
    if old_path in seen_targets:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_DUPLICATE_TARGET")
    seen_targets.add(old_path)


def _validate_worker_output(
    output: str, *, allowed_files: Iterable[str]
) -> None:
    """Reject empty, malformed, or structurally truncated worker output."""
    lines = output.splitlines()
    if not lines:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: empty output")
    if lines[0] == "NEED_FILES:":
        if len(lines) < 2 or any(not line.strip() for line in lines[1:]):
            raise RuntimeError("GENERATION_INVALID_OUTPUT: malformed NEED_FILES response")
        for name in lines[1:]:
            path = Path(name.strip())
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("GENERATION_INVALID_OUTPUT: unsafe NEED_FILES path")
        return
    allowed = frozenset(allowed_files)
    hunk_header = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
    )
    index = 0
    file_count = 0
    seen_targets: set[str] = set()
    while index < len(lines):
        while index < len(lines):
            line = lines[index]
            if line.startswith(("diff --git ", "index ", "old mode ", "new mode ")):
                index += 1
                continue
            if line.startswith("new file mode "):
                raise RuntimeError(
                    "GENERATION_INVALID_OUTPUT: PATCH_NEW_FILE_NOT_ALLOWED"
                )
            if line.startswith("deleted file mode "):
                raise RuntimeError(
                    "GENERATION_INVALID_OUTPUT: PATCH_DELETE_NOT_ALLOWED"
                )
            if line.startswith(("similarity index ", "rename from ", "rename to ")):
                raise RuntimeError("GENERATION_INVALID_OUTPUT: PATCH_RENAME_NOT_ALLOWED")
            break
        if index >= len(lines) or not lines[index].startswith("--- "):
            raise RuntimeError("GENERATION_INVALID_OUTPUT: missing unified diff header")
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise RuntimeError("GENERATION_INVALID_OUTPUT: incomplete unified diff header")
        _validate_diff_targets(
            _diff_header_path(lines[index]),
            _diff_header_path(lines[index + 1]),
            allowed_files=allowed,
            seen_targets=seen_targets,
        )
        index += 2
        file_count += 1
        hunk_count = 0
        while index < len(lines) and not lines[index].startswith(("diff --git ", "--- ")):
            match = hunk_header.match(lines[index])
            if match is None:
                raise RuntimeError("GENERATION_INVALID_OUTPUT: malformed hunk header")
            old_expected = int(match.group(2) or 1)
            new_expected = int(match.group(4) or 1)
            old_seen = 0
            new_seen = 0
            index += 1
            while index < len(lines) and not lines[index].startswith(("@@ ", "diff --git ", "--- ")):
                line = lines[index]
                if line.startswith(" "):
                    old_seen += 1
                    new_seen += 1
                elif line.startswith("-"):
                    old_seen += 1
                elif line.startswith("+"):
                    new_seen += 1
                elif not line.startswith("\\ No newline at end of file"):
                    raise RuntimeError("GENERATION_INVALID_OUTPUT: malformed hunk line")
                index += 1
            if old_seen != old_expected or new_seen != new_expected:
                raise RuntimeError("GENERATION_INVALID_OUTPUT: truncated hunk")
            hunk_count += 1
        if hunk_count == 0:
            raise RuntimeError("GENERATION_INVALID_OUTPUT: diff contains no hunks")
    if file_count == 0:
        raise RuntimeError("GENERATION_INVALID_OUTPUT: diff contains no files")


def _transport_error_kind(exc: httpx.HTTPError) -> str:
    """Return a stable, content-free transport classification."""
    if isinstance(exc, httpx.ConnectTimeout):
        return "connect_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "pool_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "write_timeout"
    if isinstance(exc, httpx.WriteError):
        return "write_error"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "protocol_error"
    if isinstance(exc, httpx.ReadError):
        return "read_error"
    if isinstance(exc, httpx.ConnectError):
        return "connect_error"
    return type(exc).__name__.lower()


def _retry_telemetry(
    *,
    operation: str,
    method: str,
    attempt: int,
    error_kind: str,
    retry: bool,
) -> None:
    """Emit retry metadata only; never include request or response content."""
    _debug(
        "RETRY "
        f"operation={operation} method={method} attempt={attempt} "
        f"error_kind={error_kind} retry={'true' if retry else 'false'}"
    )


def _build_transport_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: dict[str, Any] | None,
    timeout: httpx.Timeout | None,
) -> httpx.Request:
    """Use the client's configured timeout when no operation override exists."""
    if timeout is None:
        return client.build_request(method, url, json=json)
    return client.build_request(method, url, json=json, timeout=timeout)


async def _request_idempotent_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    stage: str,
    stream: bool = False,
    json: dict[str, Any] | None = None,
    timeout: httpx.Timeout | None = None,
    sleep: Any = asyncio.sleep,
    max_attempts: int = PREFLIGHT_MAX_CONNECT_ATTEMPTS,
    retry_backoff_seconds: tuple[float, ...] = PREFLIGHT_RETRY_BACKOFF_SECONDS,
) -> tuple[httpx.Response, int]:
    """Retry an idempotent operation using its explicit transient error policy."""
    started = time.monotonic()
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if len(retry_backoff_seconds) < max_attempts - 1:
        raise ValueError("retry_backoff_seconds must cover every retry")

    for attempt in range(1, max_attempts + 1):
        request = _build_transport_request(
            client, method, url, json=json, timeout=timeout
        )
        try:
            response = await client.send(
                request, stream=stream, follow_redirects=False
            )
            return response, attempt
        except IDEMPOTENT_PREFLIGHT_RETRYABLE_ERRORS as exc:
            error_kind = _transport_error_kind(exc)
            _retry_telemetry(
                operation="preflight",
                method=method,
                attempt=attempt,
                error_kind=error_kind,
                retry=attempt < max_attempts,
            )
            if attempt == max_attempts:
                kind = (
                    "endpoint_timeout"
                    if isinstance(
                        exc,
                        (httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ReadTimeout),
                    )
                    else "endpoint_unreachable"
                )
                raise RuntimeError(
                    f"{kind}: {stage} attempts={attempt} elapsed_ms={_elapsed_ms(started)}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            await sleep(retry_backoff_seconds[attempt - 1])
    raise AssertionError("connection retry loop exhausted unexpectedly")


async def _start_generation_request(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[str, Any],
    error_prefix: str = "GENERATION",
    timeout: httpx.Timeout | None = None,
    sleep: Any = asyncio.sleep,
    max_attempts: int = MAX_CONNECT_ATTEMPTS,
    retry_backoff_seconds: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
) -> tuple[httpx.Response, int]:
    """Start one generation POST with only safe pre-send retries.

    Once request handling might have progressed beyond pool acquisition and
    connection establishment, retrying could launch a duplicate inference.
    Therefore every other httpx transport error is terminal here.
    """
    started = time.monotonic()
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if len(retry_backoff_seconds) < max_attempts - 1:
        raise ValueError("retry_backoff_seconds must cover every retry")

    for attempt in range(1, max_attempts + 1):
        request = _build_transport_request(
            client, "POST", url, json=json, timeout=timeout
        )
        try:
            response = await client.send(
                request, stream=True, follow_redirects=False
            )
            return response, attempt
        except GENERATION_SAFE_PRE_SEND_ERRORS as exc:
            error_kind = _transport_error_kind(exc)
            _retry_telemetry(
                operation="generation",
                method="POST",
                attempt=attempt,
                error_kind=error_kind,
                retry=attempt < max_attempts,
            )
            if attempt == max_attempts:
                raise RuntimeError(
                    f"{error_prefix}_{error_kind.upper()} attempts={attempt} "
                    f"elapsed_ms={_elapsed_ms(started)}"
                ) from exc
            await sleep(retry_backoff_seconds[attempt - 1])
        except httpx.HTTPError as exc:
            error_kind = _transport_error_kind(exc)
            _retry_telemetry(
                operation="generation",
                method="POST",
                attempt=attempt,
                error_kind=error_kind,
                retry=False,
            )
            raise RuntimeError(
                f"{error_prefix}_{error_kind.upper()} attempts={attempt} "
                f"elapsed_ms={_elapsed_ms(started)}"
            ) from exc
    raise AssertionError("generation request loop exhausted unexpectedly")


async def _preflight(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    *,
    sleep: Any = asyncio.sleep,
) -> int:
    started = time.monotonic()
    response, attempts = await _request_idempotent_with_retries(
        client,
        "GET",
        f"{base_url}/models",
        stage="PREFLIGHT_CONNECT",
        timeout=PREFLIGHT_TIMEOUT,
        stream=True,
        sleep=sleep,
        max_attempts=PREFLIGHT_MAX_CONNECT_ATTEMPTS,
        retry_backoff_seconds=PREFLIGHT_RETRY_BACKOFF_SECONDS,
    )
    try:
        if not 200 <= response.status_code < 300:
            excerpt = await _read_bounded_response_excerpt(response)
            raise RuntimeError(
                f"endpoint_http_error: PREFLIGHT_HTTP attempts={attempts} "
                f"status={response.status_code} "
                f"elapsed_ms={_elapsed_ms(started)}: {excerpt}"
            )
        body = await _read_bounded_response_bytes(
            response, max_bytes=MAX_PREFLIGHT_BODY_BYTES
        )
        try:
            payload = json.loads(body)
        except (JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"PREFLIGHT_MALFORMED attempts={attempts} status={response.status_code} "
                f"elapsed_ms={_elapsed_ms(started)}: response is not valid JSON"
            ) from exc
    finally:
        await response.aclose()
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise RuntimeError(
            f"PREFLIGHT_MALFORMED attempts={attempts} status={response.status_code} "
            f"elapsed_ms={_elapsed_ms(started)}: response is missing a data array"
        )
    loaded_models = {
        item.get("id")
        for item in payload["data"]
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if model not in loaded_models:
        raise RuntimeError(
            f"PREFLIGHT_MODEL attempts={attempts} elapsed_ms={_elapsed_ms(started)}: "
            "expected model is not loaded"
        )
    return attempts


def _stream_text(value: Any, field_name: str) -> str:
    try:
        text = _text_field(value, field_name)
    except RuntimeError as exc:
        raise RuntimeError(f"GENERATION_MALFORMED: {exc}") from exc
    return text or ""


def _bounded_stream_text(
    value: Any,
    field_name: str,
    *,
    current_chars: int,
    current_bytes: int,
    max_chars: int,
    max_bytes: int,
    code: str,
) -> tuple[str, int, int]:
    text = _stream_text(value, field_name)
    next_chars = current_chars + len(text)
    next_bytes = current_bytes + len(text.encode("utf-8"))
    if next_chars > max_chars:
        raise StreamLimitError(code, next_chars, max_chars)
    if next_bytes > max_bytes:
        raise StreamLimitError(code, next_bytes, max_bytes)
    return text, next_chars, next_bytes


async def _parse_chat_completion_stream(
    response: httpx.Response,
    *,
    attempts: int,
    generation_started: float,
    headers_ms: int,
    limits: StreamLimits | None = None,
) -> str:
    limits = limits or _default_patch_stream_limits()
    parser = BoundedSSEParser(limits)
    final_parts: list[str] = []
    final_content_chars = 0
    final_content_bytes = 0
    reasoning_content_chars = 0
    reasoning_content_bytes = 0
    finish_reason: Any = None
    completion_tokens: Any = None
    first_chunk_ms: int | None = None

    try:
        async for data in parser.iter_data(response):
            if data.strip() == "[DONE]":
                break
            if first_chunk_ms is None:
                first_chunk_ms = _elapsed_ms(generation_started)
            try:
                payload = httpx.Response(200, text=data).json()
            except (JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={parser.event_count} "
                    f"headers_ms={headers_ms} first_chunk_ms={first_chunk_ms} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: invalid SSE JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={parser.event_count} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: event is not an object"
                )
            usage = payload.get("usage")
            if isinstance(usage, Mapping) and usage.get("completion_tokens") is not None:
                completion_tokens = usage.get("completion_tokens")
            choices = payload.get("choices")
            if choices == []:
                continue
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={parser.event_count} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: missing streamed choice"
                )
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={parser.event_count} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: missing delta object"
                )
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
                _text, reasoning_content_chars, reasoning_content_bytes = _bounded_stream_text(
                    delta["reasoning_content"],
                    "delta.reasoning_content",
                    current_chars=reasoning_content_chars,
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
    except httpx.ReadTimeout as exc:
        _retry_telemetry(
            operation="generation",
            method="POST",
            attempt=attempts,
            error_kind="read_timeout",
            retry=False,
        )
        raise RuntimeError(
            f"GENERATION_READ_TIMEOUT attempts={attempts} events={parser.event_count} "
            f"headers_ms={headers_ms} first_chunk_ms={first_chunk_ms} "
            f"elapsed_ms={_elapsed_ms(generation_started)}"
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
            f"GENERATION_{error_kind.upper()} attempts={attempts} "
            f"events={parser.event_count} "
            f"headers_ms={headers_ms} first_chunk_ms={first_chunk_ms} "
            f"elapsed_ms={_elapsed_ms(generation_started)}"
        ) from exc

    final_text = "".join(final_parts)
    stats = _metadata(
        attempts=attempts,
        retry_count=attempts - 1,
        events=parser.event_count,
        stream_bytes=parser.stream_bytes,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        reasoning_content_chars=reasoning_content_chars,
        reasoning_content_bytes=reasoning_content_bytes,
        final_content_chars=len(final_text),
        final_content_bytes=len(final_text.encode("utf-8")),
        headers_ms=headers_ms,
        first_chunk_ms=first_chunk_ms,
        elapsed_ms=_elapsed_ms(generation_started),
    )
    if finish_reason == "length":
        raise RuntimeError("GENERATION_LENGTH " + stats)
    if final_text.strip():
        _debug("GENERATION_COMPLETE " + stats)
        return final_text
    raise RuntimeError(
        "GENERATION_EMPTY_FINAL " + stats
    )


async def _generate_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    *,
    sleep: Any = asyncio.sleep,
) -> tuple[str, int]:
    started = time.monotonic()
    try:
        async with asyncio.timeout(MAX_GENERATION_SECONDS):
            response, attempts = await _start_generation_request(
                client,
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": WORKER_ROLE},
                        {"role": "user", "content": prompt},
                    ],
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
            headers_ms = _elapsed_ms(started)
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
                        f"GENERATION_HTTP attempts={attempts} status={response.status_code} "
                        f"headers_ms={headers_ms}: {excerpt}"
                    )
                return (
                    await _parse_chat_completion_stream(
                        response,
                        attempts=attempts,
                        generation_started=started,
                        headers_ms=headers_ms,
                    ),
                    attempts,
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
            f"GENERATION_DEADLINE limit_seconds={MAX_GENERATION_SECONDS} "
            f"elapsed_ms={_elapsed_ms(started)}"
        ) from exc


async def run_qwen_worker(
    task: str,
    repo_root: str,
    files: list[str],
    test_output: str | None = None,
) -> str:
    """Run the shared, read-only Qwen implementation worker.

    Both the MCP tool and command-line frontend call this function so they have
    identical file safety, endpoint, model, prompt, and response parsing behavior.
    """
    task = validate_text(
        task,
        field="task",
        max_bytes=MAX_TASK_BYTES,
        max_chars=MAX_TASK_CHARS,
        code="TASK_TOO_LARGE",
    )
    if not task.strip():
        raise ValueError("task must not be empty")
    if test_output is not None:
        test_output = validate_text(
            test_output,
            field="test_output",
            max_bytes=MAX_TEST_OUTPUT_BYTES,
            max_chars=MAX_TEST_OUTPUT_CHARS,
            code="TEST_OUTPUT_TOO_LARGE",
        )
    repo_root = validate_text(
        repo_root,
        field="repo_root",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
        code="PATH_TOO_LONG",
    )
    files = validate_string_list(
        files,
        field="files",
        max_items=MAX_FILES,
        count_code="TOO_MANY_FILES",
        item_kind="path",
        max_bytes=MAX_PATH_BYTES,
        max_chars=MAX_PATH_CHARS,
    )
    ensure_request_budget(
        {
            "task": task,
            "repo_root": repo_root,
            "files": files,
            "test_output": test_output,
        }
    )

    root, supplied = _read_supplied_files(repo_root, files)
    prompt = _build_prompt(task, root, supplied, test_output)
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
        await _preflight(client, base_url, model)
        final_text, _attempts = await _generate_stream(client, base_url, model, prompt)
    if not final_text.endswith("\n"):
        if len(final_text) + 1 > MAX_PATCH_FINAL_CONTENT_CHARS:
            raise RuntimeError("GENERATION_OUTPUT_LIMIT final_content_chars exceeds limit")
        if len(final_text.encode("utf-8")) + 1 > MAX_PATCH_FINAL_CONTENT_BYTES:
            raise RuntimeError("GENERATION_OUTPUT_LIMIT final_content_bytes exceeds limit")
        final_text += "\n"
    _validate_worker_output(
        final_text, allowed_files=[path for path, _source in supplied]
    )
    return final_text


def _configured_base_url() -> str:
    raw = os.environ.get("QWEN_BASE_URL", "")
    if not raw.strip():
        raise RuntimeError("endpoint_not_configured: QWEN_BASE_URL is required")
    try:
        validate_text(
            raw,
            field="QWEN_BASE_URL",
            max_bytes=MAX_ENDPOINT_BYTES,
            max_chars=MAX_ENDPOINT_CHARS,
            code="ENDPOINT_TOO_LONG",
        )
    except InputLimitError as exc:
        raise RuntimeError(str(exc)) from exc
    if raw != raw.strip() or any(character.isspace() for character in raw):
        raise RuntimeError("endpoint_invalid_url: whitespace is not allowed")

    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        # Accessing .port validates malformed and out-of-range ports without
        # allowing the parser exception to escape as an implementation detail.
        _port = parsed.port
        username = parsed.username
        password = parsed.password
    except ValueError as exc:
        raise RuntimeError("endpoint_invalid_url: malformed URL") from exc

    if scheme not in {"http", "https"}:
        raise RuntimeError("endpoint_invalid_scheme: only http and https are allowed")
    if "@" in parsed.netloc or username is not None or password is not None:
        raise RuntimeError("endpoint_credentials_not_allowed: userinfo is forbidden")
    if "?" in raw or "#" in raw:
        raise RuntimeError(
            "endpoint_query_fragment_not_allowed: endpoint must be a clean API base URL"
        )
    if not parsed.netloc or not hostname:
        raise RuntimeError("endpoint_invalid_url: a hostname is required")

    is_loopback = hostname.lower() == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise RuntimeError(
            "endpoint_remote_not_allowed: QWEN_BASE_URL must target loopback"
        )
    return raw.rstrip("/")


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
async def qwen_patch(
    task: str,
    repo_root: str,
    files: list[str],
    test_output: str | None = None,
) -> str:
    """Generate a unified diff from explicitly supplied UTF-8 repository files.

    This tool is read-only with respect to the repository. It performs no shell commands and
    returns Qwen's response text unchanged.
    """
    return await run_qwen_worker(task, repo_root, files, test_output)


if __name__ == "__main__":
    mcp.run(transport="stdio")
