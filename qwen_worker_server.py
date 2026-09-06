#!/usr/bin/env python3
"""Read-only MCP bridge from Codex to a local OpenAI-compatible Qwen server."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from collections.abc import Mapping
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


DEFAULT_BASE_URL: str | None = None
DEFAULT_MODEL = (
    "/Users/ertugrulcetinkaya/Models/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/"
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
)
MAX_FILES = 24
MAX_SOURCE_CHARS = 260_000
MAX_SOURCE_BYTES = 1_100_000
MAX_OUTPUT_TOKENS = 4_096
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5
MAX_ERROR_BODY_CHARS = 1_000
MAX_CONNECT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
PREFLIGHT_MAX_CONNECT_ATTEMPTS = 5
PREFLIGHT_RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0, 4.0)
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, pool=10.0, write=30.0, read=1_800.0)
PREFLIGHT_TIMEOUT = httpx.Timeout(connect=3.0, pool=3.0, write=3.0, read=5.0)
RETRYABLE_CONNECT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.ReadTimeout,
)
RETRYABLE_PREFLIGHT_ERRORS = RETRYABLE_CONNECT_ERRORS + (httpx.ReadError,)

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
    if not files:
        raise ValueError("files must contain at least one repository-relative path")
    if len(files) > MAX_FILES:
        raise ValueError(f"at most {MAX_FILES} files may be supplied")

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

        file_size = resolved.stat().st_size
        total_bytes += file_size
        if total_bytes > MAX_SOURCE_BYTES:
            raise ValueError("supplied files exceed the safety byte limit")

        data = resolved.read_bytes()
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
    sections = [
        "IMPLEMENTATION TASK:",
        task,
        "",
        f"REPOSITORY ROOT (context only; do not access it): {root}",
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
        "Produce only a unified diff for the requested implementation, or the specified "
        "NEED_FILES response if these files are insufficient."
    )
    return "\n".join(sections)


def _response_excerpt(response: httpx.Response) -> str:
    """Return a bounded, single-line response excerpt for error diagnostics."""
    text = " ".join(response.text.split())
    if not text:
        return "<empty response body>"
    if len(text) > MAX_ERROR_BODY_CHARS:
        return f"{text[:MAX_ERROR_BODY_CHARS]}..."
    return text


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


def _validate_worker_output(output: str) -> None:
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
    hunk_header = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
    )
    index = 0
    file_count = 0
    while index < len(lines):
        while index < len(lines) and (
            lines[index].startswith("diff --git ")
            or lines[index].startswith("index ")
            or lines[index].startswith("new file mode ")
            or lines[index].startswith("deleted file mode ")
            or lines[index].startswith("old mode ")
            or lines[index].startswith("new mode ")
            or lines[index].startswith("similarity index ")
            or lines[index].startswith("rename from ")
            or lines[index].startswith("rename to ")
        ):
            index += 1
        if index >= len(lines) or not lines[index].startswith("--- "):
            raise RuntimeError("GENERATION_INVALID_OUTPUT: missing unified diff header")
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise RuntimeError("GENERATION_INVALID_OUTPUT: incomplete unified diff header")
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


async def _request_with_connect_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    stage: str,
    stream: bool = False,
    json: dict[str, Any] | None = None,
    timeout: httpx.Timeout | None = None,
    sleep: Any = asyncio.sleep,
    max_attempts: int = MAX_CONNECT_ATTEMPTS,
    retry_backoff_seconds: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
    retryable_errors: tuple[type[httpx.HTTPError], ...] = RETRYABLE_CONNECT_ERRORS,
) -> tuple[httpx.Response, int]:
    """Establish a response, retrying only failures known to precede headers."""
    started = time.monotonic()
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if len(retry_backoff_seconds) < max_attempts - 1:
        raise ValueError("retry_backoff_seconds must cover every retry")

    for attempt in range(1, max_attempts + 1):
        request = client.build_request(method, url, json=json, timeout=timeout)
        try:
            response = await client.send(request, stream=stream)
            return response, attempt
        except retryable_errors as exc:
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


async def _preflight(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    *,
    sleep: Any = asyncio.sleep,
) -> int:
    started = time.monotonic()
    response, attempts = await _request_with_connect_retries(
        client,
        "GET",
        f"{base_url}/models",
        stage="PREFLIGHT_CONNECT",
        timeout=PREFLIGHT_TIMEOUT,
        sleep=sleep,
        max_attempts=PREFLIGHT_MAX_CONNECT_ATTEMPTS,
        retry_backoff_seconds=PREFLIGHT_RETRY_BACKOFF_SECONDS,
        retryable_errors=RETRYABLE_PREFLIGHT_ERRORS,
    )
    if not 200 <= response.status_code < 300:
        raise RuntimeError(
            f"endpoint_http_error: PREFLIGHT_HTTP attempts={attempts} "
            f"status={response.status_code} "
            f"elapsed_ms={_elapsed_ms(started)}: {_response_excerpt(response)}"
        )
    try:
        payload = response.json()
    except (JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"PREFLIGHT_MALFORMED attempts={attempts} status={response.status_code} "
            f"elapsed_ms={_elapsed_ms(started)}: response is not valid JSON"
        ) from exc
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


async def _parse_chat_completion_stream(
    response: httpx.Response,
    *,
    attempts: int,
    generation_started: float,
    headers_ms: int,
) -> str:
    final_parts: list[str] = []
    reasoning_content_chars = 0
    finish_reason: Any = None
    completion_tokens: Any = None
    event_count = 0
    first_chunk_ms: int | None = None

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
            if first_chunk_ms is None:
                first_chunk_ms = _elapsed_ms(generation_started)
            try:
                payload = httpx.Response(200, text=data).json()
            except (JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={event_count} "
                    f"headers_ms={headers_ms} first_chunk_ms={first_chunk_ms} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: invalid SSE JSON"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={event_count} "
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
                    f"GENERATION_MALFORMED attempts={attempts} events={event_count} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: missing streamed choice"
                )
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                raise RuntimeError(
                    f"GENERATION_MALFORMED attempts={attempts} events={event_count} "
                    f"elapsed_ms={_elapsed_ms(generation_started)}: missing delta object"
                )
            if "content" in delta:
                final_parts.append(_stream_text(delta["content"], "delta.content"))
            for alternate in ("text", "output_text"):
                if alternate in delta:
                    final_parts.append(_stream_text(delta[alternate], f"delta.{alternate}"))
            if "reasoning_content" in delta:
                reasoning_content_chars += len(
                    _stream_text(delta["reasoning_content"], "delta.reasoning_content")
                )
    except httpx.ReadTimeout as exc:
        raise RuntimeError(
            f"GENERATION_TIMEOUT attempts={attempts} events={event_count} "
            f"headers_ms={headers_ms} first_chunk_ms={first_chunk_ms} "
            f"elapsed_ms={_elapsed_ms(generation_started)}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"GENERATION_STREAM attempts={attempts} events={event_count} "
            f"headers_ms={headers_ms} first_chunk_ms={first_chunk_ms} "
            f"elapsed_ms={_elapsed_ms(generation_started)}: {type(exc).__name__}: {exc}"
        ) from exc

    final_text = "".join(final_parts)
    stats = _metadata(
        attempts=attempts,
        retry_count=attempts - 1,
        events=event_count,
        finish_reason=finish_reason,
        completion_tokens=completion_tokens,
        reasoning_content_chars=reasoning_content_chars,
        final_content_chars=len(final_text),
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
    response, attempts = await _request_with_connect_retries(
        client,
        "POST",
        f"{base_url}/chat/completions",
        stage="GENERATION_CONNECT",
        stream=True,
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
            try:
                await response.aread()
            except httpx.HTTPError:
                pass
            raise RuntimeError(
                f"GENERATION_HTTP attempts={attempts} status={response.status_code} "
                f"headers_ms={headers_ms}: {_response_excerpt(response)}"
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
    if not task.strip():
        raise ValueError("task must not be empty")

    root, supplied = _read_supplied_files(repo_root, files)
    prompt = _build_prompt(task, root, supplied, test_output)
    base_url = _configured_base_url()
    model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        await _preflight(client, base_url, model)
        final_text, _attempts = await _generate_stream(client, base_url, model, prompt)
    if not final_text.endswith("\n"):
        final_text += "\n"
    _validate_worker_output(final_text)
    return final_text


def _configured_base_url() -> str:
    configured = os.environ.get("QWEN_BASE_URL", "").strip()
    if not configured:
        raise RuntimeError("endpoint_not_configured: QWEN_BASE_URL is required")
    return configured.rstrip("/")


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
