"""Shared bounded input readers and request/model-input limits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, BinaryIO

# The envelope is deliberately large enough for ordinary orchestration metadata,
# but small enough that JSON parsing cannot materialize an unbounded document.
MAX_REQUEST_BYTES = 2_000_000

MAX_TASK_BYTES = 480_000
MAX_TASK_CHARS = 120_000
MAX_TEST_OUTPUT_BYTES = 480_000
MAX_TEST_OUTPUT_CHARS = 120_000
MAX_VALIDATION_FAILURE_BYTES = 32_000
MAX_VALIDATION_FAILURE_CHARS = 8_000

MAX_SOURCE_FILES = 24
MAX_ALLOW_SCOPES = 64
MAX_READ_ONLY_SCOPES = 64
MAX_ALLOWED_COMMANDS = 32
MAX_VALIDATION_COMMANDS = 32

MAX_PATH_BYTES = 4_096
MAX_PATH_CHARS = 2_048
MAX_COMMAND_BYTES = 32_768
MAX_COMMAND_CHARS = 8_192
MAX_MODEL_IDENTIFIER_BYTES = 4_096
MAX_MODEL_IDENTIFIER_CHARS = 2_048
MAX_ENDPOINT_BYTES = 8_192
MAX_ENDPOINT_CHARS = 4_096

# Python-character and UTF-8-byte guards are both intentional.  The patch
# source limits fit below these values; the budget still rejects a combined
# task/source/test context before it reaches the model client.
MAX_MODEL_INPUT_CHARS = 500_000
MAX_MODEL_INPUT_BYTES = 2_400_000


class InputLimitError(ValueError):
    """A deterministic input or model-context limit failure."""

    def __init__(
        self,
        code: str,
        field: str,
        observed: int,
        limit: int,
        metric: str,
    ) -> None:
        self.code = code
        self.field = field
        self.observed = observed
        self.limit = limit
        self.metric = metric
        detail = f"{code} field={field} observed_{metric}={observed} limit={limit}"
        if metric == "count":
            detail += f" (at most {limit} items)"
        super().__init__(detail)


def _encoded_length(value: str, *, field: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        # JSON can represent a lone surrogate using an escape even though it
        # cannot be carried to the UTF-8 model request.
        raise InputLimitError("INVALID_UTF8", field, len(value), 0, "chars") from exc


def validate_text(
    value: Any,
    *,
    field: str,
    max_bytes: int,
    max_chars: int,
    code: str | None = None,
) -> str:
    """Validate a string's type and both byte/character limits."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    failure_code = code or f"{field.upper()}_TOO_LARGE"
    if len(value) > max_chars:
        raise InputLimitError(failure_code, field, len(value), max_chars, "chars")
    encoded_length = _encoded_length(value, field=field)
    if encoded_length > max_bytes:
        raise InputLimitError(
            failure_code, field, encoded_length, max_bytes, "bytes"
        )
    return value


def validate_string_list(
    value: Any,
    *,
    field: str,
    max_items: int,
    count_code: str,
    item_kind: str,
    max_bytes: int,
    max_chars: int,
) -> list[str]:
    """Validate list cardinality before inspecting or processing its items."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of strings")
    if len(value) > max_items:
        raise InputLimitError(count_code, field, len(value), max_items, "count")
    checked: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must be an array of strings")
        checked.append(
            validate_text(
                item,
                field=f"{item_kind} item",
                max_bytes=max_bytes,
                max_chars=max_chars,
                code=(
                    "PATH_TOO_LONG"
                    if item_kind == "path"
                    else "COMMAND_TOO_LONG"
                ),
            )
        )
    return checked


def read_bounded(
    stream: BinaryIO,
    max_bytes: int,
    *,
    field: str,
    code: str = "REQUEST_TOO_LARGE",
) -> bytes:
    """Read at most ``max_bytes + 1`` bytes, handling short reads safely."""
    if max_bytes < 0:
        raise ValueError("max_bytes must not be negative")
    data = bytearray()
    while len(data) <= max_bytes:
        remaining = max_bytes + 1 - len(data)
        chunk = stream.read(remaining)
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise ValueError(f"{field} stream must return bytes")
        if len(chunk) > remaining:
            raise InputLimitError(code, field, max_bytes + 1, max_bytes, "bytes")
        chunk_bytes = bytes(chunk)
        if not chunk_bytes:
            break
        data.extend(chunk_bytes)
        if len(data) > max_bytes:
            raise InputLimitError(code, field, len(data), max_bytes, "bytes")
    return bytes(data)


def read_bounded_file(
    path: str | Path,
    max_bytes: int,
    *,
    field: str,
    code: str = "REQUEST_TOO_LARGE",
) -> bytes:
    """Stat-check then bounded-read a file, retaining the race-safe read guard."""
    file_path = Path(path)
    try:
        observed = file_path.stat().st_size
    except OSError:
        raise
    if observed > max_bytes:
        raise InputLimitError(code, field, observed, max_bytes, "bytes")
    with file_path.open("rb") as stream:
        return read_bounded(stream, max_bytes, field=field, code=code)


def ensure_model_input(prompt: str, *, field: str = "model_input") -> str:
    """Reject, rather than truncate, a fully assembled model prompt."""
    if not isinstance(prompt, str):
        raise ValueError(f"{field} must be a string")
    if len(prompt) > MAX_MODEL_INPUT_CHARS:
        raise InputLimitError(
            "MODEL_INPUT_TOO_LARGE",
            field,
            len(prompt),
            MAX_MODEL_INPUT_CHARS,
            "chars",
        )
    encoded_length = _encoded_length(prompt, field=field)
    if encoded_length > MAX_MODEL_INPUT_BYTES:
        raise InputLimitError(
            "MODEL_INPUT_TOO_LARGE",
            field,
            encoded_length,
            MAX_MODEL_INPUT_BYTES,
            "bytes",
        )
    return prompt


def ensure_request_budget(request: dict[str, Any]) -> dict[str, Any]:
    """Apply the same envelope budget to direct and flag-mode callers."""
    try:
        encoded_length = len(
            json.dumps(
                request,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, UnicodeEncodeError) as exc:
        raise InputLimitError("INVALID_UTF8", "request", 0, 0, "bytes") from exc
    if encoded_length > MAX_REQUEST_BYTES:
        raise InputLimitError(
            "REQUEST_TOO_LARGE",
            "request",
            encoded_length,
            MAX_REQUEST_BYTES,
            "bytes",
        )
    return request
