#!/usr/bin/env python3
"""Strict JSON command-line adapter for the shared Qwen worker."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from qwen_input_limits import (
    MAX_PATH_BYTES,
    MAX_PATH_CHARS,
    MAX_REQUEST_BYTES,
    MAX_SOURCE_FILES,
    MAX_TASK_BYTES,
    MAX_TASK_CHARS,
    MAX_TEST_OUTPUT_BYTES,
    MAX_TEST_OUTPUT_CHARS,
    InputLimitError,
    read_bounded,
    read_bounded_file,
    validate_string_list,
    validate_text,
)
from qwen_worker_server import run_qwen_worker


class RequestError(ValueError):
    """An invalid CLI invocation or request document."""


def _request_bytes(arguments: list[str]) -> bytes:
    if not arguments:
        try:
            return read_bounded(sys.stdin.buffer, MAX_REQUEST_BYTES, field="stdin")
        except InputLimitError as exc:
            raise RequestError(str(exc)) from exc
        except OSError as exc:
            raise RequestError(f"could not read stdin: {exc}") from exc

    if len(arguments) == 2 and arguments[0] == "--request-file":
        try:
            validate_text(
                arguments[1],
                field="request file path",
                max_bytes=MAX_PATH_BYTES,
                max_chars=MAX_PATH_CHARS,
                code="PATH_TOO_LONG",
            )
            return read_bounded_file(
                arguments[1], MAX_REQUEST_BYTES, field="request file"
            )
        except InputLimitError as exc:
            raise RequestError(str(exc)) from exc
        except OSError as exc:
            raise RequestError(f"could not read request file: {exc}") from exc

    raise RequestError("usage: qwen-patch [--request-file PATH]")


def _parse_request(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_REQUEST_BYTES:
        raise RequestError(
            f"REQUEST_TOO_LARGE field=request observed_bytes={len(raw)} "
            f"limit={MAX_REQUEST_BYTES}"
        )
    try:
        document = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestError("request must be UTF-8 JSON") from exc

    try:
        request = json.loads(document)
    except json.JSONDecodeError as exc:
        raise RequestError(
            f"malformed JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(request, dict):
        raise RequestError("request must be a JSON object")

    allowed = {"task", "repo_root", "files", "test_output"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise RequestError(f"unknown request field: {unknown[0]}")

    for field in ("task", "repo_root", "files"):
        if field not in request:
            raise RequestError(f"missing required field: {field}")

    if not isinstance(request["task"], str):
        raise RequestError("task must be a string")
    try:
        request["task"] = validate_text(
            request["task"],
            field="task",
            max_bytes=MAX_TASK_BYTES,
            max_chars=MAX_TASK_CHARS,
            code="TASK_TOO_LARGE",
        )
        if not request["task"].strip():
            raise RequestError("task must not be empty")
        request["repo_root"] = validate_text(
            request["repo_root"],
            field="repo_root",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
            code="PATH_TOO_LONG",
        )
        request["files"] = validate_string_list(
            request["files"],
            field="files",
            max_items=MAX_SOURCE_FILES,
            count_code="TOO_MANY_FILES",
            item_kind="path",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
        )
        if "test_output" in request:
            request["test_output"] = validate_text(
                request["test_output"],
                field="test_output",
                max_bytes=MAX_TEST_OUTPUT_BYTES,
                max_chars=MAX_TEST_OUTPUT_CHARS,
                code="TEST_OUTPUT_TOO_LARGE",
            )
    except InputLimitError as exc:
        raise RequestError(str(exc)) from exc
    except ValueError as exc:
        raise RequestError(str(exc)) from exc

    return request


async def _run(arguments: list[str]) -> str:
    request = _parse_request(_request_bytes(arguments))
    return await run_qwen_worker(
        task=request["task"],
        repo_root=request["repo_root"],
        files=request["files"],
        test_output=request.get("test_output"),
    )


def main(arguments: list[str] | None = None) -> int:
    # FastMCP configures application logging for the server process. The CLI has
    # a stricter stream contract, so suppress inherited library logs and emit
    # failures explicitly below.
    logging.disable(logging.CRITICAL)
    try:
        response = asyncio.run(_run(sys.argv[1:] if arguments is None else arguments))
    except Exception as exc:
        diagnostic = " ".join(str(exc).split()) or type(exc).__name__
        print(f"qwen-patch: {diagnostic}", file=sys.stderr)
        return 1

    sys.stdout.buffer.write(response.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
