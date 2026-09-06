#!/usr/bin/env python3
"""Strict JSON command-line adapter for the shared Qwen worker."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from qwen_worker_server import run_qwen_worker


class RequestError(ValueError):
    """An invalid CLI invocation or request document."""


def _request_bytes(arguments: list[str]) -> bytes:
    if not arguments:
        try:
            return sys.stdin.buffer.read()
        except OSError as exc:
            raise RequestError(f"could not read stdin: {exc}") from exc

    if len(arguments) == 2 and arguments[0] == "--request-file":
        try:
            return Path(arguments[1]).read_bytes()
        except OSError as exc:
            raise RequestError(f"could not read request file: {exc}") from exc

    raise RequestError("usage: qwen-patch [--request-file PATH]")


def _parse_request(raw: bytes) -> dict[str, Any]:
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
    if not isinstance(request["repo_root"], str):
        raise RequestError("repo_root must be a string")
    files = request["files"]
    if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
        raise RequestError("files must be an array of strings")
    if "test_output" in request and not isinstance(request["test_output"], str):
        raise RequestError("test_output must be a string")

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
