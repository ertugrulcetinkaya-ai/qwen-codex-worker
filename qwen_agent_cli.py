#!/usr/bin/env python3
"""Strict CLI adapter for the v3 sandbox filesystem agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from qwen_agent import MAX_TASK_BYTES, MAX_TASK_CHARS, run_qwen_agent


class RequestError(ValueError):
    """An invalid qwen-agent invocation or request document."""


def _read_request_bytes(arguments: list[str]) -> bytes:
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
    raise RequestError("usage: qwen-agent [--request-file PATH] or flag mode")


def _read_task_file(path_value: str) -> str:
    try:
        raw = Path(path_value).read_bytes()
    except OSError as exc:
        raise RequestError(f"could not read task file: {exc}") from exc
    if len(raw) > MAX_TASK_BYTES:
        raise RequestError("task_too_large: task file exceeds the byte limit")
    try:
        task = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestError("task file must be UTF-8") from exc
    if len(task) > MAX_TASK_CHARS:
        raise RequestError("task_too_large: task file exceeds the character limit")
    return task


def _validate_task(task: str) -> None:
    if not task.strip():
        raise RequestError("task must not be empty")
    if len(task.encode("utf-8")) > MAX_TASK_BYTES or len(task) > MAX_TASK_CHARS:
        raise RequestError("task_too_large: task exceeds the bounded task size")


def _parse_json_request(raw: bytes) -> dict[str, Any]:
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
    allowed = {
        "task",
        "sandbox",
        "allow",
        "read_only",
        "allowed_commands",
        "validation_failure",
        "max_actions",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise RequestError(f"unknown request field: {unknown[0]}")
    for field in ("task", "sandbox", "allow"):
        if field not in request:
            raise RequestError(f"missing required field: {field}")
    if not isinstance(request["task"], str):
        raise RequestError("task must be a string")
    _validate_task(request["task"])
    if not isinstance(request["sandbox"], str):
        raise RequestError("sandbox must be a string")
    if not isinstance(request["allow"], list) or any(
        not isinstance(item, str) for item in request["allow"]
    ):
        raise RequestError("allow must be an array of strings")
    for field in ("read_only", "allowed_commands"):
        if field in request and (
            not isinstance(request[field], list)
            or any(not isinstance(item, str) for item in request[field])
        ):
            raise RequestError(f"{field} must be an array of strings")
    if "validation_failure" in request and not isinstance(
        request["validation_failure"], str
    ):
        raise RequestError("validation_failure must be a string")
    if "max_actions" in request and (
        isinstance(request["max_actions"], bool)
        or not isinstance(request["max_actions"], int)
    ):
        raise RequestError("max_actions must be an integer")
    return request


def _flag_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-agent",
        description="Run Qwen in a constrained sandbox filesystem.",
    )
    parser.add_argument("--sandbox", required=True)
    parser.add_argument("--allow", action="append", required=True)
    parser.add_argument("--read-only", action="append", default=[])
    parser.add_argument("--allow-command", action="append", default=[])
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task")
    task_group.add_argument("--task-file")
    parser.add_argument("--validation-failure")
    parser.add_argument("--max-actions", type=int, default=30)
    return parser


def _request_from_flags(arguments: list[str]) -> dict[str, Any]:
    parsed = _flag_parser().parse_args(arguments)
    task = parsed.task if parsed.task is not None else _read_task_file(parsed.task_file)
    _validate_task(task)
    return {
        "task": task,
        "sandbox": parsed.sandbox,
        "allow": parsed.allow,
        "read_only": parsed.read_only,
        "allowed_commands": parsed.allow_command,
        "validation_failure": parsed.validation_failure,
        "max_actions": parsed.max_actions,
    }


def _load_request(arguments: list[str]) -> dict[str, Any]:
    if arguments and arguments[0] not in {"--request-file"}:
        return _request_from_flags(arguments)
    return _parse_json_request(_read_request_bytes(arguments))


async def _run(arguments: list[str]) -> dict[str, Any]:
    request = _load_request(arguments)
    result = await run_qwen_agent(
        task=request["task"],
        sandbox=request["sandbox"],
        allow=request["allow"],
        read_only=request.get("read_only"),
        allowed_commands=request.get("allowed_commands"),
        validation_failure=request.get("validation_failure"),
        max_actions=request.get("max_actions", 30),
    )
    return result.as_dict()


def main(arguments: list[str] | None = None) -> int:
    logging.disable(logging.CRITICAL)
    selected = sys.argv[1:] if arguments is None else arguments
    try:
        result = asyncio.run(_run(selected))
    except RequestError as exc:
        diagnostic = " ".join(str(exc).split()) or type(exc).__name__
        print(
            "[qwen-agent] stage=PRECHECK status=error kind=request_error",
            file=sys.stderr,
        )
        print(f"qwen-agent: {diagnostic}", file=sys.stderr)
        return 1
    except Exception as exc:
        diagnostic = " ".join(str(exc).split()) or type(exc).__name__
        print(f"qwen-agent: {diagnostic}", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
