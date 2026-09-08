#!/usr/bin/env python3
"""Strict CLI adapter for the v3 sandbox filesystem agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from qwen_agent import run_qwen_agent
from qwen_input_limits import (
    MAX_ALLOW_SCOPES,
    MAX_ALLOWED_COMMANDS,
    MAX_COMMAND_BYTES,
    MAX_COMMAND_CHARS,
    MAX_PATH_BYTES,
    MAX_PATH_CHARS,
    MAX_READ_ONLY_SCOPES,
    MAX_REQUEST_BYTES,
    MAX_TASK_BYTES,
    MAX_TASK_CHARS,
    MAX_VALIDATION_COMMANDS,
    MAX_VALIDATION_FAILURE_BYTES,
    MAX_VALIDATION_FAILURE_CHARS,
    InputLimitError,
    ensure_request_budget,
    read_bounded,
    read_bounded_file,
    validate_string_list,
    validate_text,
)


class RequestError(ValueError):
    """An invalid qwen-agent invocation or request document."""


def _read_request_bytes(arguments: list[str]) -> bytes:
    if not arguments:
        try:
            return read_bounded(
                sys.stdin.buffer, MAX_REQUEST_BYTES, field="stdin"
            )
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
        except (TypeError, OSError) as exc:
            raise RequestError(f"could not read request file: {exc}") from exc
    raise RequestError("usage: qwen-agent [--request-file PATH] or flag mode")


def _read_task_file(path_value: str) -> str:
    try:
        validate_text(
            path_value,
            field="task file path",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
            code="PATH_TOO_LONG",
        )
        raw = read_bounded_file(
            path_value,
            MAX_TASK_BYTES,
            field="task file",
            code="TASK_TOO_LARGE",
        )
    except InputLimitError as exc:
        raise RequestError(str(exc)) from exc
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    except OSError as exc:
        raise RequestError(f"could not read task file: {exc}") from exc
    try:
        task = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestError("task file must be UTF-8") from exc
    if len(task) > MAX_TASK_CHARS:
        raise RequestError(
            f"TASK_TOO_LARGE field=task observed_chars={len(task)} "
            f"limit={MAX_TASK_CHARS}"
        )
    return task


def _validate_task(task: str) -> None:
    try:
        validate_text(
            task,
            field="task",
            max_bytes=MAX_TASK_BYTES,
            max_chars=MAX_TASK_CHARS,
            code="TASK_TOO_LARGE",
        )
    except InputLimitError as exc:
        raise RequestError(str(exc)) from exc
    if not task.strip():
        raise RequestError("task must not be empty")


def _parse_json_request(raw: bytes) -> dict[str, Any]:
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
    allowed = {
        "task",
        "sandbox",
        "allow",
        "read_only",
        "allowed_commands",
        "validation_commands",
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
    try:
        validate_text(
            request["sandbox"],
            field="sandbox",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
            code="PATH_TOO_LONG",
        )
        request["allow"] = validate_string_list(
            request["allow"],
            field="allow",
            max_items=MAX_ALLOW_SCOPES,
            count_code="TOO_MANY_ALLOW_SCOPES",
            item_kind="path",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
        )
        for field, max_items, count_code, item_kind, max_bytes, max_chars in (
            (
                "read_only",
                MAX_READ_ONLY_SCOPES,
                "TOO_MANY_READ_ONLY_SCOPES",
                "path",
                MAX_PATH_BYTES,
                MAX_PATH_CHARS,
            ),
            (
                "allowed_commands",
                MAX_ALLOWED_COMMANDS,
                "TOO_MANY_COMMANDS",
                "command",
                MAX_COMMAND_BYTES,
                MAX_COMMAND_CHARS,
            ),
            (
                "validation_commands",
                MAX_VALIDATION_COMMANDS,
                "TOO_MANY_VALIDATION_COMMANDS",
                "command",
                MAX_COMMAND_BYTES,
                MAX_COMMAND_CHARS,
            ),
        ):
            if field in request:
                request[field] = validate_string_list(
                    request[field],
                    field=field,
                    max_items=max_items,
                    count_code=count_code,
                    item_kind=item_kind,
                    max_bytes=max_bytes,
                    max_chars=max_chars,
                )
        if "validation_failure" in request:
            request["validation_failure"] = validate_text(
                request["validation_failure"],
                field="validation_failure",
                max_bytes=MAX_VALIDATION_FAILURE_BYTES,
                max_chars=MAX_VALIDATION_FAILURE_CHARS,
                code="VALIDATION_FAILURE_TOO_LARGE",
            )
    except InputLimitError as exc:
        raise RequestError(str(exc)) from exc
    except ValueError as exc:
        raise RequestError(str(exc)) from exc
    if "validation_commands" in request and any(
        not item.strip() for item in request["validation_commands"]
    ):
        raise RequestError("validation_commands must not contain empty strings")
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
    parser.add_argument("--validation-command", action="append", default=[])
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task")
    task_group.add_argument("--task-file")
    parser.add_argument("--validation-failure")
    parser.add_argument("--max-actions", type=int, default=30)
    return parser


def _precheck_flag_arguments(arguments: list[str]) -> None:
    limits = {
        "--allow": (MAX_ALLOW_SCOPES, "TOO_MANY_ALLOW_SCOPES"),
        "--read-only": (MAX_READ_ONLY_SCOPES, "TOO_MANY_READ_ONLY_SCOPES"),
        "--allow-command": (MAX_ALLOWED_COMMANDS, "TOO_MANY_COMMANDS"),
        "--validation-command": (
            MAX_VALIDATION_COMMANDS,
            "TOO_MANY_VALIDATION_COMMANDS",
        ),
    }
    counts = {flag: 0 for flag in limits}
    for argument in arguments:
        flag = argument.split("=", 1)[0]
        if flag not in counts:
            continue
        counts[flag] += 1
        limit, code = limits[flag]
        if counts[flag] > limit:
            raise RequestError(
                f"{code} field={flag.lstrip('-').replace('-', '_')} "
                f"observed_count={counts[flag]} limit={limit}"
            )


def _request_from_flags(arguments: list[str]) -> dict[str, Any]:
    _precheck_flag_arguments(arguments)
    parsed = _flag_parser().parse_args(arguments)
    task = parsed.task if parsed.task is not None else _read_task_file(parsed.task_file)
    _validate_task(task)
    try:
        validate_text(
            parsed.sandbox,
            field="sandbox",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
            code="PATH_TOO_LONG",
        )
        allow = validate_string_list(
            parsed.allow,
            field="allow",
            max_items=MAX_ALLOW_SCOPES,
            count_code="TOO_MANY_ALLOW_SCOPES",
            item_kind="path",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
        )
        read_only = validate_string_list(
            parsed.read_only,
            field="read_only",
            max_items=MAX_READ_ONLY_SCOPES,
            count_code="TOO_MANY_READ_ONLY_SCOPES",
            item_kind="path",
            max_bytes=MAX_PATH_BYTES,
            max_chars=MAX_PATH_CHARS,
        )
        allowed_commands = validate_string_list(
            parsed.allow_command,
            field="allowed_commands",
            max_items=MAX_ALLOWED_COMMANDS,
            count_code="TOO_MANY_COMMANDS",
            item_kind="command",
            max_bytes=MAX_COMMAND_BYTES,
            max_chars=MAX_COMMAND_CHARS,
        )
        validation_commands = validate_string_list(
            parsed.validation_command,
            field="validation_commands",
            max_items=MAX_VALIDATION_COMMANDS,
            count_code="TOO_MANY_VALIDATION_COMMANDS",
            item_kind="command",
            max_bytes=MAX_COMMAND_BYTES,
            max_chars=MAX_COMMAND_CHARS,
        )
        validation_failure = (
            validate_text(
                parsed.validation_failure,
                field="validation_failure",
                max_bytes=MAX_VALIDATION_FAILURE_BYTES,
                max_chars=MAX_VALIDATION_FAILURE_CHARS,
                code="VALIDATION_FAILURE_TOO_LARGE",
            )
            if parsed.validation_failure is not None
            else None
        )
    except InputLimitError as exc:
        raise RequestError(str(exc)) from exc
    if any(not command.strip() for command in validation_commands):
        raise RequestError("validation_commands must not contain empty strings")
    request = {
        "task": task,
        "sandbox": parsed.sandbox,
        "allow": allow,
        "read_only": read_only,
        "allowed_commands": allowed_commands,
        "validation_commands": validation_commands,
        "validation_failure": validation_failure,
        "max_actions": parsed.max_actions,
    }
    try:
        return ensure_request_budget(request)
    except InputLimitError as exc:
        raise RequestError(str(exc)) from exc


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
        validation_commands=request.get("validation_commands"),
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
