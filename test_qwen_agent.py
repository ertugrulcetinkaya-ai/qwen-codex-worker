import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

import qwen_agent
import qwen_agent_cli
from qwen_agent import (
    AgentError,
    AgentGeneration,
    SandboxAgent,
    _generate_agent_turn,
    parse_action,
    run_qwen_agent,
)
from qwen_worker_server import DEFAULT_MODEL


def _sse_response(*events, status=200):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream"},
        content=body.encode("utf-8"),
    )


async def _fake_preflight(*_args, **_kwargs):
    return 1


class SandboxActionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "example.txt"
        self.source.write_text("hello world\n", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_read_allowed_file(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        result = agent.execute({"action": "read", "path": "example.txt"})
        self.assertEqual(result["content"], "hello world\n")
        self.assertEqual(result["path"], "example.txt")

    def test_reject_read_outside_sandbox(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        agent = SandboxAgent(self.root, ["example.txt"])
        with self.assertRaisesRegex(AgentError, "inside the sandbox|allowlist"):
            agent.execute({"action": "read", "path": str(outside)})

    def test_reject_path_traversal(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        with self.assertRaisesRegex(AgentError, "inside the sandbox"):
            agent.execute({"action": "read", "path": "../outside.txt"})

    def test_reject_symlink_escape(self):
        outside = self.root.parent / "outside-link.txt"
        outside.write_text("secret", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        (self.root / "link.txt").symlink_to(outside)
        with self.assertRaisesRegex(AgentError, "escapes sandbox"):
            SandboxAgent(self.root, ["link.txt"])

    def test_replace_unique_text(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        result = agent.execute(
            {
                "action": "replace",
                "path": "example.txt",
                "old": "hello world",
                "new": "hello qwen",
            }
        )
        self.assertEqual(result["replaced"], 1)
        self.assertEqual(self.source.read_text(encoding="utf-8"), "hello qwen\n")

    def test_replace_zero_match_is_rejected(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        with self.assertRaisesRegex(AgentError, "not found"):
            agent.execute(
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "missing",
                    "new": "replacement",
                }
            )

    def test_replace_ambiguous_match_is_rejected(self):
        self.source.write_text("same\nsame\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["example.txt"])
        with self.assertRaisesRegex(AgentError, "exactly once"):
            agent.execute(
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "same",
                    "new": "changed",
                }
            )

    def test_write_allowed_new_file(self):
        agent = SandboxAgent(self.root, ["new.txt"])
        result = agent.execute(
            {
                "action": "write",
                "path": "new.txt",
                "content": "agent mode works\n",
            }
        )
        self.assertEqual(result["path"], "new.txt")
        self.assertEqual(
            (self.root / "new.txt").read_text(encoding="utf-8"),
            "agent mode works\n",
        )

    def test_write_allowed_new_file_in_existing_nested_directory(self):
        nested = self.root / "nested"
        nested.mkdir()
        agent = SandboxAgent(self.root, ["nested/new.txt"])

        result = agent.execute(
            {
                "action": "write",
                "path": "nested/new.txt",
                "content": "hello\n",
            }
        )

        self.assertEqual(result["path"], "nested/new.txt")
        self.assertEqual((nested / "new.txt").read_text(encoding="utf-8"), "hello\n")
        self.assertIn("nested/new.txt", agent.changed_paths)

    def test_read_nonexistent_file_is_rejected(self):
        agent = SandboxAgent(self.root, ["missing.txt"])
        with self.assertRaisesRegex(AgentError, "invalid sandbox path|does not exist"):
            agent.execute({"action": "read", "path": "missing.txt"})

    def test_replace_nonexistent_file_is_rejected(self):
        agent = SandboxAgent(self.root, ["missing.txt"])
        with self.assertRaisesRegex(AgentError, "invalid sandbox path|does not exist"):
            agent.execute(
                {
                    "action": "replace",
                    "path": "missing.txt",
                    "old": "missing",
                    "new": "replacement",
                }
            )

    def test_parent_symlink_escape_is_rejected(self):
        outside = self.root.parent / "outside-directory"
        outside.mkdir()
        self.addCleanup(outside.rmdir)
        parent_link = self.root / "nested-link"
        parent_link.symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(AgentError, "escapes sandbox"):
            SandboxAgent(self.root, ["nested-link/new.txt"])

    def test_write_non_allowlisted_file_is_rejected(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        with self.assertRaisesRegex(AgentError, "allowlist"):
            agent.execute(
                {
                    "action": "write",
                    "path": "other.txt",
                    "content": "must not write",
                }
            )

    def test_command_allowlist_accepts_approved_command(self):
        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            allowed_commands=["git diff"],
        )
        result = agent.execute({"action": "run", "command": "git diff"})
        self.assertEqual(result["command"], "git diff")

    def test_command_allowlist_rejects_shell_and_network_commands(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        with self.assertRaisesRegex(AgentError, "shell operators"):
            agent.execute({"action": "run", "command": "git diff; cat example.txt"})
        with self.assertRaisesRegex(AgentError, "network command"):
            agent.execute({"action": "run", "command": "curl https://example.com"})
        with self.assertRaisesRegex(AgentError, "git apply"):
            agent.execute({"action": "run", "command": "git apply change.patch"})

    def test_forbidden_files_are_rejected(self):
        (self.root / ".env").write_text("SECRET=value\n", encoding="utf-8")
        with self.assertRaisesRegex(AgentError, "forbidden"):
            SandboxAgent(self.root, [".env"])


class AgentProtocolTests(unittest.TestCase):
    def test_invalid_json_action_is_rejected(self):
        with self.assertRaisesRegex(AgentError, "valid JSON"):
            parse_action("not json")

    def test_fenced_json_action_is_accepted(self):
        action = parse_action(
            '```json\n{"action":"done","summary":"fenced"}\n```'
        )

        self.assertEqual(action, {"action": "done", "summary": "fenced"})

    def test_done_action_is_compact(self):
        agent = SandboxAgent(Path(tempfile.mkdtemp()), ["new.txt"])
        try:
            result = agent.execute(
                {"action": "done", "summary": "implemented in sandbox"}
            )
            self.assertEqual(result["action"], "done")
            self.assertLessEqual(len(result["summary"]), 500)
        finally:
            agent.root.rmdir()

    def test_response_format_requires_action_specific_fields(self):
        schema = qwen_agent.AGENT_RESPONSE_FORMAT["schema"]
        alternatives = schema["oneOf"]
        self.assertEqual(len(alternatives), 5)
        self.assertNotIn("$ref", json.dumps(schema))
        self.assertNotIn("$defs", json.dumps(schema))

        expected_required = {
            "read": ["action", "path"],
            "grep": ["action", "query"],
            "replace": ["action", "path", "old", "new"],
            "write": ["action", "path", "content"],
            "done": ["action", "summary"],
        }
        for alternative in alternatives:
            action = alternative["properties"]["action"]["enum"][0]
            self.assertEqual(alternative["required"], expected_required[action])
            self.assertFalse(alternative["additionalProperties"])

    def test_stdout_status_contains_no_unified_diff(self):
        result = qwen_agent.AgentResult(
            status="done",
            actions=2,
            files_changed=["example.txt"],
            model_calls=2,
            completion_tokens=10,
            reasoning_chars=0,
            elapsed_ms=12,
            read_bytes=12,
            write_bytes=11,
        )
        with mock.patch.object(qwen_agent_cli, "run_qwen_agent", return_value=result):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    qwen_agent_cli.main(
                        [
                            "--sandbox",
                            tempfile.gettempdir(),
                            "--allow",
                            "example.txt",
                            "--task",
                            "test",
                        ]
                    ),
                    0,
                )
        document = json.loads(output.getvalue())
        self.assertEqual(document["status"], "done")
        self.assertNotIn("--- ", output.getvalue())
        self.assertNotIn("+++ ", output.getvalue())


class AgentTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.endpoint_environment = mock.patch.dict(
            "os.environ",
            {"QWEN_BASE_URL": "http://qwen.test/v1"},
        )
        self.endpoint_environment.start()

    def tearDown(self):
        self.endpoint_environment.stop()

    async def test_finish_reason_length_fails(self):
        async def handler(_request):
            return _sse_response(
                {
                    "choices": [
                        {"delta": {"content": "{}"}, "finish_reason": "length"}
                    ]
                }
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_LENGTH"):
                await _generate_agent_turn(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    [{"role": "user", "content": "task"}],
                )

    async def test_reasoning_is_disabled_in_request_payload(self):
        async def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(body["response_format"], qwen_agent.AGENT_RESPONSE_FORMAT)
            self.assertEqual(body["temperature"], 0.7)
            self.assertEqual(body["top_p"], 0.8)
            self.assertEqual(body["top_k"], 20)
            return _sse_response(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "private",
                                "content": '{"action":"done","summary":"ok"}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                },
                {"choices": [], "usage": {"completion_tokens": 7}},
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            result = await _generate_agent_turn(
                client,
                "http://qwen.test/v1",
                DEFAULT_MODEL,
                [{"role": "user", "content": "task"}],
            )
        self.assertEqual(result.reasoning_chars, len("private"))
        self.assertEqual(result.completion_tokens, 7)

    async def test_invalid_action_receives_protocol_error_then_done(self):
        with tempfile.TemporaryDirectory() as directory:
            generations = iter(
                [
                    AgentGeneration("not json", 2, 0, 1),
                    AgentGeneration(
                        '{"action":"done","summary":"corrected"}', 2, 0, 1
                    ),
                ]
            )
            captured_messages = []

            async def fake_turn(_client, _base, _model, messages):
                captured_messages.append(list(messages))
                return next(generations)

            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ):
                result = await run_qwen_agent(
                    "repair protocol",
                    directory,
                    ["example.txt"],
                    max_actions=3,
                )
            self.assertEqual(result.status, "done")
            self.assertIn("valid JSON", captured_messages[1][-1]["content"])

    async def test_done_returns_concise_status(self):
        with tempfile.TemporaryDirectory() as directory:
            async def fake_turn(*_args):
                return AgentGeneration(
                    '{"action":"done","summary":"finished"}', 3, 0, 1
                )

            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ):
                result = await run_qwen_agent(
                    "finish task",
                    directory,
                    ["example.txt"],
                    max_actions=2,
                )
        self.assertEqual(result.as_dict()["status"], "done")
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(result.reasoning_chars, 0)

    async def test_filesystem_changes_persist_in_sandbox(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "example.txt")
            source.write_text("hello world\n", encoding="utf-8")
            generations = iter(
                [
                    AgentGeneration(
                        json.dumps(
                            {
                                "action": "replace",
                                "path": "example.txt",
                                "old": "hello world",
                                "new": "hello qwen",
                            }
                        ),
                        4,
                        0,
                        1,
                    ),
                    AgentGeneration(
                        '{"action":"done","summary":"replaced text"}', 2, 0, 1
                    ),
                ]
            )

            async def fake_turn(*_args):
                return next(generations)

            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ):
                result = await run_qwen_agent(
                    "replace text",
                    directory,
                    ["example.txt"],
                )
            self.assertEqual(source.read_text(encoding="utf-8"), "hello qwen\n")
        self.assertEqual(result.files_changed, ["example.txt"])

    async def test_repair_invocation_reuses_existing_sandbox_state(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "example.txt")
            source.write_text("hello world\n", encoding="utf-8")

            generations = iter(
                [
                    AgentGeneration(
                        '{"action":"replace","path":"example.txt","old":"hello world","new":"hello qwen"}',
                        3,
                        0,
                        1,
                    ),
                    AgentGeneration(
                        '{"action":"done","summary":"initial complete"}', 2, 0, 1
                    ),
                ]
            )

            async def first_turn(*_args):
                return next(generations)

            repair_messages = []

            async def done_turn(_client, _base, _model, messages):
                repair_messages.append(messages)
                return AgentGeneration(
                    '{"action":"done","summary":"repair complete"}', 2, 0, 1
                )

            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=first_turn
            ):
                await run_qwen_agent("initial", directory, ["example.txt"])
            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=done_turn
            ):
                result = await run_qwen_agent(
                    "fix validation failure",
                    directory,
                    ["example.txt"],
                    validation_failure="the existing sandbox state is ready",
                )
            self.assertEqual(result.status, "done")
            self.assertIn(
                "CURRENT VALIDATION FAILURE",
                repair_messages[0][1]["content"],
            )
            self.assertEqual(source.read_text(encoding="utf-8"), "hello qwen\n")

    async def test_failure_emits_failed_stage_and_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            with mock.patch(
                "qwen_agent._preflight",
                side_effect=RuntimeError("endpoint_unreachable: refused"),
            ), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(RuntimeError, "endpoint_unreachable"):
                    await run_qwen_agent("preflight failure", directory, ["example.txt"])

        diagnostic = stderr.getvalue()
        self.assertIn("stage=ENDPOINT_CONNECT status=error", diagnostic)
        self.assertIn("stage=DONE status=error", diagnostic)
        self.assertIn("failed_stage=ENDPOINT_CONNECT", diagnostic)
        self.assertIn("model_calls=0", diagnostic)


class AgentCliTaskFileTests(unittest.TestCase):
    def test_task_file_preserves_utf8_and_newline(self):
        result = qwen_agent.AgentResult(
            status="done",
            actions=1,
            files_changed=[],
            model_calls=1,
            completion_tokens=3,
            reasoning_chars=0,
            elapsed_ms=1,
            read_bytes=0,
            write_bytes=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory, "task.txt")
            task = "Türkçe görev\nİkinci satır\n"
            task_path.write_text(task, encoding="utf-8")
            with mock.patch.object(
                qwen_agent_cli, "run_qwen_agent", return_value=result
            ) as run_agent:
                self.assertEqual(
                    qwen_agent_cli.main(
                        [
                            "--sandbox",
                            directory,
                            "--allow",
                            "example.txt",
                            "--task-file",
                            str(task_path),
                        ]
                    ),
                    0,
                )

        self.assertEqual(run_agent.call_args.kwargs["task"], task)


if __name__ == "__main__":
    unittest.main()
