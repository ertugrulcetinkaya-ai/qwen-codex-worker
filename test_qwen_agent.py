import asyncio
import base64
import contextlib
import io
import json
import os
import shlex
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import httpx

import qwen_agent
import qwen_agent_cli
import qwen_input_limits
import qwen_worker_server
from qwen_agent import (
    MAX_AGENT_HISTORY_CHARS,
    MAX_AGENT_RECENT_INTERACTIONS,
    MAX_VALIDATION_ROUNDS,
    AgentConversation,
    AgentError,
    AgentGeneration,
    SandboxAgent,
    ValidationPhase,
    ValidationResult,
    _generate_agent_turn,
    _parse_agent_stream,
    parse_action,
    run_qwen_agent,
)
from qwen_worker_server import (
    DEFAULT_MODEL,
    MAX_ERROR_BODY_BYTES,
    StreamLimits,
    run_qwen_worker,
)


def _sse_response(*events, status=200):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream"},
        content=body.encode("utf-8"),
    )


class _ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, delay=0.0):
        self.chunks = list(chunks)
        self.delay = delay
        self.yielded = 0

    async def __aiter__(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.yielded += 1
            yield chunk


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

    def test_grep_does_not_authorize_full_write(self):
        agent = SandboxAgent(self.root, ["example.txt"])

        result = agent.execute(
            {"action": "grep", "query": "hello", "paths": ["example.txt"]}
        )

        self.assertEqual(result["matches"], ["example.txt:1:hello world"])
        self.assertNotIn("example.txt", agent.fully_observed_files)
        with self.assertRaisesRegex(AgentError, "read existing file"):
            agent.execute(
                {
                    "action": "write",
                    "path": "example.txt",
                    "content": "rewritten\n",
                }
            )

    def test_complete_explicit_read_authorizes_full_write(self):
        agent = SandboxAgent(self.root, ["example.txt"])

        agent.execute({"action": "read", "path": "example.txt"})
        result = agent.execute(
            {
                "action": "write",
                "path": "example.txt",
                "content": "rewritten\n",
            }
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.source.read_text(encoding="utf-8"), "rewritten\n")

    def test_truncated_explicit_read_does_not_authorize_full_write(self):
        self.source.write_text("x" * (qwen_agent.MAX_READ_CHARS + 1), encoding="utf-8")
        agent = SandboxAgent(self.root, ["example.txt"])

        result = agent.execute({"action": "read", "path": "example.txt"})

        self.assertIn("[truncated]", result["content"])
        self.assertNotIn("example.txt", agent.fully_observed_files)
        with self.assertRaisesRegex(AgentError, "read existing file"):
            agent.execute(
                {
                    "action": "write",
                    "path": "example.txt",
                    "content": "rewritten\n",
                }
            )

    def test_stale_explicit_read_rejects_full_write(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        agent.execute({"action": "read", "path": "example.txt"})
        self.source.write_text("changed externally\n", encoding="utf-8")

        with self.assertRaisesRegex(AgentError, "file changed since it was read"):
            agent.execute(
                {
                    "action": "write",
                    "path": "example.txt",
                    "content": "rewritten\n",
                }
            )
        self.assertEqual(
            self.source.read_text(encoding="utf-8"), "changed externally\n"
        )

    def test_successful_mutation_invalidates_full_observation(self):
        other = self.root / "other.txt"
        other.write_text("other\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["example.txt", "other.txt"])
        agent.execute({"action": "read", "path": "example.txt"})
        agent.execute({"action": "read", "path": "other.txt"})
        agent.execute(
            {
                "action": "replace",
                "path": "example.txt",
                "old": "hello world",
                "new": "hello qwen",
            }
        )

        self.assertEqual(agent.fully_observed_files, {})
        with self.assertRaisesRegex(AgentError, "read existing file"):
            agent.execute(
                {
                    "action": "write",
                    "path": "other.txt",
                    "content": "rewritten again\n",
                }
            )

    def test_full_observation_uses_canonical_symlink_identity(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        real = allowed / "real.txt"
        real.write_text("allowed\n", encoding="utf-8")
        (allowed / "link.txt").symlink_to("real.txt")
        agent = SandboxAgent(self.root, ["allowed"])

        agent.execute({"action": "read", "path": "allowed/link.txt"})

        self.assertIn("allowed/real.txt", agent.fully_observed_files)
        self.assertNotIn("allowed/link.txt", agent.fully_observed_files)
        agent.execute(
            {"action": "write", "path": "allowed/real.txt", "content": "new\n"}
        )
        self.assertEqual(real.read_text(encoding="utf-8"), "new\n")

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

    def test_reject_intra_sandbox_symlink_read_bypass(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        (self.root / "private.txt").write_text("private", encoding="utf-8")
        (allowed / "link.txt").symlink_to(Path("..") / "private.txt")

        agent = SandboxAgent(self.root, ["allowed"])
        with self.assertRaisesRegex(AgentError, "allowlist"):
            agent.execute({"action": "read", "path": "allowed/link.txt"})

    def test_reject_forbidden_env_symlink_alias(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        (self.root / ".env").write_text("SECRET=value\n", encoding="utf-8")
        (allowed / "config.txt").symlink_to(Path("..") / ".env")

        agent = SandboxAgent(self.root, ["allowed"])
        with self.assertRaisesRegex(AgentError, "forbidden"):
            agent.execute({"action": "read", "path": "allowed/config.txt"})

    def test_reject_replace_through_intra_sandbox_symlink(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        private = self.root / "private.txt"
        private.write_bytes(b"private\n")
        (allowed / "link.txt").symlink_to(Path("..") / "private.txt")

        agent = SandboxAgent(self.root, ["allowed"])
        with self.assertRaisesRegex(AgentError, "allowlist"):
            agent.execute(
                {
                    "action": "replace",
                    "path": "allowed/link.txt",
                    "old": "private",
                    "new": "changed",
                }
            )
        self.assertEqual(private.read_bytes(), b"private\n")

    def test_reject_write_through_intra_sandbox_symlink(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        private = self.root / "private.txt"
        private.write_bytes(b"private\n")
        (allowed / "link.txt").symlink_to(Path("..") / "private.txt")

        agent = SandboxAgent(self.root, ["allowed"])
        with self.assertRaisesRegex(AgentError, "allowlist"):
            agent.execute(
                {
                    "action": "write",
                    "path": "allowed/link.txt",
                    "content": "changed\n",
                }
            )
        self.assertEqual(private.read_bytes(), b"private\n")

    def test_allow_in_scope_symlink_to_canonical_target(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        (allowed / "real.txt").write_text("allowed\n", encoding="utf-8")
        (allowed / "link.txt").symlink_to("real.txt")

        agent = SandboxAgent(self.root, ["allowed/real.txt"])
        result = agent.execute({"action": "read", "path": "allowed/link.txt"})

        self.assertEqual(result["content"], "allowed\n")

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

    @unittest.skipUnless(
        qwen_agent._command_sandbox_available(),
        "usable macOS command sandbox is unavailable",
    )
    def test_command_allowlist_accepts_approved_command(self):
        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            allowed_commands=["git diff"],
        )
        result = agent.execute({"action": "run", "command": "git diff"})
        self.assertEqual(result["command"], "git diff")

    def test_command_allowlist_rejects_unlisted_command(self):
        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            allowed_commands=["git diff"],
        )
        with self.assertRaisesRegex(AgentError, "not allowlisted"):
            agent.execute({"action": "run", "command": "git status"})

    def test_command_sandbox_unavailable_fails_closed(self):
        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            allowed_commands=["git diff"],
        )
        with mock.patch(
            "qwen_agent._discover_command_sandbox",
            side_effect=AgentError("command sandbox is unavailable"),
        ), mock.patch("qwen_agent.subprocess.Popen") as popen:
            with self.assertRaisesRegex(AgentError, "sandbox is unavailable"):
                agent.execute({"action": "run", "command": "git diff"})
        popen.assert_not_called()

    def test_command_resolution_ignores_path_shadowing(self):
        shadow = self.root / "git"
        shadow.write_text("#!/bin/sh\necho shadowed\n", encoding="utf-8")
        shadow.chmod(0o755)
        captured: list[tuple[str, ...]] = []
        captured_environment = {}

        class FakeSandbox:
            def run(self, argv, **kwargs):
                captured.append(argv)
                captured_environment.update(kwargs["environment"])
                return qwen_agent.CommandResult(0, "trusted\n", "")

        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            allowed_commands=["git --version"],
        )
        with mock.patch.dict(
            os.environ,
            {"PATH": str(self.root), "TOP_SECRET": "not inherited"},
        ), mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": "git --version"})

        self.assertEqual(result["stdout"], "trusted\n")
        self.assertEqual(Path(captured[0][0]).name, "git")
        self.assertNotEqual(Path(captured[0][0]), shadow)
        self.assertEqual(captured_environment["HOME"], str(self.root.resolve()))
        self.assertNotIn("TOP_SECRET", captured_environment)
        self.assertNotIn(str(self.root), captured_environment["PATH"])

    def test_command_sandbox_rejects_missing_executable(self):
        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            allowed_commands=["command-that-does-not-exist"],
        )
        with self.assertRaisesRegex(AgentError, "executable was not found"):
            agent.execute(
                {"action": "run", "command": "command-that-does-not-exist"}
            )

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

    def test_grep_does_not_expose_out_of_scope_symlink_target(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        (self.root / "private.txt").write_text("private needle\n", encoding="utf-8")
        (allowed / "link.txt").symlink_to(Path("..") / "private.txt")

        agent = SandboxAgent(self.root, ["allowed"])
        result = agent.execute(
            {"action": "grep", "query": "private", "paths": ["allowed"]}
        )

        self.assertEqual(result["matches"], [])

    def test_grep_finds_match_after_historical_one_hundred_file_boundary(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        target = "src/file-120.txt"
        for index in range(150):
            path = source_directory / f"file-{index:03d}.txt"
            path.write_text(
                "historical needle\n" if index == 120 else "no match\n",
                encoding="utf-8",
            )
        agent = SandboxAgent(self.root, ["src"])

        result = agent.execute(
            {"action": "grep", "query": "historical", "paths": ["src"]}
        )

        self.assertEqual(result["matches"], [f"{target}:1:historical needle"])
        self.assertEqual(result["files_scanned"], 150)
        self.assertEqual(result["files_skipped"], 0)
        self.assertFalse(result["truncated"])

    def test_grep_file_limit_is_truthful(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        for index in range(8):
            (source_directory / f"file-{index:03d}.txt").write_text(
                "late needle\n" if index == 7 else "no match\n", encoding="utf-8"
            )
        agent = SandboxAgent(self.root, ["src"])

        with mock.patch.object(qwen_agent, "MAX_GREP_FILES", 5):
            result = agent.execute(
                {"action": "grep", "query": "late", "paths": ["src"]}
            )

        self.assertEqual(result["matches"], [])
        self.assertEqual(result["files_scanned"], 5)
        self.assertGreater(result["files_skipped"], 0)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reason"], "file_limit")
        self.assertEqual(result["skipped_reasons"]["file_limit"], 1)

    def test_grep_match_limit_is_truthful(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        for index in range(4):
            (source_directory / f"file-{index}.txt").write_text(
                "needle\n", encoding="utf-8"
            )
        agent = SandboxAgent(self.root, ["src"])

        with mock.patch.object(qwen_agent, "MAX_GREP_MATCHES", 2):
            result = agent.execute(
                {"action": "grep", "query": "needle", "paths": ["src"]}
            )

        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(result["files_scanned"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reason"], "match_limit")

    def test_grep_response_limits_are_truthful(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        (source_directory / "long.txt").write_text(
            "needle " + ("x" * 100) + "\n", encoding="utf-8"
        )
        agent = SandboxAgent(self.root, ["src"])

        with mock.patch.object(qwen_agent, "MAX_GREP_RESPONSE_CHARS", 40), mock.patch.object(
            qwen_agent, "MAX_GREP_RESPONSE_BYTES", 1_000
        ):
            result = agent.execute(
                {"action": "grep", "query": "needle", "paths": ["src"]}
            )

        self.assertEqual(result["matches"], [])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reason"], "response_limit")

        with mock.patch.object(qwen_agent, "MAX_GREP_RESPONSE_CHARS", 1_000), mock.patch.object(
            qwen_agent, "MAX_GREP_RESPONSE_BYTES", 40
        ):
            byte_limited = agent.execute(
                {"action": "grep", "query": "needle", "paths": ["src"]}
            )
        self.assertEqual(byte_limited["matches"], [])
        self.assertTrue(byte_limited["truncated"])
        self.assertEqual(byte_limited["truncation_reason"], "response_limit")

    def test_grep_traversal_order_is_deterministic(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        for name in ("z.txt", "a.txt", "m.txt"):
            (source_directory / name).write_text("needle\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["src"])
        action = {"action": "grep", "query": "needle", "paths": ["src"]}

        first = agent.execute(action)
        second = agent.execute(action)

        self.assertEqual(first, second)
        self.assertEqual(
            [match.split(":", 1)[0] for match in first["matches"]],
            ["src/a.txt", "src/m.txt", "src/z.txt"],
        )

    def test_grep_skips_binary_and_invalid_utf8_files(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        (source_directory / "binary.bin").write_bytes(b"needle\x00\n")
        (source_directory / "invalid.txt").write_bytes(b"\xffneedle\n")
        (source_directory / "valid.txt").write_text("needle\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["src"])

        result = agent.execute(
            {"action": "grep", "query": "needle", "paths": ["src"]}
        )

        self.assertEqual(result["matches"], ["src/valid.txt:1:needle"])
        self.assertEqual(result["files_scanned"], 1)
        self.assertEqual(result["files_skipped"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reason"], "file_skip")
        self.assertEqual(result["skipped_reasons"]["binary_or_non_utf8"], 2)

    def test_grep_skips_oversized_file_and_reports_incomplete_scan(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        (source_directory / "large.txt").write_bytes(
            b"needle\n" + b"x" * qwen_agent.MAX_GREP_FILE_BYTES
        )
        (source_directory / "valid.txt").write_text("needle\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["src"])

        result = agent.execute(
            {"action": "grep", "query": "needle", "paths": ["src"]}
        )

        self.assertEqual(result["matches"], ["src/valid.txt:1:needle"])
        self.assertEqual(result["files_scanned"], 1)
        self.assertEqual(result["files_skipped"], 1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reason"], "file_skip")
        self.assertEqual(result["skipped_reasons"]["file_too_large"], 1)

    def test_full_observation_survives_conversation_compaction(self):
        agent = SandboxAgent(self.root, ["example.txt"])
        agent.execute({"action": "read", "path": "example.txt"})
        conversation = AgentConversation(
            "rewrite the file",
            ["example.txt"],
            [],
            [],
            None,
        )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 2):
            conversation.record(
                json.dumps({"action": "read", "path": "example.txt"}),
                {"action": "read", "path": "example.txt"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "example.txt",
                    "content": f"context-{index}",
                },
                files_read=agent.fully_observed_files,
                files_changed=(),
                actions_completed=index + 1,
                protocol_errors=0,
            )
        _messages, stats = conversation.snapshot()

        self.assertTrue(stats["history_compacted"])
        agent.execute(
            {"action": "write", "path": "example.txt", "content": "after\n"}
        )
        self.assertEqual(self.source.read_text(encoding="utf-8"), "after\n")


@unittest.skipUnless(
    qwen_agent._command_sandbox_available(),
    "usable macOS command sandbox is unavailable",
)
class MacOSCommandSandboxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "input.txt"
        self.input_path.write_text("inside sentinel\n", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _python_command(self, code: str) -> str:
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        bootstrap = f"import base64\nexec(base64.b64decode({encoded!r}))"
        return f"{shlex.quote(sys.executable)} -c {shlex.quote(bootstrap)}"

    def _run_python(self, code: str, *, allow=None):
        command = self._python_command(code)
        agent = SandboxAgent(
            self.root,
            allow or ["input.txt"],
            allowed_commands=[command],
        )
        return agent.execute({"action": "run", "command": command})

    def test_outside_file_read_is_denied_without_secret_output(self):
        outside = self.root.parent / f"{self.root.name}-outside-secret.txt"
        outside.write_text("TOP_SECRET_SENTINEL\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)

        result = self._run_python(
            f"from pathlib import Path; print(Path({str(outside)!r}).read_text())"
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("TOP_SECRET_SENTINEL", result["stdout"])
        self.assertNotIn("TOP_SECRET_SENTINEL", result["stderr"])

    def test_outside_file_write_is_denied(self):
        outside = self.root.parent / f"{self.root.name}-outside-write.txt"
        self.addCleanup(outside.unlink, missing_ok=True)

        result = self._run_python(
            f"from pathlib import Path; Path({str(outside)!r}).write_text('outside')"
        )

        self.assertFalse(result["ok"])
        self.assertFalse(outside.exists())

    def test_outside_symlink_read_is_denied_without_secret_output(self):
        outside = self.root.parent / f"{self.root.name}-outside-symlink.txt"
        outside.write_text("TOP_SECRET_SYMLINK_SENTINEL\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        (self.root / "link.txt").symlink_to(outside)

        result = self._run_python(
            "from pathlib import Path; print(Path('link.txt').read_text())"
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("TOP_SECRET_SYMLINK_SENTINEL", result["stdout"])
        self.assertNotIn("TOP_SECRET_SYMLINK_SENTINEL", result["stderr"])

    def test_sandbox_file_read_and_write_work(self):
        result = self._run_python(
            "from pathlib import Path; "
            "data=Path('input.txt').read_text(); "
            "Path('output.txt').write_text(data+'output'); print(data, end='')"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "inside sentinel\n")
        self.assertEqual(
            (self.root / "output.txt").read_text(encoding="utf-8"),
            "inside sentinel\noutput",
        )

    def test_network_connection_is_denied(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        result = self._run_python(
            "import socket; "
            f"socket.create_connection(('127.0.0.1', {port}), timeout=2); "
            "print('CONNECTED')"
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("CONNECTED", result["stdout"])
        listener.settimeout(0.2)
        with self.assertRaises(socket.timeout):
            listener.accept()

    def test_command_output_is_bounded(self):
        result = self._run_python(
            "import sys; print('o'*100000); print('e'*100000, file=sys.stderr)"
        )

        self.assertTrue(result["ok"])
        self.assertLessEqual(len(result["stdout"].encode()), qwen_agent.MAX_COMMAND_OUTPUT_CHARS)
        self.assertLessEqual(len(result["stderr"].encode()), qwen_agent.MAX_COMMAND_OUTPUT_CHARS)

    def test_timeout_kills_child_process_group(self):
        late_file = self.root / "late-child.txt"
        child_code = (
            "import pathlib,time; time.sleep(0.8); "
            "pathlib.Path('late-child.txt').write_text('alive')"
        )
        code = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
            "time.sleep(60)"
        )

        with mock.patch.object(qwen_agent, "COMMAND_TIMEOUT_SECONDS", 0.2):
            with self.assertRaisesRegex(AgentError, "command timed out"):
                self._run_python(code)
        time.sleep(1.0)
        self.assertFalse(late_file.exists())


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


class AgentConversationTests(unittest.TestCase):
    def make_conversation(self, validation_failure=None):
        return AgentConversation(
            "implement the requested change",
            ["example.txt"],
            ["tests"],
            ["pytest tests/test_example.py"],
            validation_failure,
        )

    def record(
        self,
        conversation,
        action,
        result,
        *,
        files_read=(),
        files_changed=(),
        actions_completed=1,
        protocol_errors=0,
    ):
        conversation.record(
            json.dumps(action, ensure_ascii=False, separators=(",", ":")),
            action,
            result,
            files_read=files_read,
            files_changed=files_changed,
            actions_completed=actions_completed,
            protocol_errors=protocol_errors,
        )

    def test_short_read_replace_flow_keeps_latest_result_lossless(self):
        conversation = self.make_conversation()
        read_action = {"action": "read", "path": "example.txt"}
        read_result = {
            "ok": True,
            "action": "read",
            "path": "example.txt",
            "content": "CURRENT_READ_SENTINEL\n",
        }
        self.record(
            conversation,
            read_action,
            read_result,
            files_read=("example.txt",),
        )
        messages, _stats = conversation.snapshot()
        self.assertEqual(
            json.loads(messages[-1]["content"]), read_result
        )
        self.assertIn("CURRENT_READ_SENTINEL", messages[-1]["content"])

        replace_action = {
            "action": "replace",
            "path": "example.txt",
            "old": "hello",
            "new": "goodbye",
        }
        replace_result = {
            "ok": True,
            "action": "replace",
            "path": "example.txt",
            "matches": 1,
            "replaced": 1,
        }
        self.record(
            conversation,
            replace_action,
            replace_result,
            files_read=("example.txt",),
            files_changed=("example.txt",),
            actions_completed=2,
        )
        messages, _stats = conversation.snapshot()
        self.assertIn("CURRENT_READ_SENTINEL", "\n".join(item["content"] for item in messages))
        self.assertEqual(json.loads(messages[-1]["content"]), replace_result)

    def test_history_is_bounded_after_one_hundred_interactions(self):
        conversation = self.make_conversation()
        sizes = []
        for index in range(100):
            action = {"action": "read", "path": "example.txt"}
            result = {
                "ok": True,
                "action": "read",
                "path": "example.txt",
                "content": f"content-{index}-" + ("x" * 2_000),
            }
            self.record(
                conversation,
                action,
                result,
                files_read=("example.txt",),
                actions_completed=index + 1,
            )
            _messages, stats = conversation.snapshot()
            sizes.append(stats["history_chars"])

        self.assertLessEqual(max(sizes), MAX_AGENT_HISTORY_CHARS)
        self.assertEqual(stats["history_interactions"], MAX_AGENT_RECENT_INTERACTIONS)
        self.assertTrue(stats["history_compacted"])
        self.assertGreater(stats["history_dropped_interactions"], 0)
        self.assertLessEqual(sizes[-1], MAX_AGENT_HISTORY_CHARS)

    def test_old_read_content_is_replaced_by_file_metadata(self):
        conversation = self.make_conversation()
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 1):
            self.record(
                conversation,
                {"action": "read", "path": "example.txt"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "example.txt",
                    "content": (
                        "VERY_OLD_SOURCE_SENTINEL"
                        if index == 0
                        else f"current-{index}"
                    ),
                },
                files_read=("example.txt",),
                actions_completed=index + 1,
            )
        messages, _stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertNotIn("VERY_OLD_SOURCE_SENTINEL", rendered)
        self.assertIn("fully_observed_files", rendered)
        self.assertIn("example.txt", rendered)

    def test_old_command_output_is_replaced_by_status_metadata(self):
        conversation = self.make_conversation()
        self.record(
            conversation,
            {"action": "run", "command": "pytest tests/test_example.py"},
            {
                "ok": False,
                "action": "run",
                "command": "pytest tests/test_example.py",
                "returncode": 1,
                "stdout": "OLD_COMMAND_OUTPUT_SENTINEL",
                "stderr": "failure",
            },
        )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS):
            self.record(
                conversation,
                {"action": "read", "path": "example.txt"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "example.txt",
                    "content": f"current-{index}",
                },
                files_read=("example.txt",),
                actions_completed=index + 2,
            )
        messages, _stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertNotIn("OLD_COMMAND_OUTPUT_SENTINEL", rendered)
        self.assertIn("last_command=run", rendered)
        self.assertIn("returncode=1", rendered)

    def test_old_grep_matches_are_replaced_by_metadata(self):
        conversation = self.make_conversation()
        self.record(
            conversation,
            {"action": "grep", "query": "sentinel", "paths": ["src"]},
            {
                "ok": True,
                "action": "grep",
                "matches": ["src/example.txt:1:OLD_GREP_MATCH_SENTINEL"],
                "files_scanned": 4,
                "files_skipped": 1,
                "truncated": True,
                "truncation_reason": "file_skip",
            },
        )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS):
            self.record(
                conversation,
                {"action": "read", "path": "example.txt"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "example.txt",
                    "content": f"current-{index}",
                },
                files_read=("example.txt",),
                actions_completed=index + 2,
            )
        messages, _stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertNotIn("OLD_GREP_MATCH_SENTINEL", rendered)
        self.assertIn("grep", rendered)
        self.assertIn("matches=1", rendered)
        self.assertIn("files_scanned=4", rendered)
        self.assertIn("files_skipped=1", rendered)
        self.assertIn("truncation_reason=file_skip", rendered)

    def test_failed_action_feedback_is_lossless_on_next_turn(self):
        conversation = self.make_conversation()
        action = {
            "action": "replace",
            "path": "example.txt",
            "old": "missing",
            "new": "value",
        }
        result = {
            "ok": False,
            "error": "replace text was not found: example.txt",
        }
        self.record(conversation, action, result, protocol_errors=1)
        messages, _stats = conversation.snapshot()
        self.assertEqual(json.loads(messages[-1]["content"]), result)
        self.assertIn("replace text was not found", messages[-1]["content"])

    def test_changed_files_and_validation_failure_survive_compaction(self):
        validation_failure = "VALIDATION_FAILURE_SENTINEL: fix the test output"
        conversation = self.make_conversation(validation_failure)
        self.record(
            conversation,
            {"action": "replace", "path": "example.txt"},
            {"ok": True, "action": "replace", "path": "example.txt", "replaced": 1},
            files_read=("example.txt",),
            files_changed=("example.txt",),
        )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 2):
            self.record(
                conversation,
                {"action": "read", "path": "example.txt"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "example.txt",
                    "content": str(index),
                },
                files_read=("example.txt",),
                files_changed=("example.txt",),
                actions_completed=index + 2,
            )
        messages, _stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertIn("VALIDATION_FAILURE_SENTINEL", rendered)
        self.assertIn("files_changed", rendered)
        self.assertIn("example.txt", rendered)

    def test_compaction_is_deterministic(self):
        def build():
            conversation = self.make_conversation("same validation failure")
            for index in range(12):
                self.record(
                    conversation,
                    {"action": "read", "path": "example.txt"},
                    {
                        "ok": True,
                        "action": "read",
                        "path": "example.txt",
                        "content": f"value-{index}",
                    },
                    files_read=("example.txt",),
                    actions_completed=index + 1,
                )
            return conversation.snapshot()

        first_messages, first_stats = build()
        second_messages, second_stats = build()
        self.assertEqual(first_messages, second_messages)
        self.assertEqual(first_stats, second_stats)

    def test_oversized_last_result_remains_valid_json_and_bounded(self):
        conversation = self.make_conversation()
        self.record(
            conversation,
            {"action": "read", "path": "example.txt"},
            {
                "ok": True,
                "action": "read",
                "path": "example.txt",
                "content": "Türkçe🙂" * 20_000,
            },
            files_read=("example.txt",),
        )
        messages, stats = conversation.snapshot()
        result = json.loads(messages[-1]["content"])
        self.assertTrue(result["context_truncated"])
        self.assertLessEqual(stats["history_chars"], MAX_AGENT_HISTORY_CHARS)


class AgentValidationConversationTests(unittest.TestCase):
    def test_validation_state_requires_current_generation(self):
        state = qwen_agent.ValidationState()
        self.assertEqual(state.phase, ValidationPhase.EDITING)
        state.mark_model_done()
        self.assertEqual(state.phase, ValidationPhase.MODEL_DONE)
        result = ValidationResult("git diff", 0, "", "")
        self.assertTrue(state.finish_round((result,), 0))
        self.assertEqual(state.phase, ValidationPhase.VALIDATED)
        self.assertEqual(state.validated_generation, 0)
        state.mark_mutation(1)
        self.assertEqual(state.phase, ValidationPhase.EDITING)
        self.assertIsNone(state.validated_generation)

    def test_old_validation_output_is_compacted_but_metadata_survives(self):
        conversation = AgentConversation(
            "repair the task",
            ["example.txt"],
            (),
            ["git diff"],
            None,
        )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 2):
            result = {
                "ok": False,
                "action": "validation",
                "command": "git diff",
                "returncode": 1,
                "stdout": (
                    "OLD_VALIDATION_OUTPUT_SENTINEL"
                    if index == 0
                    else f"failure-{index}"
                ),
                "stderr": "stderr",
                "validation_round": index + 1,
            }
            conversation.record(
                json.dumps({"action": "done", "summary": "finished"}),
                {"action": "validation"},
                result,
                files_read=(),
                files_changed=(),
                actions_completed=index + 1,
                protocol_errors=0,
            )

        messages, _stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertNotIn("OLD_VALIDATION_OUTPUT_SENTINEL", rendered)
        self.assertIn("validation", rendered)
        self.assertIn("returncode=1", rendered)


class AgentValidationFlowTests(unittest.IsolatedAsyncioTestCase):
    validation_command = "git diff"

    def setUp(self):
        self.endpoint_environment = mock.patch.dict(
            "os.environ",
            {"QWEN_BASE_URL": "http://127.0.0.1:11234/v1"},
        )
        self.endpoint_environment.start()

    def tearDown(self):
        self.endpoint_environment.stop()

    class FakeCommandSandbox:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)
            self.calls = []

        def run(self, argv, **kwargs):
            self.calls.append((argv, kwargs))
            if self.outcomes:
                outcome = self.outcomes.pop(0)
            else:
                outcome = (1, "", "default validation failure")
            return qwen_agent.CommandResult(*outcome)

    @staticmethod
    def generation(action):
        return AgentGeneration(
            json.dumps(action, ensure_ascii=False, separators=(",", ":")),
            1,
            0,
            1,
        )

    async def invoke(
        self,
        actions,
        outcomes,
        *,
        allowed_commands=None,
        validation_commands=None,
        max_actions=10,
        capture_messages=None,
    ):
        generations = list(actions)
        fake_sandbox = self.FakeCommandSandbox(outcomes)
        self.last_fake_sandbox = fake_sandbox

        async def fake_turn(_client, _base, _model, messages):
            if capture_messages is not None:
                capture_messages.append(messages)
            action = generations.pop(0) if generations else {
                "action": "done",
                "summary": "finished",
            }
            return self.generation(action)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.txt").write_text(
                "bad\n", encoding="utf-8"
            )
            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ), mock.patch(
                "qwen_agent._discover_command_sandbox",
                return_value=fake_sandbox,
            ):
                result = await run_qwen_agent(
                    "repair example.txt",
                    directory,
                    ["example.txt"],
                    allowed_commands=allowed_commands,
                    validation_commands=validation_commands,
                    max_actions=max_actions,
                )
        return result, fake_sandbox

    async def test_done_without_validation_is_explicitly_unvalidated(self):
        result, fake_sandbox = await self.invoke(
            [{"action": "done", "summary": "finished"}],
            [],
        )

        self.assertEqual(result.status, "done")
        self.assertFalse(result.validated)
        self.assertEqual(result.validation_commands, 0)
        self.assertEqual(result.validation_rounds, 0)
        self.assertEqual(fake_sandbox.calls, [])

    async def test_done_with_required_validation_passes_as_validated(self):
        result, fake_sandbox = await self.invoke(
            [{"action": "done", "summary": "finished"}],
            [(0, "", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
        )

        self.assertEqual(result.status, "validated")
        self.assertTrue(result.validated)
        self.assertEqual(result.validation_commands, 1)
        self.assertEqual(result.validation_rounds, 1)
        self.assertEqual(len(fake_sandbox.calls), 1)

    async def test_validation_failure_is_feedback_for_the_next_model_turn(self):
        messages = []
        with self.assertRaisesRegex(RuntimeError, "AGENT_ACTION_LIMIT"):
            await self.invoke(
                [
                    {"action": "done", "summary": "finished"},
                    {"action": "done", "summary": "finished again"},
                ],
                [(1, "PRIVATE_VALIDATION_OUTPUT", "validation failed")],
                allowed_commands=[self.validation_command],
                validation_commands=[self.validation_command],
                max_actions=2,
                capture_messages=messages,
            )

        feedback = json.loads(messages[1][-1]["content"])
        self.assertFalse(feedback["ok"])
        self.assertEqual(feedback["action"], "validation")
        self.assertEqual(feedback["returncode"], 1)
        self.assertIn("PRIVATE_VALIDATION_OUTPUT", feedback["stdout"])

    async def test_failure_fix_pass_completes_as_validated(self):
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "intermediate",
                },
                {"action": "done", "summary": "first attempt"},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "intermediate",
                    "new": "good",
                },
                {"action": "done", "summary": "fixed"},
            ],
            [(1, "first failure", ""), (0, "", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            max_actions=4,
        )

        self.assertEqual(result.status, "validated")
        self.assertTrue(result.validated)
        self.assertEqual(result.validation_rounds, 2)
        self.assertEqual(len(fake_sandbox.calls), 2)

    async def test_model_run_success_is_not_reused_after_mutation(self):
        result, fake_sandbox = await self.invoke(
            [
                {"action": "run", "command": self.validation_command},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "good",
                },
                {"action": "done", "summary": "finished"},
            ],
            [(0, "model pass", ""), (0, "host pass", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            max_actions=3,
        )

        self.assertTrue(result.validated)
        self.assertEqual(len(fake_sandbox.calls), 2)
        self.assertEqual(fake_sandbox.calls[0][0][1], "diff")
        self.assertEqual(fake_sandbox.calls[1][0][1], "diff")

    async def test_host_validation_runs_without_model_run_action(self):
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "good",
                },
                {"action": "done", "summary": "finished"},
            ],
            [(0, "", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
        )

        self.assertTrue(result.validated)
        self.assertEqual(len(fake_sandbox.calls), 1)

    async def test_required_validation_runs_after_model_selected_easy_test(self):
        result, fake_sandbox = await self.invoke(
            [
                {"action": "run", "command": "git status"},
                {"action": "done", "summary": "finished"},
            ],
            [(0, "easy pass", ""), (0, "required pass", "")],
            allowed_commands=["git status", self.validation_command],
            validation_commands=[self.validation_command],
        )

        self.assertTrue(result.validated)
        self.assertEqual(len(fake_sandbox.calls), 2)
        self.assertEqual(fake_sandbox.calls[0][0][1], "status")
        self.assertEqual(fake_sandbox.calls[1][0][1], "diff")

    async def test_multiple_validation_commands_stop_at_second_failure(self):
        with self.assertRaisesRegex(RuntimeError, "AGENT_ACTION_LIMIT"):
            await self.invoke(
                [{"action": "done", "summary": "finished"}],
                [(0, "first pass", ""), (1, "second output", "")],
                allowed_commands=["git diff", "git status"],
                validation_commands=["git diff", "git status"],
                max_actions=1,
            )

        self.assertEqual(len(self.last_fake_sandbox.calls), 2)

    async def test_forbidden_validation_command_is_rejected_before_model_call(self):
        async def fail_model(*_args, **_kwargs):
            raise AssertionError("model must not be called")

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "qwen_agent._preflight",
                side_effect=AssertionError("preflight must not be called"),
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fail_model
            ):
                with self.assertRaisesRegex(AgentError, "network command"):
                    await run_qwen_agent(
                        "task",
                        directory,
                        ["example.txt"],
                        allowed_commands=["git diff"],
                        validation_commands=["curl https://example.com"],
                    )

    async def test_validation_command_must_be_in_exact_allowed_allowlist(self):
        async def fail_model(*_args, **_kwargs):
            raise AssertionError("model must not be called")

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "qwen_agent._generate_agent_turn", new=fail_model
            ):
                with self.assertRaisesRegex(AgentError, "not in the allowed"):
                    await run_qwen_agent(
                        "task",
                        directory,
                        ["example.txt"],
                        allowed_commands=["git diff"],
                        validation_commands=["git status"],
                    )

    async def test_validation_round_limit_is_bounded(self):
        with self.assertRaisesRegex(RuntimeError, "AGENT_VALIDATION_LIMIT"):
            await self.invoke(
                [{"action": "done", "summary": "finished"}],
                [(1, "failure", "")],
                allowed_commands=[self.validation_command],
                validation_commands=[self.validation_command],
                max_actions=100,
            )

        self.assertEqual(
            len(self.last_fake_sandbox.calls), MAX_VALIDATION_ROUNDS
        )

    async def test_validation_telemetry_contains_metadata_but_not_output(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result, _fake_sandbox = await self.invoke(
                [{"action": "done", "summary": "finished"}],
                [(0, "PRIVATE_VALIDATION_OUTPUT", "PRIVATE_VALIDATION_ERROR")],
                allowed_commands=[self.validation_command],
                validation_commands=[self.validation_command],
            )

        self.assertTrue(result.validated)
        diagnostic = stderr.getvalue()
        self.assertIn("stage=VALIDATION_COMMAND", diagnostic)
        self.assertIn("returncode=0", diagnostic)
        self.assertNotIn("PRIVATE_VALIDATION_OUTPUT", diagnostic)
        self.assertNotIn("PRIVATE_VALIDATION_ERROR", diagnostic)


class AgentValidationCliContractTests(unittest.TestCase):
    def request(self, **updates):
        request = {
            "task": "verify the task",
            "sandbox": "/tmp/sandbox",
            "allow": ["example.txt"],
            "allowed_commands": ["git diff"],
            "validation_commands": ["git diff"],
        }
        request.update(updates)
        return request

    def test_json_request_parses_validation_commands_strictly(self):
        request = qwen_agent_cli._parse_json_request(
            json.dumps(self.request()).encode("utf-8")
        )
        self.assertEqual(request["validation_commands"], ["git diff"])

        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "array of strings"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(validation_commands="git diff")).encode(
                    "utf-8"
                )
            )
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "empty strings"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(validation_commands=[""])).encode("utf-8")
            )
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "unknown request field"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(unknown=True)).encode("utf-8")
            )

    def test_flag_mode_accepts_repeated_validation_command_flags(self):
        request = qwen_agent_cli._request_from_flags(
            [
                "--sandbox",
                "/tmp/sandbox",
                "--allow",
                "example.txt",
                "--allow-command",
                "git diff",
                "--validation-command",
                "git diff",
                "--validation-command",
                "git status",
                "--task",
                "verify",
            ]
        )
        self.assertEqual(
            request["validation_commands"], ["git diff", "git status"]
        )

        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "empty strings"):
            qwen_agent_cli._request_from_flags(
                [
                    "--sandbox",
                    "/tmp/sandbox",
                    "--allow",
                    "example.txt",
                    "--validation-command",
                    "",
                    "--task",
                    "verify",
                ]
            )

    def test_cli_json_contract_exposes_validation_metadata(self):
        result = qwen_agent.AgentResult(
            status="validated",
            actions=2,
            files_changed=["example.txt"],
            model_calls=2,
            completion_tokens=4,
            reasoning_chars=0,
            elapsed_ms=1,
            read_bytes=0,
            write_bytes=4,
            validated=True,
            validation_commands=2,
            validation_failures=0,
            validation_rounds=1,
        )
        output = io.StringIO()
        with mock.patch.object(
            qwen_agent_cli, "run_qwen_agent", return_value=result
        ) as run_agent, contextlib.redirect_stdout(output):
            self.assertEqual(
                qwen_agent_cli.main(
                    [
                        "--sandbox",
                        "/tmp/sandbox",
                        "--allow",
                        "example.txt",
                        "--allow-command",
                        "git diff",
                        "--validation-command",
                        "git diff",
                        "--task",
                        "verify",
                    ]
                ),
                0,
            )
        self.assertEqual(
            run_agent.call_args.kwargs["validation_commands"], ["git diff"]
        )
        document = json.loads(output.getvalue())
        self.assertEqual(document["status"], "validated")
        self.assertTrue(document["validated"])
        self.assertEqual(document["validation_commands"], 2)
        self.assertEqual(document["validation_rounds"], 1)
        self.assertNotIn("PRIVATE_VALIDATION_OUTPUT", output.getvalue())


class AgentTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.endpoint_environment = mock.patch.dict(
            "os.environ",
            {"QWEN_BASE_URL": "http://127.0.0.1:11234/v1"},
        )
        self.endpoint_environment.start()

    def tearDown(self):
        self.endpoint_environment.stop()

    async def test_max_action_case_keeps_history_bounded_and_telemetry_private(self):
        sizes = []
        stderr = io.StringIO()

        async def fake_turn(_client, _base, _model, messages):
            sizes.append(sum(len(item["content"]) for item in messages))
            return AgentGeneration(
                '{"action":"read","path":"example.txt"}', 1, 0, 1
            )

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.txt").write_text(
                "SECRET_SOURCE_SENTINEL\n", encoding="utf-8"
            )
            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(RuntimeError, "AGENT_ACTION_LIMIT"):
                    await run_qwen_agent(
                        "repeat the read action",
                        directory,
                        ["example.txt"],
                        max_actions=100,
                    )

        self.assertEqual(len(sizes), 100)
        self.assertLessEqual(max(sizes), MAX_AGENT_HISTORY_CHARS)
        self.assertLessEqual(sizes[99], MAX_AGENT_HISTORY_CHARS)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", stderr.getvalue())
        self.assertIn("history_chars=", stderr.getvalue())

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

    async def test_agent_done_is_terminal_and_unicode_chunks_are_safe(self):
        event = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "content": json.dumps(
                                {"action": "done", "summary": "café"},
                                ensure_ascii=False,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
            ensure_ascii=False,
        ).encode("utf-8")
        body = b"data: " + event + b"\r\n\r\ndata: [DONE]\r\n\r\n"
        body += b"data: invalid-after-done\r\n\r\n"
        split = body.index("é".encode("utf-8")) + 1
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((body[:split], body[split:])),
        )
        try:
            generation = await _parse_agent_stream(
                response,
                started=asyncio.get_running_loop().time(),
                attempts=1,
                http_status=200,
            )
        finally:
            await response.aclose()
        self.assertIn("café", generation.text)
        self.assertEqual(generation.event_count, 2)

    async def test_agent_output_and_reasoning_are_bounded_during_streaming(self):
        output_response = _sse_response(
            {
                "choices": [
                    {
                        "delta": {"content": "12345"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        with self.assertRaisesRegex(RuntimeError, "GENERATION_OUTPUT_LIMIT"):
            await _parse_agent_stream(
                output_response,
                started=asyncio.get_running_loop().time(),
                attempts=1,
                http_status=200,
                limits=StreamLimits(max_final_chars=4, max_final_bytes=16),
            )

        reasoning_response = _sse_response(
            {
                "choices": [
                    {
                        "delta": {"reasoning_content": "private"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        with self.assertRaisesRegex(RuntimeError, "GENERATION_REASONING_LIMIT"):
            await _parse_agent_stream(
                reasoning_response,
                started=asyncio.get_running_loop().time(),
                attempts=1,
                http_status=200,
                limits=StreamLimits(max_reasoning_chars=3),
            )

        byte_response = _sse_response(
            {
                "choices": [
                    {
                        "delta": {"content": "éé"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        with self.assertRaisesRegex(RuntimeError, "GENERATION_OUTPUT_LIMIT"):
            await _parse_agent_stream(
                byte_response,
                started=asyncio.get_running_loop().time(),
                attempts=1,
                http_status=200,
                limits=StreamLimits(max_final_chars=10, max_final_bytes=3),
            )

    async def test_agent_raw_bytes_and_malformed_utf8_fail_closed(self):
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((b"data: " + b"x" * 100,)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_EVENT_SIZE_LIMIT"):
                await _parse_agent_stream(
                    response,
                    started=asyncio.get_running_loop().time(),
                    attempts=1,
                    http_status=200,
                    limits=StreamLimits(max_event_bytes=16),
                )
        finally:
            await response.aclose()

        body = b": heartbeat\n\n" + b'data: {"choices":[]}' + b"\n\n"
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((body,)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_STREAM_LIMIT"):
                await _parse_agent_stream(
                    response,
                    started=asyncio.get_running_loop().time(),
                    attempts=1,
                    http_status=200,
                    limits=StreamLimits(max_stream_bytes=len(body) - 1),
                )
        finally:
            await response.aclose()

        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((b"data: \xff\n\n",)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "AGENT_GENERATION_MALFORMED"):
                await _parse_agent_stream(
                    response,
                    started=asyncio.get_running_loop().time(),
                    attempts=1,
                    http_status=200,
                )
        finally:
            await response.aclose()

    async def test_agent_heartbeat_event_count_is_bounded(self):
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((b": heartbeat\n\n" * 4,)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_EVENT_LIMIT"):
                await _parse_agent_stream(
                    response,
                    started=asyncio.get_running_loop().time(),
                    attempts=1,
                    http_status=200,
                    limits=StreamLimits(max_events=3),
                )
        finally:
            await response.aclose()

    async def test_agent_non_2xx_body_is_bounded(self):
        calls = 0
        error_stream = _ChunkedStream((b"E" * 512 for _ in range(100)))

        async def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(500, stream=error_stream)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError, "AGENT_GENERATION_HTTP.*status=500"
            ) as raised:
                await _generate_agent_turn(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    [{"role": "user", "content": "task"}],
                )
        self.assertLess(len(str(raised.exception)), MAX_ERROR_BODY_BYTES)
        self.assertLess(error_stream.yielded, 100)
        self.assertEqual(calls, 1)

    async def test_agent_read_timeout_is_not_retried(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("PRIVATE_PROMPT_SENTINEL", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError, "AGENT_GENERATION_READ_TIMEOUT.*attempts=1"
            ):
                await _generate_agent_turn(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    [{"role": "user", "content": "PRIVATE_PROMPT_SENTINEL"}],
                    sleep=lambda _delay: asyncio.sleep(0),
                )

        self.assertEqual(calls, 1)

    async def test_agent_connect_error_is_not_retried(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("ambiguous connection failure", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError, "AGENT_GENERATION_CONNECT_ERROR.*attempts=1"
            ):
                await _generate_agent_turn(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    [{"role": "user", "content": "task"}],
                )

        self.assertEqual(calls, 1)

    async def test_agent_generation_deadline_covers_slow_drip(self):
        async def handler(_request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_ChunkedStream((b": heartbeat\n\n", b": delayed\n\n"), delay=0.05),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with mock.patch("qwen_agent.MAX_GENERATION_SECONDS", 0.01):
                with self.assertRaisesRegex(RuntimeError, "AGENT_GENERATION_DEADLINE"):
                    await _generate_agent_turn(
                        client,
                        "http://qwen.test/v1",
                        DEFAULT_MODEL,
                        [{"role": "user", "content": "task"}],
                    )

    async def test_agent_midstream_failure_is_not_retried(self):
        calls = 0

        class FailingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
                raise httpx.ReadError("stream broke")

        async def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=FailingStream(),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "AGENT_GENERATION_READ_ERROR"):
                await _generate_agent_turn(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    [{"role": "user", "content": "task"}],
                )
        self.assertEqual(calls, 1)

    async def test_agent_client_disables_proxies_and_prompt_omits_absolute_sandbox_path(
        self,
    ):
        client_options = []
        captured_messages = []

        class TestClient(httpx.AsyncClient):
            def __init__(self, *args, **kwargs):
                client_options.append(dict(kwargs))
                super().__init__(*args, **kwargs)

        async def fake_turn(_client, _base, _model, messages):
            captured_messages.append(messages)
            return AgentGeneration(
                '{"action":"done","summary":"finished"}', 2, 0, 1
            )

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "qwen_agent.httpx.AsyncClient", TestClient
            ), mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ):
                result = await run_qwen_agent(
                    "finish task",
                    directory,
                    ["example.txt"],
                    max_actions=1,
                )

        self.assertEqual(result.status, "done")
        self.assertEqual([options["trust_env"] for options in client_options], [False])
        self.assertEqual(
            [options["follow_redirects"] for options in client_options], [False]
        )
        prompt = captured_messages[0][1]["content"]
        self.assertNotIn(str(Path(directory).resolve()), prompt)
        self.assertIn("example.txt", prompt)

    async def test_agent_rejects_remote_endpoint_before_model_client_use(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                "os.environ", {"QWEN_BASE_URL": "https://example.com/v1"}
            ), mock.patch.object(
                qwen_agent.httpx, "AsyncClient", side_effect=AssertionError
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "endpoint_remote_not_allowed"
                ):
                    await run_qwen_agent("task", directory, ["example.txt"])

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

    def test_oversized_task_file_is_rejected_before_model(self):
        result = qwen_agent.AgentResult(
            status="done",
            actions=0,
            files_changed=[],
            model_calls=0,
            completion_tokens=0,
            reasoning_chars=0,
            elapsed_ms=0,
            read_bytes=0,
            write_bytes=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory, "oversized-task.txt")
            task_path.write_bytes(
                b"x" * (qwen_input_limits.MAX_TASK_BYTES + 1)
            )
            with mock.patch.object(
                qwen_agent_cli, "run_qwen_agent", return_value=result
            ) as run_agent, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ) as stderr:
                exit_code = qwen_agent_cli.main(
                    [
                        "--sandbox",
                        directory,
                        "--allow",
                        "example.txt",
                        "--task-file",
                        str(task_path),
                    ]
                )

        self.assertEqual(exit_code, 1)
        self.assertEqual(run_agent.call_count, 0)
        self.assertIn("TASK_TOO_LARGE", stderr.getvalue())


class InputBudgetCliTests(unittest.TestCase):
    @staticmethod
    def agent_request(**updates):
        request = {
            "task": "finish the task",
            "sandbox": "/tmp/sandbox",
            "allow": ["example.txt"],
        }
        request.update(updates)
        return request

    def test_oversized_stdin_reads_only_one_byte_over_limit(self):
        class ShortReadStream:
            def __init__(self, data):
                self.data = data
                self.offset = 0
                self.requested = []

            def read(self, size):
                self.requested.append(size)
                end = min(self.offset + min(size, 97), len(self.data))
                chunk = self.data[self.offset : end]
                self.offset = end
                return chunk

        raw = json.dumps(self.agent_request()).encode("utf-8")
        raw += b" PRIVATE_REQUEST_SENTINEL"
        raw += b" " * (qwen_input_limits.MAX_REQUEST_BYTES + 17 - len(raw))
        stream = ShortReadStream(raw)
        stdin = type("Stdin", (), {"buffer": stream})()
        with mock.patch.object(
            qwen_agent_cli, "run_qwen_agent", side_effect=AssertionError
        ) as run_agent, mock.patch.object(
            qwen_agent_cli.sys, "stdin", stdin
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ) as stderr:
            result = qwen_agent_cli.main([])

        self.assertEqual(result, 1)
        self.assertEqual(run_agent.call_count, 0)
        self.assertEqual(stream.offset, qwen_input_limits.MAX_REQUEST_BYTES + 1)
        self.assertEqual(stream.requested[0], qwen_input_limits.MAX_REQUEST_BYTES + 1)
        self.assertIn("REQUEST_TOO_LARGE", stderr.getvalue())
        self.assertNotIn("PRIVATE_REQUEST_SENTINEL", stderr.getvalue())

    def test_request_json_exact_boundary_and_plus_one(self):
        request = self.agent_request()
        raw = json.dumps(request, separators=(",", ":")).encode("utf-8")
        raw += b" " * (qwen_input_limits.MAX_REQUEST_BYTES - len(raw))
        self.assertEqual(len(raw), qwen_input_limits.MAX_REQUEST_BYTES)
        self.assertEqual(qwen_agent_cli._parse_json_request(raw)["task"], request["task"])
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "REQUEST_TOO_LARGE"):
            qwen_agent_cli._parse_json_request(raw + b" ")

    def test_oversized_request_file_is_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "request.json")
            path.write_bytes(b"x" * (qwen_input_limits.MAX_REQUEST_BYTES + 1))
            with mock.patch.object(
                qwen_agent_cli, "run_qwen_agent", side_effect=AssertionError
            ) as run_agent, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ) as stderr:
                result = qwen_agent_cli.main(["--request-file", str(path)])

        self.assertEqual(result, 1)
        self.assertEqual(run_agent.call_count, 0)
        self.assertIn("REQUEST_TOO_LARGE", stderr.getvalue())

    def test_invalid_utf8_is_rejected_deterministically(self):
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "UTF-8"):
            qwen_agent_cli._parse_json_request(b'{"task":"\xff"}')

    def test_unicode_byte_limit_is_checked_separately_from_character_limit(self):
        with self.assertRaisesRegex(qwen_input_limits.InputLimitError, "PATH_TOO_LONG"):
            qwen_input_limits.validate_text(
                "😀" * qwen_input_limits.MAX_PATH_CHARS,
                field="path",
                max_bytes=qwen_input_limits.MAX_PATH_BYTES,
                max_chars=qwen_input_limits.MAX_PATH_CHARS,
                code="PATH_TOO_LONG",
            )

    def test_task_scopes_commands_and_validation_failure_have_limits(self):
        with tempfile.TemporaryDirectory():
            cases = (
                ("task", "x" * (qwen_input_limits.MAX_TASK_CHARS + 1)),
                (
                    "allow",
                    ["path"] * (qwen_input_limits.MAX_ALLOW_SCOPES + 1),
                ),
                (
                    "read_only",
                    ["path"] * (qwen_input_limits.MAX_READ_ONLY_SCOPES + 1),
                ),
                (
                    "allowed_commands",
                    ["python"] * (qwen_input_limits.MAX_ALLOWED_COMMANDS + 1),
                ),
                (
                    "validation_commands",
                    ["python"] * (qwen_input_limits.MAX_VALIDATION_COMMANDS + 1),
                ),
                (
                    "validation_failure",
                    "x" * (qwen_input_limits.MAX_VALIDATION_FAILURE_CHARS + 1),
                ),
                (
                    "allow",
                    ["x" * (qwen_input_limits.MAX_PATH_CHARS + 1)],
                ),
                (
                    "allowed_commands",
                    ["x" * (qwen_input_limits.MAX_COMMAND_CHARS + 1)],
                ),
            )
            for field, value in cases:
                with self.subTest(field=field), mock.patch.object(
                    qwen_agent, "SandboxAgent", side_effect=AssertionError
                ):
                    with self.assertRaises((qwen_agent.InputLimitError, ValueError)):
                        request = self.agent_request(**{field: value})
                        asyncio.run(
                            qwen_agent.run_qwen_agent(**request)
                        )


class SharedInputBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_patch_task_and_test_output_reject_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "example.txt")
            source.write_text("hello\n", encoding="utf-8")
            with mock.patch(
                "qwen_worker_server.httpx.AsyncClient", side_effect=AssertionError
            ):
                with self.assertRaisesRegex(ValueError, "TASK_TOO_LARGE"):
                    await run_qwen_worker(
                        "x" * (qwen_input_limits.MAX_TASK_CHARS + 1),
                        directory,
                        ["example.txt"],
                    )

    async def test_patch_source_task_and_test_output_budget_rejects_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "example.txt")
            source.write_text("s" * qwen_worker_server.MAX_SOURCE_CHARS, encoding="utf-8")
            with mock.patch(
                "qwen_worker_server.httpx.AsyncClient", side_effect=AssertionError
            ):
                with self.assertRaisesRegex(ValueError, "MODEL_INPUT_TOO_LARGE"):
                    await run_qwen_worker(
                        "t" * qwen_input_limits.MAX_TASK_CHARS,
                        directory,
                        ["example.txt"],
                        "o" * qwen_input_limits.MAX_TEST_OUTPUT_CHARS,
                    )



if __name__ == "__main__":
    unittest.main()
