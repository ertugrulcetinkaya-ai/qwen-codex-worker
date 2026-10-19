import asyncio
import base64
import contextlib
import io
import json
import os
import re
import shlex
import shutil
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


def _sse_response(*events, status=200, headers=None):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream", **(headers or {})},
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

    def test_read_bounded_range_from_large_text_file(self):
        lines = [f"line-{index}" for index in range(1, 20_001)]
        self.source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["example.txt"])

        with self.assertRaisesRegex(AgentError, "file exceeds read limit"):
            agent.execute({"action": "read", "path": "example.txt"})
        result = agent.execute(
            {
                "action": "read",
                "path": "example.txt",
                "start_line": 15_000,
                "end_line": 15_002,
            }
        )

        self.assertEqual(result["content"], "\n".join(lines[14_999:15_002]))
        self.assertNotIn("example.txt", agent.fully_observed_files)
        self.assertGreater(agent.read_bytes, qwen_agent.MAX_READ_BYTES)

    def test_read_keeps_source_under_byte_limit_complete(self):
        source_text = "x" * (qwen_agent.MAX_READ_CHARS - 1)
        self.source.write_text(source_text, encoding="utf-8")
        agent = SandboxAgent(self.root, ["example.txt"])

        result = agent.execute({"action": "read", "path": "example.txt"})

        self.assertEqual(result["content"], source_text)
        self.assertNotIn("[truncated]", result["content"])
        self.assertIn("example.txt", agent.fully_observed_files)

    def test_read_uses_character_limit_and_keeps_multibyte_text_complete(self):
        source_text = "م" * 21_000
        self.assertLess(len(source_text.encode("utf-8")), qwen_agent.MAX_READ_BYTES)
        self.assertLess(len(source_text), qwen_agent.MAX_READ_CHARS)
        self.source.write_text(source_text, encoding="utf-8")
        agent = SandboxAgent(self.root, ["example.txt"])

        result = agent.execute({"action": "read", "path": "example.txt"})

        self.assertEqual(result["content"], source_text)
        self.assertNotIn("[truncated]", result["content"])
        self.assertIn("example.txt", agent.fully_observed_files)

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
        scratch = Path(captured_environment["HOME"])
        self.assertNotEqual(scratch, self.root.resolve())
        self.assertFalse(scratch.exists())
        self.assertEqual(
            captured_environment["PYTHONPYCACHEPREFIX"], str(scratch / "pycache")
        )
        self.assertNotIn("TOP_SECRET", captured_environment)
        self.assertNotIn(str(self.root), captured_environment["PATH"])

    def test_command_access_policy_uses_canonical_scopes(self):
        context = self.root / "context.txt"
        context.write_text("context\n", encoding="utf-8")
        scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, scratch)
        agent = SandboxAgent(
            self.root, ["example.txt"], read_only=["context.txt"]
        )

        model_policy = agent._command_access_policy(scratch, validation=False)
        validation_policy = agent._command_access_policy(scratch, validation=True)

        self.assertEqual(
            [scope.path for scope in model_policy.readable_paths],
            [
                self.root.resolve() / "example.txt",
                self.root.resolve() / "context.txt",
            ],
        )
        self.assertEqual(
            [scope.path for scope in model_policy.writable_paths],
            [self.root.resolve() / "example.txt"],
        )
        self.assertFalse(model_policy.validation)
        self.assertEqual(
            [scope.path for scope in validation_policy.readable_paths],
            [
                self.root.resolve() / "example.txt",
                self.root.resolve() / "context.txt",
            ],
        )
        self.assertEqual(validation_policy.writable_paths, ())
        self.assertTrue(validation_policy.validation)

    def test_validation_paths_are_separate_from_model_file_scopes(self):
        generated = self.root / "build-output"
        generated.mkdir()
        validation_source = self.root / "validation-source"
        validation_source.mkdir()
        agent = SandboxAgent(
            self.root,
            ["example.txt"],
            validation_read_paths=["validation-source"],
            validation_output_paths=["build-output"],
        )

        model_policy = agent._command_access_policy(Path("/tmp"), validation=False)
        validation_policy = agent._command_access_policy(Path("/tmp"), validation=True)

        self.assertEqual(
            [scope.path for scope in model_policy.readable_paths],
            [self.root.resolve() / "example.txt"],
        )
        self.assertEqual(
            [scope.path for scope in validation_policy.readable_paths],
            [validation_source.resolve()],
        )
        self.assertEqual(
            [scope.path for scope in validation_policy.writable_paths],
            [generated.resolve()],
        )
        self.assertTrue(validation_policy.validation)

    def test_command_profile_denies_forbidden_components_without_root_authority(self):
        scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, scratch)
        agent = SandboxAgent(self.root, ["example.txt"])
        policy = agent._command_access_policy(scratch, validation=False)
        profile, parameters = qwen_agent._macos_sandbox_profile(
            self.root, Path(sys.executable).resolve(), policy
        )

        pattern = qwen_agent._forbidden_command_path_regex(self.root)
        self.assertRegex(str(self.root / ".env.local"), re.compile(pattern))
        self.assertRegex(str(self.root / "nested" / ".ENV"), re.compile(pattern))
        self.assertRegex(
            str(self.root / "config" / "secrets.json"), re.compile(pattern)
        )
        self.assertNotRegex(str(self.root / "example.txt"), re.compile(pattern))
        self.assertIn(
            '(deny file-read* (regex (string-append "^"', profile
        )
        self.assertIn(
            '(deny file-write* (regex (string-append "^"', profile
        )
        self.assertNotIn(
            '(allow file-write* (subpath (param "SANDBOX_ROOT")))', profile
        )
        self.assertNotIn(
            '(allow file-read* file-map-executable (subpath (param "SANDBOX_ROOT")))',
            profile,
        )
        self.assertIn(("SCRATCH", str(scratch)), parameters)

    def test_model_command_mutations_are_accounted_once(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "changed.txt").write_text("before\n", encoding="utf-8")
        (generated / "deleted.txt").write_text("delete\n", encoding="utf-8")
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["generated"], allowed_commands=[command]
        )
        captured = {}

        class FakeSandbox:
            def run(self, _argv, **kwargs):
                captured.update(kwargs)
                (generated / "changed.txt").write_text("after\n", encoding="utf-8")
                (generated / "created.txt").write_text("new\n", encoding="utf-8")
                (generated / "deleted.txt").unlink()
                return qwen_agent.CommandResult(0, "", "")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": command})

        self.assertTrue(result["ok"])
        self.assertEqual(agent.mutation_generation, 1)
        self.assertEqual(
            agent.changed_paths,
            {
                "generated/changed.txt",
                "generated/created.txt",
                "generated/deleted.txt",
            },
        )
        self.assertFalse(captured["access_policy"].validation)
        self.assertFalse(captured["access_policy"].scratch_path.exists())

    def test_failed_model_command_rolls_back_writable_changes(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "changed.txt").write_text("before\n", encoding="utf-8")
        (generated / "deleted.txt").write_text("restore\n", encoding="utf-8")
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(self.root, ["generated"], allowed_commands=[command])

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                (generated / "changed.txt").write_text("partial\n", encoding="utf-8")
                (generated / "created.txt").write_text("remove\n", encoding="utf-8")
                (generated / "deleted.txt").unlink()
                return qwen_agent.CommandResult(1, "failed", "command error")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": command})

        self.assertFalse(result["ok"])
        self.assertCountEqual(
            result["rolled_back_paths"],
            [
                "generated/changed.txt",
                "generated/created.txt",
                "generated/deleted.txt",
            ],
        )
        self.assertEqual((generated / "changed.txt").read_text(), "before\n")
        self.assertFalse((generated / "created.txt").exists())
        self.assertEqual((generated / "deleted.txt").read_text(), "restore\n")
        self.assertEqual(agent.changed_paths, set())
        self.assertEqual(agent.mutation_generation, 0)
        self.assertEqual(agent.rollback_count, 3)

    def test_failed_command_symlink_swap_cannot_write_outside_writable_scope(self):
        command = f"{shlex.quote(sys.executable)} --version"
        for target_kind in ("outside", "private"):
            with self.subTest(target_kind=target_kind):
                generated = self.root / f"generated-{target_kind}"
                nested = generated / "nested"
                nested.mkdir(parents=True)
                (nested / "victim.txt").write_text("before snapshot\n")

                if target_kind == "outside":
                    outside_directory = tempfile.TemporaryDirectory(
                        prefix="qwen-rollback-outside-"
                    )
                    self.addCleanup(outside_directory.cleanup)
                    target = Path(outside_directory.name)
                else:
                    target = self.root / "private"
                    target.mkdir()
                (target / "victim.txt").write_text("protected sentinel\n")

                agent = SandboxAgent(
                    self.root, [generated.name], allowed_commands=[command]
                )

                class FakeSandbox:
                    def run(self, _argv, **_kwargs):
                        shutil.rmtree(nested)
                        nested.symlink_to(target, target_is_directory=True)
                        return qwen_agent.CommandResult(1, "failed", "injected")

                with mock.patch(
                    "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
                ):
                    result = agent.execute({"action": "run", "command": command})

                self.assertFalse(result["ok"])
                self.assertEqual(
                    (target / "victim.txt").read_text(encoding="utf-8"),
                    "protected sentinel\n",
                )
                self.assertTrue(nested.is_dir())
                self.assertFalse(nested.is_symlink())
                self.assertEqual(
                    (nested / "victim.txt").read_text(encoding="utf-8"),
                    "before snapshot\n",
                )
                self.assertEqual(agent.changed_paths, set())

    def test_failed_command_with_symlinked_file_scope_parent_fails_closed(self):
        generated = self.root / "generated"
        generated.mkdir()
        private = self.root / "private"
        private.mkdir()
        protected = private / "new.txt"
        protected.write_text("protected sentinel\n", encoding="utf-8")
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["generated/new.txt"], allowed_commands=[command]
        )

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                generated.rmdir()
                generated.symlink_to(private, target_is_directory=True)
                return qwen_agent.CommandResult(1, "failed", "injected")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            with self.assertRaisesRegex(
                AgentError, "could not be safely tracked or rolled back"
            ):
                agent.execute({"action": "run", "command": command})

        self.assertEqual(protected.read_text(encoding="utf-8"), "protected sentinel\n")
        self.assertTrue(generated.is_symlink())

    def test_failed_command_restores_symlink_object_without_following_target(self):
        generated = self.root / "generated"
        generated.mkdir()
        outside_directory = tempfile.TemporaryDirectory(
            prefix="qwen-rollback-link-target-"
        )
        self.addCleanup(outside_directory.cleanup)
        target = Path(outside_directory.name) / "target.txt"
        target.write_text("protected sentinel\n", encoding="utf-8")
        link = generated / "link.txt"
        link.symlink_to(target)
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(self.root, ["generated"], allowed_commands=[command])

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                link.unlink()
                link.write_text("temporary replacement\n", encoding="utf-8")
                return qwen_agent.CommandResult(1, "failed", "injected")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": command})

        self.assertFalse(result["ok"])
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.readlink(), target)
        self.assertEqual(target.read_text(encoding="utf-8"), "protected sentinel\n")

    def test_failed_command_restores_file_changes_in_read_only_nested_directory(self):
        generated = self.root / "generated"
        nested = generated / "nested"
        nested.mkdir(parents=True)
        tracked = nested / "tracked.txt"
        tracked.write_text("before\n", encoding="utf-8")
        nested.chmod(0o555)
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(self.root, ["generated"], allowed_commands=[command])

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                nested.chmod(0o755)
                tracked.write_text("partial\n", encoding="utf-8")
                (nested / "created.txt").write_text("created\n", encoding="utf-8")
                nested.chmod(0o555)
                return qwen_agent.CommandResult(1, "failed", "injected")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": command})

        self.assertFalse(result["ok"])
        self.assertEqual(tracked.read_text(encoding="utf-8"), "before\n")
        self.assertFalse((nested / "created.txt").exists())
        self.assertEqual(nested.stat().st_mode & 0o777, 0o555)

    def test_snapshot_rejects_directory_symlink_swap_between_check_and_open(self):
        generated = self.root / "generated"
        nested = generated / "nested"
        nested.mkdir(parents=True)
        (nested / "tracked.txt").write_text("inside\n", encoding="utf-8")
        outside_directory = tempfile.TemporaryDirectory(
            prefix="qwen-rollback-race-target-"
        )
        self.addCleanup(outside_directory.cleanup)
        outside = Path(outside_directory.name)
        protected = outside / "protected.txt"
        protected.write_text("protected sentinel\n", encoding="utf-8")
        agent = SandboxAgent(self.root, ["generated"])
        root_fd = agent._open_command_root_fd()
        original_open_directory = agent._open_verified_command_directory
        swapped = False

        def swap_before_open(parent_fd, name, observed):
            nonlocal swapped
            if name == "nested" and not swapped:
                nested.rename(generated / "nested-moved")
                nested.symlink_to(outside, target_is_directory=True)
                swapped = True
            return original_open_directory(parent_fd, name, observed)

        try:
            with mock.patch.object(
                agent,
                "_open_verified_command_directory",
                side_effect=swap_before_open,
            ):
                with self.assertRaisesRegex(AgentError, "changed during safe inspection"):
                    agent._snapshot_writable_state(root_fd)
        finally:
            os.close(root_fd)

        self.assertTrue(swapped)
        self.assertEqual(protected.read_text(encoding="utf-8"), "protected sentinel\n")

    def test_model_command_without_mutation_keeps_generation(self):
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["example.txt"], allowed_commands=[command]
        )

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                return qwen_agent.CommandResult(0, "", "")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": command})

        self.assertTrue(result["ok"])
        self.assertEqual(agent.changed_paths, set())
        self.assertEqual(agent.mutation_generation, 0)

    def test_command_scratch_path_is_redacted_from_command_result(self):
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["example.txt"], allowed_commands=[command]
        )

        class FakeSandbox:
            def run(self, _argv, **kwargs):
                scratch = kwargs["access_policy"].scratch_path
                return qwen_agent.CommandResult(0, f"scratch={scratch}\n", "")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            result = agent.execute({"action": "run", "command": command})

        self.assertEqual(result["stdout"], "scratch=[command scratch]\n")

    def test_command_scope_rebinding_to_in_root_symlink_is_rejected(self):
        allowed = self.root / "allowed"
        private = self.root / "private"
        allowed.mkdir()
        private.mkdir()
        (private / "secret.txt").write_text("secret\n", encoding="utf-8")
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["allowed"], allowed_commands=[command]
        )
        calls = 0

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                nonlocal calls
                calls += 1
                allowed.rmdir()
                allowed.symlink_to("private", target_is_directory=True)
                return qwen_agent.CommandResult(0, "", "")

        with mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            agent.execute({"action": "run", "command": command})
            with self.assertRaisesRegex(AgentError, "scope changed"):
                agent.execute({"action": "run", "command": command})
            with self.assertRaisesRegex(AgentError, "scope changed"):
                agent.run_validation_command(qwen_agent._parse_command(command))

        self.assertEqual(calls, 1)
        self.assertEqual((private / "secret.txt").read_text(encoding="utf-8"), "secret\n")

    def test_command_snapshot_limit_fails_before_execution(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "one.txt").write_text("one\n", encoding="utf-8")
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["generated"], allowed_commands=[command]
        )

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                raise AssertionError("command must not execute")

        with mock.patch.object(qwen_agent, "MAX_COMMAND_TRACKED_FILES", 1), mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            with self.assertRaisesRegex(AgentError, "snapshot exceeds file limit"):
                agent.execute({"action": "run", "command": command})

    def test_command_snapshot_byte_limit_fails_before_execution(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "one.txt").write_text("too large", encoding="utf-8")
        command = f"{shlex.quote(sys.executable)} --version"
        agent = SandboxAgent(
            self.root, ["generated"], allowed_commands=[command]
        )

        class FakeSandbox:
            def run(self, _argv, **_kwargs):
                raise AssertionError("command must not execute")

        with mock.patch.object(qwen_agent, "MAX_COMMAND_TRACKED_BYTES", 1), mock.patch(
            "qwen_agent._discover_command_sandbox", return_value=FakeSandbox()
        ):
            with self.assertRaisesRegex(AgentError, "snapshot exceeds byte limit"):
                agent.execute({"action": "run", "command": command})

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

    def test_grep_searches_large_text_file_with_bounded_result(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        content = b"x" * (qwen_agent.MAX_READ_BYTES * 2) + b"\nneedle at end\n"
        self.assertGreater(len(content), qwen_agent.MAX_READ_BYTES)
        self.assertLessEqual(len(content), qwen_agent.MAX_GREP_FILE_BYTES)
        (source_directory / "large.json").write_bytes(content)
        agent = SandboxAgent(self.root, ["src"])

        result = agent.execute(
            {"action": "grep", "query": "needle", "paths": ["src"]}
        )

        self.assertEqual(result["matches"], ["src/large.json:2:needle at end"])
        self.assertEqual(result["files_scanned"], 1)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["truncation_reason"], None)

    def test_grep_enforces_total_file_byte_budget(self):
        source_directory = self.root / "src"
        source_directory.mkdir()
        (source_directory / "a.txt").write_bytes(b"needle\n" + b"a" * 53)
        (source_directory / "b.txt").write_bytes(b"needle\n" + b"b" * 53)
        agent = SandboxAgent(self.root, ["src"])

        with mock.patch.object(qwen_agent, "MAX_GREP_TOTAL_BYTES", 64):
            result = agent.execute(
                {"action": "grep", "query": "needle", "paths": ["src"]}
            )

        self.assertEqual(result["matches"], ["src/a.txt:1:needle"])
        self.assertEqual(result["files_scanned"], 1)
        self.assertEqual(result["files_skipped"], 1)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["truncation_reason"], "byte_limit")
        self.assertEqual(result["skipped_reasons"]["total_byte_limit"], 1)

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

    def _run_python(
        self,
        code: str,
        *,
        allow=None,
        read_only=None,
        validation=False,
        validation_read_paths=None,
        validation_output_paths=None,
    ):
        command = self._python_command(code)
        agent = SandboxAgent(
            self.root,
            allow or ["input.txt"],
            read_only=read_only or [],
            allowed_commands=[command],
            validation_read_paths=validation_read_paths or [],
            validation_output_paths=validation_output_paths or [],
        )
        if validation:
            return agent.run_validation_command(qwen_agent._parse_command(command)), agent
        return agent.execute({"action": "run", "command": command}), agent

    def test_validation_writes_only_declared_generated_directory(self):
        generated = self.root / "build-output"
        generated.mkdir()
        result, _agent = self._run_python(
            "from pathlib import Path; Path('build-output/result.txt').write_text('built')",
            validation=True,
            validation_read_paths=["input.txt"],
            validation_output_paths=["build-output"],
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual((generated / "result.txt").read_text(), "built")

        denied, _agent = self._run_python(
            "from pathlib import Path; Path('input.txt').write_text('changed')",
            validation=True,
            validation_read_paths=["input.txt"],
            validation_output_paths=["build-output"],
        )
        self.assertFalse(denied["ok"])
        self.assertEqual(self.input_path.read_text(), "inside sentinel\n")

    def test_outside_file_read_is_denied_without_secret_output(self):
        outside = self.root.parent / f"{self.root.name}-outside-secret.txt"
        outside.write_text("TOP_SECRET_SENTINEL\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)

        result, _agent = self._run_python(
            f"from pathlib import Path; print(Path({str(outside)!r}).read_text())"
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("TOP_SECRET_SENTINEL", result["stdout"])
        self.assertNotIn("TOP_SECRET_SENTINEL", result["stderr"])

    def test_outside_file_write_is_denied(self):
        outside = self.root.parent / f"{self.root.name}-outside-write.txt"
        self.addCleanup(outside.unlink, missing_ok=True)

        result, _agent = self._run_python(
            f"from pathlib import Path; Path({str(outside)!r}).write_text('outside')"
        )

        self.assertFalse(result["ok"])
        self.assertFalse(outside.exists())

    def test_outside_symlink_read_is_denied_without_secret_output(self):
        outside = self.root.parent / f"{self.root.name}-outside-symlink.txt"
        outside.write_text("TOP_SECRET_SYMLINK_SENTINEL\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        (self.root / "link.txt").symlink_to(outside)

        result, _agent = self._run_python(
            "from pathlib import Path; print(Path('link.txt').read_text())"
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("TOP_SECRET_SYMLINK_SENTINEL", result["stdout"])
        self.assertNotIn("TOP_SECRET_SYMLINK_SENTINEL", result["stderr"])

    def test_failed_command_symlink_swap_rollback_stays_inside_writable_scope(self):
        generated = self.root / "generated"
        nested = generated / "nested"
        nested.mkdir(parents=True)
        (nested / "victim.txt").write_text("before snapshot\n", encoding="utf-8")
        outside = self.root.parent / f"{self.root.name}-rollback-outside"
        outside.mkdir()
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (outside / "victim.txt").write_text("protected sentinel\n", encoding="utf-8")

        code = (
            "import os, shutil, sys; "
            "shutil.rmtree('generated/nested'); "
            f"os.symlink({str(outside)!r}, 'generated/nested', target_is_directory=True); "
            "print('SYMLINK_CREATED', flush=True); sys.exit(23)"
        )
        result, _agent = self._run_python(code, allow=["generated"])

        self.assertFalse(result["ok"], result)
        self.assertIn("SYMLINK_CREATED", result["stdout"])
        self.assertEqual(
            (outside / "victim.txt").read_text(encoding="utf-8"),
            "protected sentinel\n",
        )
        self.assertTrue(nested.is_dir())
        self.assertFalse(nested.is_symlink())
        self.assertEqual(
            (nested / "victim.txt").read_text(encoding="utf-8"),
            "before snapshot\n",
        )

    def test_failed_command_cannot_move_writable_scope_outside_sandbox(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "nested").mkdir()
        private = self.root / "private"
        private.mkdir()
        destination = private / "generated"

        code = (
            "import os, sys\n"
            "try:\n"
            "    os.rename('generated', 'private/generated')\n"
            "except OSError as exc:\n"
            "    print('RENAME_DENIED', type(exc).__name__, flush=True)\n"
            "    sys.exit(23)\n"
            "print('RENAME_ALLOWED', flush=True)\n"
        )
        result, _agent = self._run_python(code, allow=["generated"])

        self.assertFalse(result["ok"], result)
        self.assertIn("RENAME_DENIED", result["stdout"])
        self.assertTrue(generated.is_dir())
        self.assertFalse(destination.exists())

    def test_out_of_scope_sandbox_read_is_denied_without_secret_output(self):
        private = self.root / "private.txt"
        private.write_text("PRIVATE_SENTINEL\n", encoding="utf-8")

        result, _agent = self._run_python(
            "from pathlib import Path; print(Path('private.txt').read_text())"
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("PRIVATE_SENTINEL", result["stdout"])
        self.assertNotIn("PRIVATE_SENTINEL", result["stderr"])

    def test_read_only_file_is_readable(self):
        context = self.root / "context.txt"
        context.write_text("READ_ONLY_SENTINEL\n", encoding="utf-8")

        result, _agent = self._run_python(
            "from pathlib import Path; print(Path('context.txt').read_text(), end='')",
            read_only=["context.txt"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "READ_ONLY_SENTINEL\n")

    def test_read_only_file_write_is_denied(self):
        context = self.root / "context.txt"
        context.write_text("original\n", encoding="utf-8")

        result, _agent = self._run_python(
            "from pathlib import Path; Path('context.txt').write_text('changed')",
            read_only=["context.txt"],
        )

        self.assertFalse(result["ok"])
        self.assertEqual(context.read_bytes(), b"original\n")

    def test_model_run_writable_file_modification_is_accounted(self):
        result, agent = self._run_python(
            "from pathlib import Path; Path('input.txt').write_text('changed')"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self.input_path.read_text(encoding="utf-8"), "changed")
        self.assertEqual(agent.changed_paths, {"input.txt"})
        self.assertEqual(agent.mutation_generation, 1)

    def test_model_run_multiple_mutations_advance_generation_once(self):
        generated = self.root / "generated"
        generated.mkdir()
        (generated / "one.txt").write_text("one", encoding="utf-8")
        (generated / "two.txt").write_text("two", encoding="utf-8")

        result, agent = self._run_python(
            "from pathlib import Path; "
            "Path('generated/one.txt').write_text('changed'); "
            "Path('generated/two.txt').unlink(); "
            "Path('generated/three.txt').write_text('created')",
            allow=["generated"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual(agent.mutation_generation, 1)
        self.assertEqual(
            agent.changed_paths,
            {
                "generated/one.txt",
                "generated/two.txt",
                "generated/three.txt",
            },
        )

    def test_unauthorized_new_file_is_denied(self):
        result, _agent = self._run_python(
            "from pathlib import Path; Path('output.txt').write_text('unauthorized')"
        )

        self.assertFalse(result["ok"])
        self.assertFalse((self.root / "output.txt").exists())

    def test_writable_directory_allows_child_creation_only_inside_scope(self):
        generated = self.root / "generated"
        generated.mkdir()
        result, _agent = self._run_python(
            "from pathlib import Path; Path('generated/output.txt').write_text('ok')",
            allow=["generated"],
        )
        denied, _agent = self._run_python(
            "from pathlib import Path; Path('other.txt').write_text('denied')",
            allow=["generated"],
        )

        self.assertTrue(result["ok"])
        self.assertEqual((generated / "output.txt").read_text(encoding="utf-8"), "ok")
        self.assertFalse(denied["ok"])
        self.assertFalse((self.root / "other.txt").exists())

    def test_forbidden_env_files_are_denied_inside_readable_directory(self):
        source = self.root / "source"
        nested = source / "nested"
        nested.mkdir(parents=True)
        (source / ".env").write_text("ENV_SENTINEL\n", encoding="utf-8")
        (source / ".env.local").write_text("LOCAL_SENTINEL\n", encoding="utf-8")
        (nested / ".env").write_text("NESTED_SENTINEL\n", encoding="utf-8")

        for relative, sentinel in (
            ("source/.env", "ENV_SENTINEL"),
            ("source/.env.local", "LOCAL_SENTINEL"),
            ("source/nested/.env", "NESTED_SENTINEL"),
        ):
            with self.subTest(relative=relative):
                result, _agent = self._run_python(
                    f"from pathlib import Path; print(Path({relative!r}).read_text())",
                    read_only=["source"],
                )

                self.assertFalse(result["ok"])
                self.assertNotIn(sentinel, result["stdout"])
                self.assertNotIn(sentinel, result["stderr"])

    def test_secret_basename_is_denied_inside_readable_directory(self):
        config = self.root / "config"
        config.mkdir()
        (config / "secrets.json").write_text("SECRET_SENTINEL\n", encoding="utf-8")

        result, _agent = self._run_python(
            "from pathlib import Path; print(Path('config/secrets.json').read_text())",
            read_only=["config"],
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("SECRET_SENTINEL", result["stdout"])
        self.assertNotIn("SECRET_SENTINEL", result["stderr"])

    def test_intra_sandbox_symlink_bypass_is_denied(self):
        allowed = self.root / "allowed"
        allowed.mkdir()
        private = self.root / "private.txt"
        private.write_text("SYMLINK_SECRET\n", encoding="utf-8")
        (allowed / "link.txt").symlink_to(Path("..") / "private.txt")

        result, _agent = self._run_python(
            "from pathlib import Path; print(Path('allowed/link.txt').read_text())",
            read_only=["allowed"],
        )

        self.assertFalse(result["ok"])
        self.assertNotIn("SYMLINK_SECRET", result["stdout"])
        self.assertNotIn("SYMLINK_SECRET", result["stderr"])

    def test_rebound_directory_scope_is_rejected_for_model_and_validation(self):
        allowed = self.root / "allowed"
        private = self.root / "private"
        allowed.mkdir()
        private.mkdir()
        (private / "secret.txt").write_text("REBIND_SECRET\n", encoding="utf-8")
        rebind_command = self._python_command(
            "from pathlib import Path; Path('allowed').rmdir(); "
            "Path('allowed').symlink_to('private', target_is_directory=True)"
        )
        read_command = self._python_command(
            "from pathlib import Path; print(Path('allowed/secret.txt').read_text())"
        )
        agent = SandboxAgent(
            self.root,
            ["allowed"],
            allowed_commands=[rebind_command, read_command],
        )

        first = agent.execute({"action": "run", "command": rebind_command})
        self.assertTrue(first["ok"])
        with self.assertRaisesRegex(AgentError, "scope changed"):
            agent.execute({"action": "run", "command": read_command})
        with self.assertRaisesRegex(AgentError, "scope changed"):
            agent.run_validation_command(qwen_agent._parse_command(read_command))
        self.assertEqual(
            (private / "secret.txt").read_text(encoding="utf-8"),
            "REBIND_SECRET\n",
        )

    def test_validation_source_write_is_denied(self):
        result, agent = self._run_python(
            "from pathlib import Path; Path('input.txt').write_text('changed')",
            validation=True,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(self.input_path.read_text(encoding="utf-8"), "inside sentinel\n")
        policy = agent._command_access_policy(Path("/tmp"), validation=True)
        self.assertTrue(policy.validation)
        self.assertEqual(policy.writable_paths, ())

    def test_validation_read_works(self):
        result, _agent = self._run_python(
            "from pathlib import Path; print(Path('input.txt').read_text(), end='')",
            validation=True,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "inside sentinel\n")

        developer_directory = qwen_agent._active_developer_directory()
        self.assertIsNotNone(developer_directory)
        assert developer_directory is not None
        xcode_select = "/usr/bin/xcode-select -p"
        toolchain_listing = shlex.join(
            ["/bin/ls", "-ld", str(developer_directory)]
        )
        agent = SandboxAgent(
            self.root,
            ["input.txt"],
            allowed_commands=[xcode_select, toolchain_listing],
        )

        model_xcode_result = agent.execute(
            {"action": "run", "command": xcode_select}
        )
        self.assertFalse(model_xcode_result["ok"])
        model_toolchain_result = agent.execute(
            {"action": "run", "command": toolchain_listing}
        )
        self.assertFalse(model_toolchain_result["ok"])

        validation_xcode_result = agent.run_validation_command(
            qwen_agent._parse_command(xcode_select)
        )
        self.assertEqual(validation_xcode_result["returncode"], 0)
        self.assertEqual(
            validation_xcode_result["stdout"].strip(),
            str(developer_directory),
        )
        validation_toolchain_result = agent.run_validation_command(
            qwen_agent._parse_command(toolchain_listing)
        )
        self.assertEqual(validation_toolchain_result["returncode"], 0)
        self.assertIn(
            str(developer_directory), validation_toolchain_result["stdout"]
        )

    def test_flutter_sdk_cache_is_available_only_to_host_validation(self):
        with tempfile.TemporaryDirectory(prefix="flutter-sdk-fixture-") as sdk_dir:
            sdk_root = Path(sdk_dir) / "flutter"
            binary_directory = sdk_root / "bin"
            cache_directory = binary_directory / "cache"
            cache_directory.mkdir(parents=True)
            flutter = binary_directory / "flutter"
            flutter.write_text(
                "#!/bin/sh\n"
                'SDK_BIN=${0%/*}\n'
                'printf "ready\\n" > "$SDK_BIN/cache/engine.stamp"\n'
                "exit 0\n",
                encoding="utf-8",
            )
            flutter.chmod(0o755)
            stamp = cache_directory / "engine.stamp"
            command = shlex.join([str(flutter), "validate"])
            agent = SandboxAgent(
                self.root,
                ["input.txt"],
                allowed_commands=[command],
                validation_read_paths=["input.txt"],
            )

            model_result = agent.execute(
                {"action": "run", "command": command}
            )
            self.assertFalse(model_result["ok"])
            self.assertEqual(model_result["returncode"], 1)
            self.assertIn(
                "Flutter SDK cache access was denied",
                model_result["stderr"],
            )
            self.assertFalse(stamp.exists())

            validation_result = agent.run_validation_command(
                qwen_agent._parse_command(command)
            )
            self.assertEqual(validation_result["returncode"], 0, validation_result)
            self.assertEqual(stamp.read_text(encoding="utf-8"), "ready\n")
            self.assertEqual(
                self.input_path.read_text(encoding="utf-8"), "inside sentinel\n"
            )

    def test_command_scratch_is_writable_without_project_mutation(self):
        result, agent = self._run_python(
            "import os; from pathlib import Path; "
            "(Path(os.environ['TMPDIR']) / 'runtime-cache').write_text('cache')"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(agent.changed_paths, set())
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["input.txt"])

    def test_command_scratch_path_is_not_returned_to_model(self):
        result, _agent = self._run_python(
            "import os; print(os.environ['TMPDIR'])"
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stdout"], "[command scratch]\n")

    def test_command_scratch_is_cleaned_up(self):
        created: list[Path] = []

        class RecordingTemporaryDirectory:
            def __init__(self, *args, **kwargs):
                self.path = Path(tempfile.mkdtemp(*args, **kwargs))
                created.append(self.path)

            def __enter__(self):
                return str(self.path)

            def __exit__(self, _type, _value, _traceback):
                shutil.rmtree(self.path)
                return False

        with mock.patch.object(
            qwen_agent.tempfile, "TemporaryDirectory", RecordingTemporaryDirectory
        ):
            result, _agent = self._run_python(
                "import os; from pathlib import Path; "
                "(Path(os.environ['TMPDIR']) / 'runtime-cache').write_text('cache')"
            )

        self.assertTrue(result["ok"])
        # discover() also uses a temporary directory for its backend probe.
        self.assertEqual(len(created), 2)
        self.assertTrue(all(not path.exists() for path in created))

    def test_network_connection_is_denied(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(listener.close)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        result, _agent = self._run_python(
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
        result, _agent = self._run_python(
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
    def test_action_trace_metadata_is_hashed_and_content_free(self):
        metadata = qwen_agent._action_trace_metadata(
            {"action": "read", "path": "private/source.py"},
            {
                "ok": True,
                "action": "read",
                "path": "private/source.py",
                "content": "SOURCE_CONTENT_SENTINEL",
            },
        )
        self.assertEqual(len(metadata["action_sha256"]), 16)
        self.assertEqual(len(metadata["target_sha256"]), 16)
        self.assertEqual(len(metadata["result_sha256"]), 16)
        self.assertEqual(metadata["path"], "private/source.py")
        self.assertEqual(metadata["result_ok"], True)
        self.assertIsNone(metadata["returncode"])
        self.assertNotIn("SOURCE_CONTENT_SENTINEL", str(metadata))

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
        self.assertEqual(qwen_agent.AGENT_RESPONSE_FORMAT["type"], "json_schema")
        self.assertEqual(
            qwen_agent.AGENT_RESPONSE_FORMAT["json_schema"]["name"],
            "qwen_worker_action",
        )
        schema = qwen_agent.AGENT_RESPONSE_FORMAT["json_schema"]["schema"]
        alternatives = schema["oneOf"]
        self.assertEqual(len(alternatives), 6)
        self.assertNotIn("$ref", json.dumps(schema))
        self.assertNotIn("$defs", json.dumps(schema))

        expected_required = {
            "read": ["action", "path"],
            "grep": ["action", "query"],
            "replace": ["action", "path", "old", "new"],
            "write": ["action", "path", "content"],
            "run": ["action", "command"],
            "done": ["action", "summary"],
        }
        action_names = []
        for alternative in alternatives:
            action = alternative["properties"]["action"]["enum"][0]
            action_names.append(action)
            self.assertEqual(alternative["required"], expected_required[action])
            self.assertFalse(alternative["additionalProperties"])
        self.assertEqual(set(action_names), set(expected_required))
        self.assertIn('{"action":"run","command":"one explicitly allowed command"}', qwen_agent.AGENT_SYSTEM_PROMPT)

    def test_edit_response_format_only_allows_mutating_actions(self):
        self.assertEqual(qwen_agent.EDIT_RESPONSE_FORMAT["type"], "json_schema")
        allowed = [
            alternative["properties"]["action"]["enum"][0]
            for alternative in qwen_agent.EDIT_RESPONSE_FORMAT["json_schema"]["schema"]["oneOf"]
        ]
        self.assertEqual(allowed, ["replace", "write"])

    def test_edit_response_format_constrains_exact_file_scopes(self):
        response_format = qwen_agent._edit_response_format_for_paths(
            ["src/worker.py", "src/runtime.py"]
        )
        paths = [
            alternative["properties"]["path"]["enum"]
            for alternative in response_format["json_schema"]["schema"]["oneOf"]
        ]
        self.assertEqual(paths, [["src/runtime.py", "src/worker.py"]] * 2)
        self.assertNotIn(
            "enum",
            qwen_agent.EDIT_RESPONSE_FORMAT["json_schema"]["schema"]["oneOf"][0][
                "properties"
            ]["path"],
        )
        directory_format = qwen_agent._edit_response_format_for_paths(None)
        self.assertNotIn(
            "enum",
            directory_format["json_schema"]["schema"]["oneOf"][0]["properties"][
                "path"
            ],
        )
        repair_format = qwen_agent._edit_response_format_for_paths(
            ["src/worker.py", "src/runtime.py"],
            repair_read_path="src/runtime.py",
        )
        repair_actions = {
            alternative["properties"]["action"]["enum"][0]: alternative
            for alternative in repair_format["json_schema"]["schema"]["oneOf"]
        }
        self.assertEqual(set(repair_actions), {"read"})
        self.assertEqual(
            repair_actions["read"]["properties"]["path"]["enum"],
            ["src/runtime.py"],
        )

    def test_action_progress_guard_blocks_exact_edit_that_failed_validation(self):
        guard = qwen_agent.ActionProgressGuard()
        action = {
            "action": "write",
            "path": "src/runtime.py",
            "content": "incorrect implementation\n",
        }
        result = {
            "ok": True,
            "action": "write",
            "post_edit_validation": {"ok": False, "stderr": "compile error"},
        }

        self.assertEqual(
            guard.observe(action, result, mutation_occurred=True), 0
        )
        self.assertTrue(guard.failure_pending)
        self.assertTrue(guard.is_previously_failed_edit(action))
        self.assertFalse(
            guard.is_previously_failed_edit(
                {**action, "content": "corrected implementation\n"}
            )
        )

    def test_action_progress_guard_warns_then_stops_stagnant_repeats(self):
        guard = qwen_agent.ActionProgressGuard()
        action = {"action": "grep", "query": "TODO", "paths": ["src"]}
        result = {"ok": True, "action": "grep", "matches": []}

        self.assertEqual(
            guard.observe(action, result, mutation_occurred=False), 1
        )
        self.assertEqual(
            guard.observe(action, result, mutation_occurred=False), 2
        )
        self.assertEqual(
            guard.observe(action, result, mutation_occurred=False), 3
        )
        with self.assertRaisesRegex(RuntimeError, "AGENT_REPEATED_ACTION_LIMIT"):
            guard.observe(action, result, mutation_occurred=False)
        self.assertEqual(guard.repeated_actions, 3)
        self.assertEqual(
            guard.observe(action, result, mutation_occurred=True), 0
        )

    def test_action_progress_guard_treats_done_summaries_as_same_action(self):
        guard = qwen_agent.ActionProgressGuard()
        result = {
            "ok": False,
            "action": "done",
            "error": "validation_repair_required",
        }
        self.assertEqual(
            guard.observe(
                {"action": "done", "summary": "first"},
                result,
                mutation_occurred=False,
            ),
            1,
        )
        self.assertEqual(
            guard.observe(
                {"action": "done", "summary": "different wording"},
                result,
                mutation_occurred=False,
            ),
            2,
        )

    def test_action_progress_guard_treats_noisy_failed_run_as_same_attempt(self):
        guard = qwen_agent.ActionProgressGuard()
        action = {
            "action": "run",
            "command": "python3 -m unittest test_example",
        }
        self.assertEqual(
            guard.observe(
                action,
                {
                    "ok": False,
                    "returncode": 1,
                    "stdout": "Ran 1 test in 0.01s",
                    "stderr": "FAILED (failures=1)",
                },
                mutation_occurred=False,
            ),
            1,
        )
        self.assertEqual(
            guard.observe(
                action,
                {
                    "ok": False,
                    "returncode": 1,
                    "stdout": "Ran 1 test in 0.02s",
                    "stderr": "FAILED (failures=1)",
                },
                mutation_occurred=False,
            ),
            2,
        )

    def test_action_progress_guard_bounds_discovery_until_pass_or_mutation(self):
        guard = qwen_agent.ActionProgressGuard()
        for index in range(qwen_agent.MAX_DISCOVERY_ACTIONS_WITHOUT_PROGRESS):
            self.assertEqual(
                guard.observe(
                    {"action": "grep", "query": f"term-{index}"},
                    {"ok": True, "matches": []},
                    mutation_occurred=False,
                ),
                1,
            )
        self.assertEqual(
            guard.discovery_actions,
            qwen_agent.MAX_DISCOVERY_ACTIONS_WITHOUT_PROGRESS,
        )
        with self.assertRaisesRegex(RuntimeError, "AGENT_DISCOVERY_LIMIT"):
            guard.observe(
                {"action": "read", "path": "src/example.py"},
                {"ok": True, "content": ""},
                mutation_occurred=False,
            )
        with self.assertRaisesRegex(RuntimeError, "AGENT_DISCOVERY_LIMIT"):
            guard.observe(
                {"action": "run", "command": "python -m unittest"},
                {"ok": False, "returncode": 1},
                mutation_occurred=False,
            )
        self.assertEqual(
            guard.observe(
                {"action": "run", "command": "python -m unittest"},
                {"ok": True, "returncode": 0},
                mutation_occurred=False,
            ),
            1,
        )
        self.assertEqual(guard.discovery_actions, 0)

    def test_parse_action_accepts_run_and_rejects_schema_mismatches(self):
        action = parse_action(
            '{"action":"run","command":"python -m unittest -q"}'
        )
        self.assertEqual(
            action,
            {"action": "run", "command": "python -m unittest -q"},
        )
        for raw, message in (
            ('{"action":"run"}', "missing required field"),
            (
                '{"action":"run","command":"git status","extra":true}',
                "unsupported field",
            ),
            (
                '{"action":"read","path":"file.py","start_line":true}',
                "start_line must be integer",
            ),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(AgentError, message):
                parse_action(raw)

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
        with mock.patch.object(
            qwen_agent_cli,
            "run_qwen_agent_profile",
            new_callable=mock.AsyncMock,
            return_value=result.as_dict(),
        ):
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

        self.record(
            conversation,
            read_action,
            read_result,
            files_read=("example.txt",),
            actions_completed=2,
        )
        messages, stats = conversation.snapshot()
        duplicate_read = json.loads(messages[-1]["content"])
        self.assertTrue(duplicate_read["content_reused"])
        self.assertIn("refer to that content", duplicate_read["content"])
        self.assertNotIn("CURRENT_READ_SENTINEL", messages[-1]["content"])
        self.assertEqual(stats["deduplicated_reads"], 1)
        self.assertEqual(
            sum(
                item["content"].count("CURRENT_READ_SENTINEL")
                for item in messages
                if item["role"] == "user"
                and item["content"].startswith("{")
            ),
            1,
        )

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
            actions_completed=3,
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

    def test_large_complete_read_survives_compaction_and_is_invalidated_by_edit(self):
        conversation = self.make_conversation()
        source = "PINNED_SOURCE_SENTINEL\n" + ("x" * 35_000)
        self.record(
            conversation,
            {"action": "read", "path": "src/large.py"},
            {
                "ok": True,
                "action": "read",
                "path": "src/large.py",
                "content": source,
            },
            files_read=("src/large.py",),
        )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 2):
            self.record(
                conversation,
                {"action": "read", "path": "tests/test_large.py"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "tests/test_large.py",
                    "content": f"later-read-{index}",
                },
                files_read=("src/large.py", "tests/test_large.py"),
                actions_completed=index + 2,
            )

        messages, stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertIn("PINNED COMPLETE READ", rendered)
        self.assertIn("PINNED_SOURCE_SENTINEL", rendered)
        self.assertEqual(stats["pinned_read_chars"], len(source))
        self.assertTrue(stats["history_compacted"])
        self.assertLessEqual(stats["history_chars"], MAX_AGENT_HISTORY_CHARS)

        self.record(
            conversation,
            {"action": "read", "path": "src/large.py"},
            {
                "ok": True,
                "action": "read",
                "path": "src/large.py",
                "content": source,
            },
            files_read=("src/large.py",),
            actions_completed=MAX_AGENT_RECENT_INTERACTIONS + 3,
        )
        self.assertEqual(conversation._deduplicated_reads, 1)

        self.record(
            conversation,
            {"action": "replace", "path": "src/large.py", "old": "x", "new": "y"},
            {"ok": True, "action": "replace", "path": "src/large.py", "replaced": 1},
            files_changed=("src/large.py",),
        )
        messages, stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertNotIn("PINNED_SOURCE_SENTINEL", rendered)
        self.assertEqual(stats["pinned_read_chars"], 0)

    def test_multiple_complete_reads_survive_compaction_until_each_file_changes(self):
        conversation = self.make_conversation()
        source_reads = {
            "test/runtime_test.dart": "PINNED_TEST_SENTINEL\n" + ("t" * 21_400),
            "lib/runtime.dart": "PINNED_RUNTIME_SENTINEL\n" + ("r" * 12_500),
            "lib/settings.dart": "PINNED_SETTINGS_SENTINEL\n" + ("s" * 12_000),
        }
        for index, (path, content) in enumerate(source_reads.items(), start=1):
            self.record(
                conversation,
                {"action": "read", "path": path},
                {
                    "ok": True,
                    "action": "read",
                    "path": path,
                    "content": content,
                },
                files_read=source_reads,
                actions_completed=index,
            )
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 2):
            self.record(
                conversation,
                {"action": "read", "path": "test/other_test.dart"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "test/other_test.dart",
                    "content": f"filler-{index}",
                },
                files_read=source_reads,
                actions_completed=index + len(source_reads) + 1,
            )

        messages, stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        for content in source_reads.values():
            self.assertIn(content.splitlines()[0], rendered)
        self.assertEqual(
            stats["pinned_read_chars"],
            sum(len(content) for content in source_reads.values()),
        )
        self.assertTrue(stats["history_compacted"])
        self.assertLessEqual(stats["history_chars"], MAX_AGENT_HISTORY_CHARS)

        path, content = next(iter(source_reads.items()))
        self.record(
            conversation,
            {"action": "read", "path": path},
            {"ok": True, "action": "read", "path": path, "content": content},
            files_read=source_reads,
            actions_completed=20,
        )
        self.assertEqual(conversation._deduplicated_reads, 1)

        self.record(
            conversation,
            {"action": "replace", "path": path, "old": "old", "new": "new"},
            {"ok": True, "action": "replace", "path": path, "replaced": 1},
            files_read=source_reads,
            files_changed=(path,),
            actions_completed=21,
        )
        messages, stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertNotIn(content.splitlines()[0], rendered)
        for other_path, other_content in list(source_reads.items())[1:]:
            self.assertIn(other_content.splitlines()[0], rendered)
        self.assertEqual(
            stats["pinned_read_chars"],
            sum(len(value) for key, value in source_reads.items() if key != path),
        )

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

        self.record(
            conversation,
            {"action": "read", "path": "example.txt"},
            {
                "ok": True,
                "action": "read",
                "path": "example.txt",
                "content": "current-6",
            },
            files_read=("example.txt",),
            actions_completed=MAX_AGENT_RECENT_INTERACTIONS + 2,
        )
        messages, _stats = conversation.snapshot()
        self.assertEqual(
            json.loads(messages[-1]["content"])["content"], "current-6"
        )

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
                "stdout": (
                    "OLD_COMMAND_OUTPUT_SENTINEL\n"
                    "lib/example.dart:7: Error: The getter 'missing' isn't defined."
                ),
                "stderr": "compilation failed",
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
        self.assertIn("The getter 'missing' isn't defined", rendered)
        self.assertIn("last_command=run", rendered)
        self.assertIn("returncode=1", rendered)

    def test_post_edit_validation_failure_survives_context_compaction(self):
        conversation = self.make_conversation()
        for index in range(MAX_AGENT_RECENT_INTERACTIONS + 1):
            self.record(
                conversation,
                {"action": "read", "path": "src/example.py"},
                {
                    "ok": True,
                    "action": "read",
                    "path": "src/example.py",
                    "content": f"read-{index}",
                },
                files_read=("src/example.py",),
                actions_completed=index + 1,
            )
        self.record(
            conversation,
            {"action": "replace", "path": "src/example.py"},
            {
                "ok": True,
                "action": "replace",
                "path": "src/example.py",
                "post_edit_validation": {
                    "ok": False,
                    "action": "post_edit_validation",
                    "command": "dart analyze",
                    "returncode": 1,
                    "stderr": "lib/example.dart:9: Error: The getter 'missing' isn't defined.",
                },
            },
            files_changed=("src/example.py",),
        )

        messages, stats = conversation.snapshot()
        rendered = "\n".join(item["content"] for item in messages)
        self.assertTrue(stats["history_compacted"])
        self.assertIn("last_validation_failure", rendered)
        self.assertIn("The getter 'missing' isn't defined", rendered)
        self.assertIn("last_edit_summary:", rendered)
        self.assertIn("replace path=src/example.py", rendered)

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

    def test_failed_validation_requires_repair_evidence(self):
        state = qwen_agent.ValidationState()
        failed = ValidationResult("git diff", 1, "failure", "")
        self.assertFalse(state.finish_round((failed,), 0))
        self.assertTrue(state.repair_required)
        state.prepare_retry()
        self.assertTrue(state.repair_required)
        state.mark_mutation(1)
        self.assertFalse(state.repair_required)
        self.assertEqual(state.phase, ValidationPhase.EDITING)

    def test_required_validation_pass_clears_repair_requirement(self):
        state = qwen_agent.ValidationState()
        state.finish_round((ValidationResult("git diff", 1, "", ""),), 0)
        state.mark_required_validation_passed()
        self.assertFalse(state.repair_required)

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

    def test_compact_state_tracks_remaining_action_budget(self):
        conversation = AgentConversation(
            "implement the requested change",
            ["example.txt"],
            (),
            (),
            None,
            max_actions=8,
        )
        conversation.record(
            json.dumps({"action": "grep", "query": "target", "paths": ["src"]}),
            {"action": "grep", "query": "target", "paths": ["src"]},
            {"ok": True, "action": "grep", "matches": []},
            files_read=(),
            files_changed=(),
            actions_completed=3,
            protocol_errors=0,
        )

        messages, _stats = conversation.snapshot()
        state = messages[2]["content"]
        self.assertIn("action_limit=8", state)
        self.assertIn("actions_remaining=5", state)


@mock.patch("qwen_agent._omlx_api_headers", new=lambda: {})
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
        capture_edit_modes=None,
        capture_edit_paths=None,
        capture_repair_read_paths=None,
        resume=False,
    ):
        generations = list(actions)
        fake_sandbox = self.FakeCommandSandbox(outcomes)
        self.last_fake_sandbox = fake_sandbox

        async def fake_turn(_client, _base, _model, messages, **kwargs):
            if capture_messages is not None:
                capture_messages.append(messages)
            if capture_edit_modes is not None:
                capture_edit_modes.append(kwargs.get("require_edit", False))
            if capture_edit_paths is not None:
                capture_edit_paths.append(kwargs.get("editable_file_paths"))
            if capture_repair_read_paths is not None:
                capture_repair_read_paths.append(kwargs.get("repair_read_path"))
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
                    resume=resume,
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
        self.assertTrue(fake_sandbox.calls[0][1]["access_policy"].validation)

    async def test_host_validation_infrastructure_failure_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.txt").write_text("unchanged\n", encoding="utf-8")
            agent = SandboxAgent(
                directory,
                ["example.txt"],
                allowed_commands=[self.validation_command],
            )
            policy = qwen_agent.ValidationPolicy.from_commands(
                [self.validation_command], agent.allowed_commands
            )
            runner = qwen_agent.ValidationRunner(agent, policy)
            with mock.patch(
                "qwen_agent._discover_command_sandbox",
                side_effect=AgentError("command sandbox is unavailable"),
            ):
                infrastructure_result = runner.run()

            self.assertEqual(infrastructure_result[0].returncode, 127)
            self.assertFalse(infrastructure_result[0].passed)
            self.assertIn(
                "validation command execution failed",
                infrastructure_result[0].stderr,
            )

            state = qwen_agent.ValidationState()
            state.start_round(1)
            self.assertFalse(
                state.finish_round(infrastructure_result, agent.mutation_generation)
            )
            self.assertIsNone(state.validated_generation)

            ordinary_failure_sandbox = self.FakeCommandSandbox(
                [(1, "test assertion failed", "")]
            )
            with mock.patch(
                "qwen_agent._discover_command_sandbox",
                return_value=ordinary_failure_sandbox,
            ):
                test_failure = runner.run()
            self.assertEqual(test_failure[0].returncode, 1)
            self.assertFalse(test_failure[0].passed)

    async def test_resume_returns_validated_without_model_call_when_tests_pass(self):
        result, fake_sandbox = await self.invoke(
            [{"action": "done", "summary": "must not be requested"}],
            [(0, "existing sandbox passes", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            resume=True,
        )

        self.assertEqual(result.status, "validated")
        self.assertEqual(result.actions, 0)
        self.assertEqual(result.model_calls, 0)
        self.assertIsNone(result.prompt_tokens)
        self.assertEqual(result.validation_rounds, 1)
        self.assertEqual(len(fake_sandbox.calls), 1)

    async def test_resume_failure_is_context_for_repair_and_counted(self):
        messages = []
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "good",
                },
                {"action": "done", "summary": "fixed"},
            ],
            [
                (1, "resume failure output", "resume failed"),
                (0, "post-edit pass", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            capture_messages=messages,
            resume=True,
        )

        self.assertTrue(result.validated)
        self.assertEqual(result.validation_rounds, 2)
        self.assertEqual(result.validation_failures, 1)
        self.assertIn("RESUME_PREFLIGHT_FAILURES", messages[0][1]["content"])
        self.assertIn("resume failure output", messages[0][1]["content"])
        self.assertEqual(len(fake_sandbox.calls), 3)

    async def test_resume_requires_host_validation(self):
        with self.assertRaisesRegex(AgentError, "resume requires"):
            await self.invoke(
                [{"action": "done", "summary": "unused"}],
                [],
                resume=True,
            )

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

    async def test_failed_validation_rejects_done_until_code_is_repaired(self):
        messages = []
        result, fake_sandbox = await self.invoke(
            [
                {"action": "done", "summary": "first attempt"},
                {"action": "done", "summary": "same state"},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "good",
                },
                {"action": "done", "summary": "repaired"},
            ],
            [
                (1, "failure", ""),
                (0, "post-edit pass", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            max_actions=4,
            capture_messages=messages,
        )

        self.assertTrue(result.validated)
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.validation_rounds, 2)
        self.assertEqual(result.validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)
        rejected_done = json.loads(messages[2][-1]["content"])
        self.assertEqual(rejected_done["action"], "done")
        self.assertIn("validation_repair_required", rejected_done["error"])

    async def test_repeated_done_after_failed_validation_stops_early(self):
        with self.assertRaisesRegex(RuntimeError, "AGENT_DONE_WITHOUT_REPAIR"):
            await self.invoke(
                [
                    {"action": "done", "summary": "first"},
                    {"action": "done", "summary": "second"},
                    {"action": "done", "summary": "third"},
                ],
                [(1, "failure", "")],
                allowed_commands=[self.validation_command],
                validation_commands=[self.validation_command],
                max_actions=5,
            )
        self.assertEqual(len(self.last_fake_sandbox.calls), 1)

    async def test_successful_required_test_allows_done_after_failure(self):
        result, fake_sandbox = await self.invoke(
            [
                {"action": "done", "summary": "first attempt"},
                {"action": "run", "command": self.validation_command},
                {"action": "done", "summary": "required test passed"},
            ],
            [(1, "host failure", ""), (0, "model pass", ""), (0, "host pass", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            max_actions=3,
        )

        self.assertTrue(result.validated)
        self.assertEqual(result.validation_rounds, 2)
        self.assertEqual(result.validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)

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
            [
                (1, "first edit failure", ""),
                (0, "repair pass", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            max_actions=4,
        )

        self.assertEqual(result.status, "validated")
        self.assertTrue(result.validated)
        self.assertEqual(result.validation_rounds, 1)
        self.assertEqual(result.validation_failures, 1)
        self.assertEqual(result.post_edit_validation_runs, 2)
        self.assertEqual(result.post_edit_validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)

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
            [
                (0, "model pass", ""),
                (0, "post-edit pass", ""),
                (0, "host pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            max_actions=3,
        )

        self.assertTrue(result.validated)
        self.assertEqual(len(fake_sandbox.calls), 3)
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
            [(0, "post-edit pass", ""), (0, "final pass", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
        )

        self.assertTrue(result.validated)
        self.assertEqual(len(fake_sandbox.calls), 2)

    async def test_edit_validation_failure_is_repaired_before_done(self):
        messages = []
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "candidate",
                },
                {"action": "read", "path": "example.txt"},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "candidate",
                    "new": "good",
                },
                {"action": "done", "summary": "fixed and validated"},
            ],
            [
                (1, "compile error: missing method", "compiler failed"),
                (0, "post-edit pass", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            capture_messages=messages,
        )

        self.assertTrue(result.validated)
        self.assertEqual(result.post_edit_validation_runs, 2)
        self.assertEqual(result.post_edit_validation_failures, 1)
        self.assertEqual(result.validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)
        first_edit_feedback = json.loads(messages[1][-1]["content"])
        self.assertIn("post_edit_validation", first_edit_feedback)
        self.assertIn(
            "compile error",
            first_edit_feedback["post_edit_validation"]["stdout"],
        )

    async def test_failed_edit_validation_limits_discovery_before_repair(self):
        edit_modes = []
        edit_paths = []
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "candidate",
                },
                {"action": "read", "path": "example.txt"},
                {
                    "action": "grep",
                    "query": "candidate",
                    "paths": ["example.txt"],
                },
                {"action": "read", "path": "example.txt"},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "candidate",
                    "new": "good",
                },
                {"action": "done", "summary": "repaired"},
            ],
            [
                (1, "compile failure: wrong API", "compiler failed"),
                (0, "repair passes", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            capture_edit_modes=edit_modes,
            capture_edit_paths=edit_paths,
        )

        self.assertTrue(result.validated)
        self.assertEqual(edit_modes, [False, False, False, False, True, False])
        self.assertEqual(
            edit_paths,
            [None, None, None, None, ["example.txt"], None],
        )
        self.assertEqual(result.post_edit_validation_runs, 2)
        self.assertEqual(result.post_edit_validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)

    async def test_identical_edit_after_failed_validation_is_rejected(self):
        repeated_edit = {
            "action": "replace",
            "path": "example.txt",
            "old": "bad",
            "new": "candidate",
        }
        messages = []
        result, fake_sandbox = await self.invoke(
            [
                repeated_edit,
                repeated_edit,
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "candidate",
                    "new": "good",
                },
                {"action": "done", "summary": "fixed"},
            ],
            [
                (1, "compile error: missing method", "compiler failed"),
                (0, "repaired", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            capture_messages=messages,
        )

        self.assertTrue(result.validated)
        self.assertEqual(result.post_edit_validation_runs, 2)
        self.assertEqual(result.post_edit_validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)
        repeated_action_feedback = json.loads(messages[2][-1]["content"])
        self.assertIn("AGENT_IDENTICAL_FAILED_EDIT", repeated_action_feedback["error"])
        self.assertIn("Do not repeat", repeated_action_feedback["repair_directive"])

    async def test_missing_replace_text_forces_targeted_source_refresh(self):
        repair_read_paths = []
        messages = []
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "bad",
                    "new": "candidate",
                },
                {"action": "read", "path": "example.txt"},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "missing",
                    "new": "wrong",
                },
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "candidate",
                    "new": "wrong",
                },
                {"action": "read", "path": "example.txt"},
                {
                    "action": "replace",
                    "path": "example.txt",
                    "old": "candidate",
                    "new": "good",
                },
                {"action": "done", "summary": "repaired"},
            ],
            [
                (1, "compile error: first edit", "compiler failed"),
                (0, "repair passes", ""),
                (0, "final pass", ""),
            ],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            capture_messages=messages,
            capture_repair_read_paths=repair_read_paths,
        )

        self.assertTrue(result.validated)
        self.assertEqual(repair_read_paths[3], "example.txt", repr(repair_read_paths))
        self.assertEqual(repair_read_paths[4], "example.txt")
        self.assertIn(
            "must read exactly example.txt",
            messages[4][-1]["content"],
        )
        self.assertEqual(result.post_edit_validation_runs, 2)
        self.assertEqual(result.post_edit_validation_failures, 1)
        self.assertEqual(len(fake_sandbox.calls), 3)

    async def test_rejected_write_forces_exact_source_read_before_retry(self):
        repair_read_paths = []
        messages = []
        result, fake_sandbox = await self.invoke(
            [
                {
                    "action": "read",
                    "path": "example.txt",
                    "start_line": 1,
                    "end_line": 1,
                },
                {
                    "action": "write",
                    "path": "example.txt",
                    "content": "stale rewrite\n",
                },
                {
                    "action": "write",
                    "path": "example.txt",
                    "content": "repeat rejected rewrite\n",
                },
                {"action": "read", "path": "example.txt"},
                {
                    "action": "write",
                    "path": "example.txt",
                    "content": "repaired\n",
                },
                {"action": "done", "summary": "repaired"},
            ],
            [(0, "post-edit validation passed", ""), (0, "final validation passed", "")],
            allowed_commands=[self.validation_command],
            validation_commands=[self.validation_command],
            capture_messages=messages,
            capture_repair_read_paths=repair_read_paths,
        )

        self.assertTrue(result.validated)
        self.assertEqual(repair_read_paths[2], "example.txt")
        self.assertEqual(repair_read_paths[3], "example.txt")
        repair_feedback = "\n".join(
            message["content"] for message in messages[3]
        )
        self.assertIn("AGENT_REPAIR_SOURCE_REFRESH_REQUIRED", repair_feedback)
        self.assertIn("must read exactly example.txt", repair_feedback)
        self.assertEqual(fake_sandbox.calls[-1][0][1], "diff")
        self.assertEqual(len(fake_sandbox.calls), 2)

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
        self.assertFalse(fake_sandbox.calls[0][1]["access_policy"].validation)
        self.assertTrue(fake_sandbox.calls[1][1]["access_policy"].validation)

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
                [
                    {"action": "done", "summary": "first"},
                    {
                        "action": "replace",
                        "path": "example.txt",
                        "old": "bad",
                        "new": "attempt one",
                    },
                    {"action": "done", "summary": "second"},
                    {
                        "action": "replace",
                        "path": "example.txt",
                        "old": "attempt one",
                        "new": "attempt two",
                    },
                    {"action": "done", "summary": "third"},
                    {
                        "action": "replace",
                        "path": "example.txt",
                        "old": "attempt two",
                        "new": "attempt three",
                    },
                    {"action": "done", "summary": "fourth"},
                ],
                [
                    (1, "failure one", ""),
                    (0, "edit pass one", ""),
                    (1, "failure two", ""),
                    (0, "edit pass two", ""),
                    (1, "failure three", ""),
                    (0, "edit pass three", ""),
                ],
                allowed_commands=[self.validation_command],
                validation_commands=[self.validation_command],
                max_actions=100,
            )

        self.assertEqual(
            len(self.last_fake_sandbox.calls), 2 * MAX_VALIDATION_ROUNDS - 1
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
            "validation_read_paths": ["src", "tests"],
            "validation_output_paths": [".dart_tool", "build"],
        }
        request.update(updates)
        return request

    def test_json_request_parses_validation_commands_strictly(self):
        request = qwen_agent_cli._parse_json_request(
            json.dumps(self.request()).encode("utf-8")
        )
        self.assertEqual(request["validation_commands"], ["git diff"])
        self.assertEqual(
            request["validation_output_paths"], [".dart_tool", "build"]
        )
        self.assertEqual(request["validation_read_paths"], ["src", "tests"])

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
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "array of strings"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(validation_output_paths="build")).encode(
                    "utf-8"
                )
            )
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "array of strings"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(validation_read_paths="src")).encode(
                    "utf-8"
                )
            )
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "unknown request field"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(unknown=True)).encode("utf-8")
            )

    def test_resume_json_field_is_strict_boolean(self):
        request = qwen_agent_cli._parse_json_request(
            json.dumps(self.request(resume=True)).encode("utf-8")
        )
        self.assertTrue(request["resume"])
        self.assertFalse(
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request()).encode("utf-8")
            )["resume"]
        )
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "resume must be a boolean"):
            qwen_agent_cli._parse_json_request(
                json.dumps(self.request(resume="yes")).encode("utf-8")
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
                "--validation-read-path",
                "src",
                "--validation-read-path",
                "tests",
                "--validation-output-path",
                ".dart_tool",
                "--validation-output-path",
                "build",
                "--resume",
                "--task",
                "verify",
            ]
        )
        self.assertEqual(
            request["validation_commands"], ["git diff", "git status"]
        )
        self.assertEqual(
            request["validation_output_paths"], [".dart_tool", "build"]
        )
        self.assertEqual(request["validation_read_paths"], ["src", "tests"])
        self.assertTrue(request["resume"])

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
            qwen_agent_cli,
            "run_qwen_agent_profile",
            new_callable=mock.AsyncMock,
            return_value=result.as_dict(),
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
            run_agent.await_args.kwargs["validation_commands"], ["git diff"]
        )
        document = json.loads(output.getvalue())
        self.assertEqual(document["status"], "validated")
        self.assertTrue(document["validated"])
        self.assertEqual(document["validation_commands"], 2)
        self.assertEqual(document["validation_rounds"], 1)
        self.assertNotIn("PRIVATE_VALIDATION_OUTPUT", output.getvalue())


@mock.patch("qwen_agent._omlx_api_headers", new=lambda: {})
class AgentTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.endpoint_environment = mock.patch.dict(
            "os.environ",
            {"QWEN_BASE_URL": "http://127.0.0.1:11234/v1"},
        )
        self.endpoint_environment.start()

    def test_token_aggregate_is_unknown_when_any_model_call_omits_usage(self):
        self.assertIsNone(qwen_agent._aggregate_token_count([12, None], 2))

    def tearDown(self):
        self.endpoint_environment.stop()

    async def test_discovery_limit_keeps_history_bounded_and_telemetry_private(self):
        sizes = []
        stderr = io.StringIO()

        async def fake_turn(_client, _base, _model, messages):
            sizes.append(sum(len(item["content"]) for item in messages))
            return AgentGeneration(
                json.dumps(
                    {
                        "action": "grep",
                        "query": f"query-{len(sizes)}",
                        "paths": ["example.txt"],
                    }
                ),
                1,
                0,
                1,
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
                with self.assertRaisesRegex(RuntimeError, "AGENT_DISCOVERY_LIMIT"):
                    await run_qwen_agent(
                        "repeat the read action",
                        directory,
                        ["example.txt"],
                        max_actions=100,
                    )

        self.assertEqual(len(sizes), 23)
        self.assertLessEqual(max(sizes), MAX_AGENT_HISTORY_CHARS)
        self.assertLessEqual(sizes[-1], MAX_AGENT_HISTORY_CHARS)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", stderr.getvalue())
        self.assertIn("history_chars=", stderr.getvalue())

    async def test_baseline_failure_forces_edit_after_bounded_discovery(self):
        requested_edit_mode = []
        actions = [
            {"action": "read", "path": "example.txt"},
            *[
                {
                    "action": "grep",
                    "query": f"needle-{index}",
                    "paths": ["example.txt"],
                }
                for index in range(5)
            ],
            {
                "action": "replace",
                "path": "example.txt",
                "old": "before",
                "new": "after",
            },
            {"action": "done", "summary": "updated the failing source"},
        ]

        async def fake_turn(_client, _base, _model, _messages, **kwargs):
            requested_edit_mode.append(kwargs.get("require_edit", False))
            action = actions[len(requested_edit_mode) - 1]
            return AgentGeneration(json.dumps(action), 1, 0, 1)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.txt").write_text(
                "before\nline\n", encoding="utf-8"
            )
            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fake_turn
            ):
                result = await run_qwen_agent(
                    "repair the implementation",
                    directory,
                    ["example.txt"],
                    validation_failure="baseline test failed",
                    max_actions=10,
                )
            self.assertEqual(
                Path(directory, "example.txt").read_text(encoding="utf-8"),
                "after\nline\n",
            )
        self.assertEqual(requested_edit_mode, [False] * 6 + [True, False])
        self.assertEqual(result.files_changed, ["example.txt"])
        self.assertEqual(result.generation_request_attempts, 8)
        self.assertEqual(result.prefill_memory_retries, 0)

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

    async def test_prefill_memory_error_retries_once_with_lower_output_cap(self):
        request_caps = []

        async def handler(request):
            request_caps.append(json.loads(request.content)["max_tokens"])
            if len(request_caps) == 1:
                return httpx.Response(
                    400,
                    json={"error": {"code": "prefill_memory_exceeded"}},
                )
            return _sse_response(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": json.dumps(
                                    {"action": "done", "summary": "finished"}
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            generation = await _generate_agent_turn(
                client,
                "http://qwen.test/v1",
                DEFAULT_MODEL,
                [{"role": "user", "content": "task"}],
                sleep=lambda _delay: asyncio.sleep(0),
            )

        self.assertEqual(request_caps, [qwen_agent.MAX_OUTPUT_TOKENS, 2048])
        self.assertEqual(generation.generation_request_attempts, 2)
        self.assertEqual(generation.generation_token_cap, 2048)
        self.assertEqual(generation.prefill_memory_retries, 1)
        self.assertEqual(json.loads(generation.text)["action"], "done")

    async def test_prefill_memory_error_stops_after_one_retry(self):
        request_caps = []

        async def handler(request):
            request_caps.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "prefill_memory_exceeded",
                        "estimated_bytes": 79363703680,
                        "limit_bytes": 79054241792,
                    }
                },
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaisesRegex(
                RuntimeError,
                "generation_token_cap=2048 prefill_memory_retries=1",
            ):
                await _generate_agent_turn(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    [{"role": "user", "content": "task"}],
                    sleep=lambda _delay: asyncio.sleep(0),
                )

        self.assertEqual(request_caps, [qwen_agent.MAX_OUTPUT_TOKENS, 2048])

    async def test_failed_prefill_retry_counts_are_logged(self):
        async def fail_generation(*_args, **_kwargs):
            raise RuntimeError(
                "AGENT_GENERATION_HTTP attempts=2 generation_token_cap=2048 "
                "prefill_memory_retries=1 status=400: prefill_memory_exceeded"
            )

        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "example.txt").write_text("source\n", encoding="utf-8")
            with mock.patch(
                "qwen_agent._preflight", new=_fake_preflight
            ), mock.patch(
                "qwen_agent._generate_agent_turn", new=fail_generation
            ), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(RuntimeError, "prefill_memory_exceeded"):
                    await run_qwen_agent(
                        "repair example.txt",
                        directory,
                        ["example.txt"],
                        max_actions=1,
                    )

        self.assertIn("generation_request_attempts=2", stderr.getvalue())
        self.assertIn("prefill_memory_retries=1", stderr.getvalue())
        self.assertIn("generation_token_cap=2048", stderr.getvalue())

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
                "qwen_agent._omlx_api_headers",
                return_value={"Authorization": "Bearer unit-test-key"},
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
        self.assertEqual(
            client_options[0]["headers"]["Authorization"], "Bearer unit-test-key"
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
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 31,
                        "completion_tokens": 7,
                        "total_tokens": 38,
                    },
                },
                headers={
                    "Warning": '199 omlx "response_format not enforced; output is best-effort"'
                },
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
        self.assertEqual(result.prompt_tokens, 31)
        self.assertEqual(result.total_tokens, 38)
        self.assertTrue(result.response_format_warning)

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
                        prompt_tokens=11,
                        total_tokens=15,
                    ),
                    AgentGeneration(
                        '{"action":"done","summary":"replaced text"}',
                        2,
                        0,
                        1,
                        prompt_tokens=13,
                        total_tokens=15,
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
        self.assertEqual(result.prompt_tokens, 24)
        self.assertEqual(result.completion_tokens, 6)
        self.assertEqual(result.total_tokens, 30)
        self.assertEqual(result.model, DEFAULT_MODEL)

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
                qwen_agent_cli,
                "run_qwen_agent_profile",
                new_callable=mock.AsyncMock,
                return_value=result.as_dict(),
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

        self.assertEqual(run_agent.await_args.kwargs["task"], task)

    def test_oversized_task_file_is_rejected_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            task_path = Path(directory, "oversized-task.txt")
            task_path.write_bytes(
                b"x" * (qwen_input_limits.MAX_TASK_BYTES + 1)
            )
            with mock.patch.object(
                qwen_agent_cli,
                "run_qwen_agent_profile",
                new_callable=mock.AsyncMock,
                side_effect=AssertionError("model must not be called"),
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
            qwen_agent_cli,
            "run_qwen_agent_profile",
            new_callable=mock.AsyncMock,
            side_effect=AssertionError,
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
                qwen_agent_cli,
                "run_qwen_agent_profile",
                new_callable=mock.AsyncMock,
                side_effect=AssertionError,
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
