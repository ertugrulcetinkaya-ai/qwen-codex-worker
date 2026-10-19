"""Tests for qwen_agent_profiles adapter."""

import asyncio
import json
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Provide a minimal qwen_agent stub so the adapter module can import.
_qa = types.ModuleType("qwen_agent")


class _Generation:
    def __init__(self, text="", reasoning_content=None):
        self.text = text
        self.reasoning_content = reasoning_content


_qa._Generation = _Generation


async def _parse_agent_stream(response, *, started, attempts, http_status):
    content_parts = []
    reasoning_parts = []
    async for chunk in response.aiter_bytes():
        line = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else chunk
        for piece in line.split("\n"):
            piece = piece.strip()
            if not piece.startswith("data:"):
                continue
            payload = piece[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if "content" in delta:
                content_parts.append(delta["content"])
            if "reasoning_content" in delta:
                reasoning_parts.append(delta["reasoning_content"])
    return _Generation(text="".join(content_parts), reasoning_content="".join(reasoning_parts))


def _parse_action(text):
    return json.loads(text)


_qa._parse_agent_stream = _parse_agent_stream
_qa.parse_action = _parse_action
_qa.run_qwen_agent = AsyncMock()

sys.modules.setdefault("qwen_agent", _qa)

import qwen_agent_profiles as profiles  # noqa: E402


class TestNormalizeProfile(unittest.TestCase):
    def test_canonical_names(self):
        self.assertEqual(profiles.normalize_profile("non-thinking"), "non-thinking")
        self.assertEqual(profiles.normalize_profile("thinking"), "thinking")

    def test_aliases(self):
        self.assertEqual(profiles.normalize_profile("A"), "non-thinking")
        self.assertEqual(profiles.normalize_profile("a"), "non-thinking")
        self.assertEqual(profiles.normalize_profile("B"), "thinking")
        self.assertEqual(profiles.normalize_profile("b"), "thinking")

    def test_default(self):
        self.assertEqual(profiles.normalize_profile(None), "non-thinking")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            profiles.normalize_profile("unknown")
        with self.assertRaises(ValueError):
            profiles.normalize_profile(123)


class TestApplyProfilePayload(unittest.TestCase):
    def test_non_thinking_payload(self):
        req = {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.5,
            "top_k": 5,
            "presence_penalty": 2.0,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        out = profiles.apply_profile_payload(req, "non-thinking")
        self.assertEqual(out["temperature"], 0.7)
        self.assertEqual(out["top_p"], 0.8)
        self.assertEqual(out["top_k"], 20)
        self.assertEqual(out["presence_penalty"], 1.5)
        self.assertEqual(out["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(out["model"], "qwen")
        self.assertEqual(out["max_tokens"], 4096)
        self.assertEqual(out["messages"], req["messages"])

    def test_thinking_payload(self):
        req = {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
        }
        out = profiles.apply_profile_payload(req, "thinking")
        self.assertEqual(out["temperature"], 1.0)
        self.assertEqual(out["top_p"], 0.95)
        self.assertEqual(out["top_k"], 20)
        self.assertEqual(out["presence_penalty"], 0)
        self.assertEqual(
            out["chat_template_kwargs"],
            {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "medium",
            },
        )
        self.assertEqual(out["max_tokens"], 4096)

    def test_preserves_extra_fields(self):
        req = {
            "model": "qwen",
            "messages": [],
            "custom_field": "keep_me",
            "max_tokens": 4096,
        }
        out = profiles.apply_profile_payload(req, "non-thinking")
        self.assertEqual(out["custom_field"], "keep_me")

    def test_does_not_mutate_original(self):
        req = {"model": "qwen", "messages": [], "temperature": 0.1}
        profiles.apply_profile_payload(req, "non-thinking")
        self.assertEqual(req["temperature"], 0.1)

    def test_default_profile(self):
        req = {"model": "qwen", "messages": []}
        out = profiles.apply_profile_payload(req)
        self.assertEqual(out["temperature"], 0.7)


class TestRunProfileRejectsUnknown(unittest.TestCase):
    def test_unknown_rejected_before_network(self):
        with patch.object(profiles.qwen_agent, "run_qwen_agent", new_callable=AsyncMock) as mock_run:
            with self.assertRaises(ValueError):
                asyncio.run(
                    profiles.run_qwen_agent_profile(profile="bogus")
                )
            mock_run.assert_not_called()


class TestRunProfileMetrics(unittest.TestCase):
    def _make_result(self, as_dict_value=None):
        result = MagicMock()
        result.as_dict.return_value = as_dict_value or {"output": "ok"}
        return result

    def test_basic_run_returns_profile_and_metrics(self):
        result = self._make_result()
        with patch.object(profiles.qwen_agent, "run_qwen_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            out = asyncio.run(
                profiles.run_qwen_agent_profile(profile="non-thinking")
            )
        self.assertEqual(out["profile"], "non-thinking")
        self.assertIn("first_token_ms", out)
        self.assertIn("reasoning_tokens", out)
        self.assertIn("protocol_errors", out)
        self.assertEqual(out["output"], "ok")

    def test_reasoning_tokens_null_when_no_usage(self):
        result = self._make_result()
        with patch.object(profiles.qwen_agent, "run_qwen_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            out = asyncio.run(
                profiles.run_qwen_agent_profile(profile="thinking")
            )
        self.assertIsNone(out["reasoning_tokens"])

    def test_agent_protocol_error_total_is_not_overwritten_by_parse_metric(self):
        result = self._make_result({"protocol_errors": 2})
        with patch.object(profiles.qwen_agent, "run_qwen_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = result
            out = asyncio.run(
                profiles.run_qwen_agent_profile(profile="non-thinking")
            )
        self.assertEqual(out["protocol_errors"], 2)
        self.assertEqual(out["response_parse_errors"], 0)

    def test_one_sandbox_cannot_mix_sampling_profiles(self):
        result = self._make_result()
        with tempfile.TemporaryDirectory() as directory:
            sandbox = Path(directory) / "sandbox"
            sandbox.mkdir()
            database = Path(directory) / "state" / "profiles.sqlite3"
            profiles._PROFILE_SANDBOX_BINDINGS.clear()
            with patch.object(profiles, "_PROFILE_BINDINGS_DB_PATH", database):
                with patch.object(
                    profiles.qwen_agent,
                    "run_qwen_agent",
                    new_callable=AsyncMock,
                ) as mock_run:
                    mock_run.return_value = result
                    asyncio.run(
                        profiles.run_qwen_agent_profile(
                            profile="non-thinking", sandbox=sandbox
                        )
                    )
                    with self.assertRaisesRegex(
                        ValueError, "clean, separate sandbox"
                    ):
                        asyncio.run(
                            profiles.run_qwen_agent_profile(
                                profile="thinking", sandbox=sandbox
                            )
                        )
                    mock_run.assert_awaited_once()
            profiles._PROFILE_SANDBOX_BINDINGS.clear()


class TestProfileSandboxIsolation(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.sandbox = self.directory / "sandbox"
        self.sandbox.mkdir()
        self.state_db = self.directory / "private-state" / "profiles.sqlite3"
        self.binding_path = patch.object(
            profiles, "_PROFILE_BINDINGS_DB_PATH", self.state_db
        )
        self.binding_path.start()
        profiles._PROFILE_SANDBOX_BINDINGS.clear()
        self.sentinel = self.sandbox / "sentinel.txt"
        self.sentinel.write_bytes(b"unchanged sentinel\n")

    def tearDown(self):
        profiles._PROFILE_SANDBOX_BINDINGS.clear()
        self.binding_path.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def _result():
        result = MagicMock()
        result.as_dict.return_value = {"validated": True}
        return result

    def _invoke(self, profile, runner=None, sandbox=None):
        if runner is None:
            runner = AsyncMock(return_value=self._result())
        with patch.object(profiles.qwen_agent, "run_qwen_agent", new=runner):
            return asyncio.run(
                profiles.run_qwen_agent_profile(
                    profile=profile,
                    sandbox=str(sandbox if sandbox is not None else self.sandbox),
                )
            )

    def test_concurrent_different_profiles_are_rejected(self):
        async def run_concurrent_calls():
            started = asyncio.Event()
            release = asyncio.Event()
            result = self._result()

            async def hold_first_call(**_kwargs):
                started.set()
                await release.wait()
                return result

            with patch.object(
                profiles.qwen_agent,
                "run_qwen_agent",
                new=hold_first_call,
            ):
                first = asyncio.create_task(
                    profiles.run_qwen_agent_profile(
                        profile="non-thinking", sandbox=str(self.sandbox)
                    )
                )
                await started.wait()
                with self.assertRaisesRegex(
                    ValueError, "clean, separate sandbox"
                ):
                    await profiles.run_qwen_agent_profile(
                        profile="thinking", sandbox=str(self.sandbox)
                    )
                release.set()
                await first

        asyncio.run(run_concurrent_calls())
        self.assertEqual(self.sentinel.read_bytes(), b"unchanged sentinel\n")

    def test_failed_task_keeps_profile_binding_and_preserves_files(self):
        async def fail_run(**_kwargs):
            raise RuntimeError("expected test failure")

        with self.assertRaisesRegex(RuntimeError, "expected test failure"):
            self._invoke("non-thinking", runner=fail_run)
        next_run = AsyncMock(return_value=self._result())
        with self.assertRaisesRegex(ValueError, "clean, separate sandbox"):
            self._invoke("thinking", runner=next_run)
        next_run.assert_not_awaited()
        self.assertEqual(self.sentinel.read_bytes(), b"unchanged sentinel\n")

    def test_reopened_sandbox_keeps_profile_binding_and_preserves_files(self):
        self._invoke("non-thinking")
        script = (
            "import sys; from pathlib import Path; "
            "import qwen_agent_profiles as p; "
            "p._PROFILE_BINDINGS_DB_PATH = Path(sys.argv[1]); "
            "p._claim_profile_sandbox('thinking', sys.argv[2])"
        )
        reopened = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(self.state_db),
                str(self.sandbox),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertNotEqual(reopened.returncode, 0)
        self.assertIn("already assigned to the non-thinking profile", reopened.stderr)
        self.assertEqual(self.sentinel.read_bytes(), b"unchanged sentinel\n")

    def test_symlink_and_alternate_paths_share_the_binding(self):
        alias = self.directory / "alias"
        alias.symlink_to(self.sandbox, target_is_directory=True)
        self._invoke("non-thinking")
        next_run = AsyncMock(return_value=self._result())
        with self.assertRaisesRegex(ValueError, "clean, separate sandbox"):
            self._invoke("thinking", runner=next_run, sandbox=alias / ".")
        next_run.assert_not_awaited()
        self.assertEqual(self.sentinel.read_bytes(), b"unchanged sentinel\n")


class TestParseActionProtocolErrors(unittest.TestCase):
    def setUp(self):
        profiles._install_wrappers()

    def tearDown(self):
        token = profiles._metrics_var.set(None)
        profiles._metrics_var.reset(token)

    def test_parse_action_failure_increments(self):
        metrics = {
            "start_time": time.monotonic(),
            "first_token_ms": None,
            "reasoning_valid": True,
            "reasoning_sum": 0,
            "reasoning_call_count": 0,
            "protocol_errors": 0,
        }
        token = profiles._metrics_var.set(metrics)
        try:
            with self.assertRaises(Exception):
                profiles.qwen_agent.parse_action("invalid json {{{")
            self.assertEqual(metrics["protocol_errors"], 1)
        finally:
            profiles._metrics_var.reset(token)


class TestReasoningNotFedToParser(unittest.TestCase):
    """reasoning_content must never enter parse_action; only delta.content."""

    def setUp(self):
        profiles._install_wrappers()

    def tearDown(self):
        token = profiles._metrics_var.set(None)
        profiles._metrics_var.reset(token)

    def test_reasoning_text_not_parsed(self):
        reasoning_text = "This is just thinking, not JSON {{{"
        action_json = json.dumps(
            {"action": "grep", "query": "hello", "paths": ["src"]}
        )

        reasoning_delta = {"choices": [{"delta": {"reasoning_content": reasoning_text}}]}
        content_delta = {"choices": [{"delta": {"content": action_json}}]}

        events = [
            ("data: " + json.dumps(reasoning_delta) + "\n\n").encode(),
            ("data: " + json.dumps(content_delta) + "\n\n").encode(),
            b'data: [DONE]\n\n',
        ]

        async def fake_aiter_bytes():
            for e in events:
                yield e

        response = MagicMock()
        response.aiter_bytes = fake_aiter_bytes

        metrics = {
            "start_time": time.monotonic(),
            "first_token_ms": None,
            "reasoning_valid": True,
            "reasoning_sum": 0,
            "reasoning_call_count": 0,
            "protocol_errors": 0,
        }

        token = profiles._metrics_var.set(metrics)
        try:
            generation = asyncio.run(
                profiles.qwen_agent._parse_agent_stream(
                    response,
                    started=time.monotonic(),
                    attempts=1,
                    http_status=200,
                )
            )
        finally:
            profiles._metrics_var.reset(token)

        self.assertEqual(generation.text, action_json)
        self.assertNotIn(reasoning_text, generation.text)

        parsed = profiles.qwen_agent.parse_action(generation.text)
        self.assertEqual(
            parsed, {"action": "grep", "query": "hello", "paths": ["src"]}
        )


if __name__ == "__main__":
    unittest.main()
