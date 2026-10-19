import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import qwen_agent_cli


class QwenAgentProfileCliTests(unittest.TestCase):
    def test_parse_json_request_defaults_profile_to_non_thinking(self):
        raw = json.dumps(
            {
                "task": "example task",
                "sandbox": "/tmp/sandbox",
                "allow": ["/tmp/sandbox"],
            }
        ).encode("utf-8")
        request = qwen_agent_cli._parse_json_request(raw)
        self.assertEqual(request["profile"], "non-thinking")

    def test_parse_json_request_accepts_thinking_profile(self):
        raw = json.dumps(
            {
                "task": "example task",
                "sandbox": "/tmp/sandbox",
                "allow": ["/tmp/sandbox"],
                "profile": "thinking",
            }
        ).encode("utf-8")
        request = qwen_agent_cli._parse_json_request(raw)
        self.assertEqual(request["profile"], "thinking")

    def test_parse_json_request_rejects_unknown_profile(self):
        raw = json.dumps(
            {
                "task": "example task",
                "sandbox": "/tmp/sandbox",
                "allow": ["/tmp/sandbox"],
                "profile": "turbo",
            }
        ).encode("utf-8")
        with self.assertRaisesRegex(qwen_agent_cli.RequestError, "unknown profile"):
            qwen_agent_cli._parse_json_request(raw)

    def test_request_from_flags_defaults_profile_to_non_thinking(self):
        arguments = [
            "--sandbox",
            "/tmp/sandbox",
            "--allow",
            "/tmp/sandbox",
            "--task",
            "example task",
        ]
        request = qwen_agent_cli._request_from_flags(arguments)
        self.assertEqual(request["profile"], "non-thinking")

    def test_request_from_flags_accepts_thinking_profile(self):
        arguments = [
            "--sandbox",
            "/tmp/sandbox",
            "--allow",
            "/tmp/sandbox",
            "--task",
            "example task",
            "--profile",
            "thinking",
        ]
        request = qwen_agent_cli._request_from_flags(arguments)
        self.assertEqual(request["profile"], "thinking")

    def test_run_forwards_request_fields_and_returns_mocked_dict(self):
        expected = {"status": "completed", "profile": "thinking"}
        arguments = [
            "--sandbox",
            "/tmp/sandbox",
            "--allow",
            "/tmp/sandbox",
            "--task",
            "example task",
            "--profile",
            "thinking",
            "--read-only",
            "/tmp/readonly",
            "--allow-command",
            "python",
            "--validation-command",
            "python -m unittest test_example",
            "--max-actions",
            "7",
        ]
        with patch.object(
            qwen_agent_cli,
            "run_qwen_agent_profile",
            new=AsyncMock(return_value=expected),
        ) as mock_run:
            result = asyncio.run(qwen_agent_cli._run(arguments))

        self.assertIs(result, expected)
        mock_run.assert_awaited_once()
        mock_call = mock_run.await_args
        kwargs = mock_call.kwargs
        self.assertEqual(kwargs["profile"], "thinking")
        self.assertEqual(kwargs["task"], "example task")
        self.assertEqual(kwargs["sandbox"], "/tmp/sandbox")
        self.assertEqual(kwargs["allow"], ["/tmp/sandbox"])
        self.assertEqual(kwargs["read_only"], ["/tmp/readonly"])
        self.assertEqual(kwargs["allowed_commands"], ["python"])
        self.assertEqual(kwargs["validation_commands"], ["python -m unittest test_example"])
        self.assertEqual(kwargs["max_actions"], 7)


if __name__ == "__main__":
    unittest.main()
