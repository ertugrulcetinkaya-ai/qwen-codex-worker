import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import qwen_worker_server


class QwenAgentProfileMcpTest(unittest.IsolatedAsyncioTestCase):
    def make_sandbox(self):
        parent = tempfile.TemporaryDirectory()
        path = Path(parent.name) / "task"
        path.mkdir()
        return parent, path

    async def test_forwards_thinking_profile_and_returns_dict(self):
        parent, sandbox = self.make_sandbox()
        expected = {"validated": True, "profile": "thinking"}
        try:
            with patch("qwen_agent_profiles.run_qwen_agent_profile", new_callable=AsyncMock) as run:
                run.return_value = expected
                result = await qwen_worker_server.qwen_agent(
                    task="small task", sandbox=str(sandbox), allow=["file.py"], profile="thinking"
                )
            self.assertEqual(result, expected)
            self.assertEqual(run.await_args.kwargs["profile"], "thinking")
            self.assertEqual(run.await_args.kwargs["task"], "small task")
            self.assertEqual(run.await_args.kwargs["allow"], ["file.py"])
            self.assertEqual(Path(run.await_args.kwargs["sandbox"]), sandbox.resolve())
        finally:
            parent.cleanup()

    async def test_default_profile_is_non_thinking(self):
        parent, sandbox = self.make_sandbox()
        expected = {"validated": True, "profile": "non-thinking"}
        try:
            with patch("qwen_agent_profiles.run_qwen_agent_profile", new_callable=AsyncMock) as run:
                run.return_value = expected
                result = await qwen_worker_server.qwen_agent(
                    task="small task", sandbox=str(sandbox), allow=["file.py"]
                )
            self.assertEqual(result, expected)
            self.assertEqual(run.await_args.kwargs["profile"], "non-thinking")
        finally:
            parent.cleanup()

    async def test_invalid_profile_is_rejected_before_helper(self):
        parent, sandbox = self.make_sandbox()
        try:
            with patch("qwen_agent_profiles.run_qwen_agent_profile", new_callable=AsyncMock) as run:
                with self.assertRaises(ValueError):
                    await qwen_worker_server.qwen_agent(
                        task="small task", sandbox=str(sandbox), allow=["file.py"], profile="bogus"
                    )
            run.assert_not_awaited()
        finally:
            parent.cleanup()


if __name__ == "__main__":
    unittest.main()
