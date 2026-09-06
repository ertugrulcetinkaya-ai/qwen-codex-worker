import json
import os
import stat
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qwen_worker_server import (
    DEFAULT_MODEL,
    MAX_SOURCE_CHARS,
    _read_supplied_files,
)


PROJECT_ROOT = Path(__file__).resolve().parent
QWEN_PATCH = PROJECT_ROOT / "qwen-patch"


class _QwenHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.get_count += 1
        body = json.dumps({"data": [{"id": self.server.model}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.server.post_count += 1
        length = int(self.headers["Content-Length"])
        self.server.last_request = json.loads(self.rfile.read(length))
        if self.server.response_status == 200:
            event = {"choices": [{"delta": {"content": self.server.final_text}, "finish_reason": "stop"}]}
            usage = {"choices": [], "usage": {"completion_tokens": 24}}
            body = (
                f"data: {json.dumps(event)}\n\n"
                f"data: {json.dumps(usage)}\n\n"
                "data: [DONE]\n\n"
            ).encode("utf-8")
            content_type = "text/event-stream"
        else:
            body = self.server.response_body
            content_type = "application/json"
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class MockQwenServer:
    def __init__(self, final_text=None, status=200, raw_body=None, model=DEFAULT_MODEL):
        if raw_body is None:
            raw_body = b""
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _QwenHandler)
        self.server.response_status = status
        self.server.response_body = raw_body
        self.server.final_text = final_text
        self.server.model = model
        self.server.last_request = None
        self.server.get_count = 0
        self.server.post_count = 0
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"


class CliIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temporary_directory.name)
        self.source = self.repo_root / "example.txt"
        self.source.write_text("alpha\n", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def request(self):
        return {
            "task": "Change alpha to beta",
            "repo_root": str(self.repo_root),
            "files": ["example.txt"],
        }

    def invoke(self, request_bytes=None, arguments=(), base_url=None):
        environment = os.environ.copy()
        if base_url is not None:
            environment["QWEN_BASE_URL"] = base_url
        return subprocess.run(
            [str(QWEN_PATCH), *arguments],
            input=request_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            cwd=PROJECT_ROOT,
            check=False,
        )

    def test_valid_stdin_json_preserves_output_and_does_not_write(self):
        diff = "--- example.txt\n+++ example.txt\n@@ -1 +1 @@\n-alpha\n+beta\n"
        original_mode = stat.S_IMODE(self.source.stat().st_mode)
        with MockQwenServer(diff) as qwen:
            result = self.invoke(
                json.dumps(self.request()).encode("utf-8"), base_url=qwen.base_url
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, diff.encode("utf-8"))
        self.assertEqual(result.stderr, b"")
        self.assertEqual(self.source.read_bytes(), b"alpha\n")
        self.assertEqual(stat.S_IMODE(self.source.stat().st_mode), original_mode)
        self.assertEqual(qwen.server.get_count, 1)
        self.assertEqual(qwen.server.post_count, 1)
        self.assertEqual(qwen.server.last_request["model"], DEFAULT_MODEL)
        self.assertIs(qwen.server.last_request["stream"], True)
        self.assertIn("alpha\n", qwen.server.last_request["messages"][1]["content"])

    def test_request_file(self):
        response = "NEED_FILES:\nother.txt"
        request_path = self.repo_root / "request.json"
        request_path.write_text(json.dumps(self.request()), encoding="utf-8")
        with MockQwenServer(response) as qwen:
            result = self.invoke(
                arguments=("--request-file", str(request_path)),
                base_url=qwen.base_url,
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, f"{response}\n".encode("utf-8"))
        self.assertEqual(result.stderr, b"")

    def test_missing_terminal_newline_is_canonicalized_for_git(self):
        diff = "--- example.txt\n+++ example.txt\n@@ -1 +1 @@\n-alpha\n+beta"
        with MockQwenServer(diff) as qwen:
            result = self.invoke(
                json.dumps(self.request()).encode("utf-8"), base_url=qwen.base_url
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, f"{diff}\n".encode("utf-8"))
        self.assertEqual(result.stderr, b"")

    def test_debug_telemetry_uses_stderr_without_polluting_stdout(self):
        diff = "--- example.txt\n+++ example.txt\n@@ -1 +1 @@\n-alpha\n+beta\n"
        environment = os.environ.copy()
        environment["QWEN_DEBUG"] = "1"
        with MockQwenServer(diff) as qwen:
            environment["QWEN_BASE_URL"] = qwen.base_url
            result = subprocess.run(
                [str(QWEN_PATCH)],
                input=json.dumps(self.request()).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                cwd=PROJECT_ROOT,
                check=False,
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, diff.encode("utf-8"))
        diagnostic = result.stderr.decode()
        self.assertIn("finish_reason='stop'", diagnostic)
        self.assertIn("completion_tokens=24", diagnostic)
        self.assertIn("reasoning_content_chars=0", diagnostic)
        self.assertIn(f"final_content_chars={len(diff)}", diagnostic)
        self.assertIn("elapsed_ms=", diagnostic)
        self.assertIn("retry_count=0", diagnostic)

    def test_malformed_json_has_only_stderr_and_nonzero_exit(self):
        result = self.invoke(b'{"task":')

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertRegex(result.stderr.decode(), r"^qwen-patch: malformed JSON")

    def test_missing_required_field(self):
        request = self.request()
        del request["files"]
        result = self.invoke(json.dumps(request).encode("utf-8"))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertEqual(result.stderr, b"qwen-patch: missing required field: files\n")

    def test_worker_error_propagates_to_stderr_and_nonzero_exit(self):
        with MockQwenServer(
            status=503, raw_body=b'{"error":{"message":"model unavailable"}}'
        ) as qwen:
            result = self.invoke(
                json.dumps(self.request()).encode("utf-8"), base_url=qwen.base_url
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertRegex(
            result.stderr.decode(), r"GENERATION_HTTP.*status=503.*model unavailable"
        )
        self.assertEqual(qwen.server.post_count, 1)

    def test_preflight_failure_leaves_stdout_empty_and_uses_stderr(self):
        with MockQwenServer("unused", model="wrong-model") as qwen:
            result = self.invoke(
                json.dumps(self.request()).encode("utf-8"), base_url=qwen.base_url
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertRegex(result.stderr.decode(), r"^qwen-patch: PREFLIGHT_MODEL")
        self.assertEqual(qwen.server.post_count, 0)

    def test_path_traversal_guard_is_enforced_before_model_call(self):
        outside = self.repo_root.parent / "outside-qwen-cli-test.txt"
        outside.write_text("secret\n", encoding="utf-8")
        self.addCleanup(outside.unlink, missing_ok=True)
        request = self.request()
        request["files"] = [f"../{outside.name}"]

        result = self.invoke(json.dumps(request).encode("utf-8"))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"file path escapes repo_root", result.stderr)


class SharedFileSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary_directory.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        (self.root / "text.txt").write_text("hello\n", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_root_is_resolved_absolutely(self):
        root, files = _read_supplied_files(str(self.root / "."), ["text.txt"])
        self.assertEqual(root, self.root.resolve())
        self.assertEqual(files, [("text.txt", "hello\n")])

    def test_absolute_file_and_traversal_are_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        for path in (str(outside), "../outside.txt"):
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "repository-relative|escapes repo_root"
            ):
                _read_supplied_files(str(self.root), [path])

    def test_escaping_symlink_is_rejected(self):
        outside = self.base / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        (self.root / "link.txt").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes repo_root"):
            _read_supplied_files(str(self.root), ["link.txt"])

    def test_binary_and_non_utf8_files_are_rejected(self):
        (self.root / "nul.bin").write_bytes(b"a\x00b")
        (self.root / "invalid.txt").write_bytes(b"\xff")
        for path in ("nul.bin", "invalid.txt"):
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "binary|non-UTF-8"
            ):
                _read_supplied_files(str(self.root), [path])

    def test_file_count_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "at most 24"):
            _read_supplied_files(str(self.root), ["text.txt"] * 25)

    def test_source_character_limit_is_enforced(self):
        (self.root / "large.txt").write_text(
            "x" * (MAX_SOURCE_CHARS + 1), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "character limit"):
            _read_supplied_files(str(self.root), ["large.txt"])


if __name__ == "__main__":
    unittest.main()
