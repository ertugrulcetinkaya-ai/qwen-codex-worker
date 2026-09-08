import json
import os
import stat
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from qwen_input_limits import (
    MAX_MODEL_INPUT_CHARS,
    MAX_PATH_CHARS,
    MAX_REQUEST_BYTES,
    MAX_SOURCE_FILES,
)
from qwen_worker_server import (
    DEFAULT_MODEL,
    MAX_SOURCE_CHARS,
    MAX_TASK_CHARS,
    MAX_TEST_OUTPUT_CHARS,
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


class _ProxyHandler(BaseHTTPRequestHandler):
    def _reject(self):
        self.server.request_count += 1
        body = b"proxy must not receive model traffic"
        self.send_response(502)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_CONNECT = _reject
    do_GET = _reject
    do_POST = _reject

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


class MockProxy:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ProxyHandler)
        self.server.request_count = 0
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
        return f"http://{host}:{port}"


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

    def invoke(
        self, request_bytes=None, arguments=(), base_url=None, environment_overrides=None
    ):
        environment = os.environ.copy()
        if base_url is not None:
            environment["QWEN_BASE_URL"] = base_url
        environment.update(environment_overrides or {})
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
        prompt = qwen.server.last_request["messages"][1]["content"]
        self.assertIn("alpha\n", prompt)
        self.assertIn("example.txt", prompt)
        self.assertNotIn(str(self.repo_root), prompt)

    def test_unauthorized_structurally_valid_patch_fails_without_mutation(self):
        diff = "--- a/other.txt\n+++ b/other.txt\n@@ -1 +1 @@\n-old\n+new\n"
        original = self.source.read_bytes()
        with MockQwenServer(diff) as qwen:
            result = self.invoke(
                json.dumps(self.request()).encode("utf-8"), base_url=qwen.base_url
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"PATCH_TARGET_NOT_ALLOWED", result.stderr)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(qwen.server.post_count, 1)

    def test_local_model_traffic_bypasses_proxy_environment(self):
        diff = "--- example.txt\n+++ example.txt\n@@ -1 +1 @@\n-alpha\n+beta\n"
        with MockQwenServer(diff) as qwen, MockProxy() as proxy:
            proxy_environment = {
                name: proxy.base_url
                for name in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                )
            }
            proxy_environment.update({"NO_PROXY": "", "no_proxy": ""})
            result = self.invoke(
                json.dumps(self.request()).encode("utf-8"),
                base_url=qwen.base_url,
                environment_overrides=proxy_environment,
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, diff.encode("utf-8"))
        self.assertEqual(qwen.server.get_count, 1)
        self.assertEqual(qwen.server.post_count, 1)
        self.assertEqual(proxy.server.request_count, 0)

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

    def test_oversized_request_file_is_rejected_before_model(self):
        request_path = self.repo_root / "oversized-request.json"
        request_path.write_bytes(b"x" * (MAX_REQUEST_BYTES + 1))
        result = self.invoke(arguments=("--request-file", str(request_path)))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"REQUEST_TOO_LARGE", result.stderr)

    def test_oversized_stdin_is_rejected_before_model(self):
        result = self.invoke(
            b"x" * (MAX_REQUEST_BYTES + 1),
            environment_overrides={"QWEN_BASE_URL": "http://127.0.0.1:1/v1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"")
        self.assertIn(b"REQUEST_TOO_LARGE", result.stderr)

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

    def test_source_path_length_is_enforced_before_filesystem_access(self):
        with self.assertRaisesRegex(ValueError, "PATH_TOO_LONG"):
            _read_supplied_files(str(self.root), ["x" * (MAX_PATH_CHARS + 1)])


class PatchRequestBudgetTests(unittest.TestCase):
    @staticmethod
    def request(**updates):
        request = {
            "task": "change the supplied file",
            "repo_root": "/tmp/repo",
            "files": ["example.txt"],
        }
        request.update(updates)
        return request

    def test_parser_rejects_global_request_limit_before_json(self):
        raw = b"{" + b"x" * MAX_REQUEST_BYTES
        with self.assertRaisesRegex(Exception, "REQUEST_TOO_LARGE"):
            from qwen_patch_cli import _parse_request

            _parse_request(raw)

    def test_parser_rejects_task_test_output_and_file_cardinality_limits(self):
        from qwen_patch_cli import _parse_request

        cases = (
            ("task", "x" * (MAX_TASK_CHARS + 1), "TASK_TOO_LARGE"),
            (
                "test_output",
                "x" * (MAX_TEST_OUTPUT_CHARS + 1),
                "TEST_OUTPUT_TOO_LARGE",
            ),
            (
                "files",
                ["file.txt"] * (MAX_SOURCE_FILES + 1),
                "TOO_MANY_FILES",
            ),
        )
        for field, value, code in cases:
            with self.subTest(field=field), self.assertRaisesRegex(Exception, code):
                _parse_request(json.dumps(self.request(**{field: value})).encode())

    def test_exact_request_boundary_and_invalid_utf8(self):
        from qwen_patch_cli import _parse_request

        raw = json.dumps(self.request(), separators=(",", ":")).encode()
        raw += b" " * (MAX_REQUEST_BYTES - len(raw))
        self.assertEqual(len(raw), MAX_REQUEST_BYTES)
        self.assertEqual(_parse_request(raw)["files"], ["example.txt"])
        with self.assertRaisesRegex(Exception, "UTF-8"):
            _parse_request(b'{"task":"\xff"}')

    def test_combined_patch_prompt_budget_rejects_before_model(self):
        from qwen_worker_server import _build_prompt

        with self.assertRaisesRegex(ValueError, "MODEL_INPUT_TOO_LARGE"):
            _build_prompt(
                "t" * MAX_TASK_CHARS,
                Path("/tmp"),
                [("text.txt", "s" * MAX_SOURCE_CHARS)],
                "o" * MAX_TEST_OUTPUT_CHARS,
            )
        self.assertEqual(
            MAX_TASK_CHARS + MAX_SOURCE_CHARS + MAX_TEST_OUTPUT_CHARS,
            MAX_MODEL_INPUT_CHARS,
        )


if __name__ == "__main__":
    unittest.main()
