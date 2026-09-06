import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from qwen_worker_server import (
    DEFAULT_MODEL,
    MAX_CONNECT_ATTEMPTS,
    MAX_OUTPUT_TOKENS,
    PRESENCE_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    _generate_stream,
    _parse_chat_completion,
    _parse_chat_completion_response,
    _preflight,
    _validate_worker_output,
    qwen_patch,
    run_qwen_worker,
)


class ChatCompletionParserTests(unittest.TestCase):
    def test_normal_message_content(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "--- a/example.txt\n+++ b/example.txt\n",
                        "reasoning_content": "private metadata",
                    },
                }
            ],
            "usage": {"completion_tokens": 12},
        }

        self.assertEqual(
            _parse_chat_completion(payload),
            "--- a/example.txt\n+++ b/example.txt\n",
        )

    def test_empty_content_with_reasoning_is_an_error(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "internal work"},
                }
            ],
            "usage": {"completion_tokens": 99},
        }

        with self.assertRaisesRegex(
            RuntimeError,
            r"exceeded the output token limit",
        ):
            _parse_chat_completion(payload)

    def test_length_finish_with_nonempty_content_is_an_error(self):
        payload = {
            "choices": [
                {"finish_reason": "length", "message": {"content": "partial diff"}}
            ]
        }

        with self.assertRaisesRegex(RuntimeError, "exceeded the output token limit"):
            _parse_chat_completion(payload)

    def test_missing_choices_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "missing a non-empty choices array"):
            _parse_chat_completion({"usage": {"completion_tokens": 0}})

    def test_malformed_message_shape_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "missing a message object"):
            _parse_chat_completion({"choices": [{"message": []}]})

    def test_http_failure_includes_status_and_body(self):
        response = httpx.Response(
            503,
            text='{"error":{"message":"model unavailable"}}',
        )

        with self.assertRaisesRegex(
            RuntimeError, r"HTTP error 503.*model unavailable"
        ):
            _parse_chat_completion_response(response)

    def test_malformed_json_is_an_error(self):
        response = httpx.Response(200, text="not-json")

        with self.assertRaisesRegex(RuntimeError, r"malformed JSON .*not-json"):
            _parse_chat_completion_response(response)

    def test_content_parts_and_message_text_fallbacks(self):
        parts = {
            "choices": [
                {"message": {"content": [{"type": "text", "text": "TEST_OK"}]}}
            ]
        }
        alternate = {
            "choices": [{"message": {"content": "", "output_text": "TEST_OK"}}]
        }

        self.assertEqual(_parse_chat_completion(parts), "TEST_OK")
        self.assertEqual(_parse_chat_completion(alternate), "TEST_OK")


async def _no_sleep(_delay):
    return None


def _sse_response(*events, status=200):
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"
    return httpx.Response(
        status,
        headers={"content-type": "text/event-stream"},
        content=body.encode("utf-8"),
    )


class _FailingStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadError("stream broke")


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_succeeds_with_exact_expected_model(self):
        async def handler(request):
            self.assertEqual(request.url.path, "/v1/models")
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            attempts = await _preflight(
                client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
            )

        self.assertEqual(attempts, 1)

    async def test_preflight_expected_model_missing(self):
        async def handler(_request):
            return httpx.Response(200, json={"data": [{"id": "other-model"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_MODEL"):
                await _preflight(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
                )

    async def test_preflight_malformed_json(self):
        async def handler(_request):
            return httpx.Response(200, text="not-json")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_MALFORMED"):
                await _preflight(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
                )

    async def test_preflight_http_failure_is_distinct(self):
        async def handler(_request):
            return httpx.Response(503, text="model service unavailable")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_HTTP.*status=503"):
                await _preflight(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
                )

    async def test_preflight_connection_failure_then_successful_retry(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            attempts = await _preflight(
                client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
            )

        self.assertEqual(attempts, 2)
        self.assertEqual(calls, 2)

    async def test_preflight_timeout_is_classified_as_endpoint_timeout(self):
        async def handler(request):
            raise httpx.ConnectTimeout("timed out", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "endpoint_timeout"):
                await _preflight(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    sleep=_no_sleep,
                )

    async def test_preflight_retries_stop_after_max_attempts(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError, rf"PREFLIGHT_CONNECT attempts={MAX_CONNECT_ATTEMPTS}"
            ):
                await _preflight(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
                )

        self.assertEqual(calls, MAX_CONNECT_ATTEMPTS)

    async def test_generation_connection_failure_then_safe_retry(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectTimeout("connect timed out", request=request)
            return _sse_response(
                {"choices": [{"delta": {"content": "PATCH"}, "finish_reason": "stop"}]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            text, attempts = await _generate_stream(
                client,
                "http://qwen.test/v1",
                DEFAULT_MODEL,
                "prompt",
                sleep=_no_sleep,
            )

        self.assertEqual(text, "PATCH")
        self.assertEqual(attempts, 2)
        self.assertEqual(calls, 2)

    async def test_generation_http_error_is_not_retried(self):
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(503, text="unavailable")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_HTTP.*status=503"):
                await _generate_stream(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    "prompt",
                    sleep=_no_sleep,
                )

        self.assertEqual(calls, 1)

    async def test_streaming_content_reconstructs_typed_and_string_parts(self):
        async def handler(request):
            body = json.loads(request.content)
            self.assertIs(body["stream"], True)
            return _sse_response(
                {"choices": [{"delta": {"content": "--- a/file\n"}}]},
                {
                    "choices": [
                        {
                            "delta": {"content": [{"type": "text", "text": "+++ b/file\n"}]},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            text, _attempts = await _generate_stream(
                client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
            )

        self.assertEqual(text, "--- a/file\n+++ b/file\n")

    async def test_generation_uses_mechanical_mode_request_settings(self):
        async def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["max_tokens"], MAX_OUTPUT_TOKENS)
            self.assertEqual(body["temperature"], TEMPERATURE)
            self.assertEqual(body["top_p"], TOP_P)
            self.assertEqual(body["top_k"], TOP_K)
            self.assertEqual(body["presence_penalty"], PRESENCE_PENALTY)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            return _sse_response(
                {"choices": [{"delta": {"content": "PATCH"}, "finish_reason": "stop"}]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _generate_stream(client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt")

    async def test_reasoning_content_chunks_are_not_final_output(self):
        async def handler(_request):
            return _sse_response(
                {"choices": [{"delta": {"reasoning_content": "private"}}]},
                {
                    "choices": [
                        {"delta": {"content": "FINAL"}, "finish_reason": "stop"}
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            text, _attempts = await _generate_stream(
                client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
            )

        self.assertEqual(text, "FINAL")

    async def test_reasoning_only_stream_is_an_error_without_reasoning_text(self):
        async def handler(_request):
            return _sse_response(
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": "do-not-expose"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError,
                r"GENERATION_EMPTY_FINAL .*reasoning_content_chars=13",
            ) as raised:
                await _generate_stream(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
                )

        self.assertNotIn("do-not-expose", str(raised.exception))

    async def test_length_finish_with_empty_final_output_is_an_error(self):
        async def handler(_request):
            return _sse_response(
                {
                    "choices": [
                        {"delta": {"content": ""}, "finish_reason": "length"}
                    ]
                }
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError, r"GENERATION_LENGTH .*finish_reason='length'"
            ):
                await _generate_stream(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
                )

    async def test_length_finish_with_partial_final_output_is_an_error(self):
        async def handler(_request):
            return _sse_response(
                {
                    "choices": [
                        {"delta": {"content": "partial"}, "finish_reason": "length"}
                    ]
                }
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, r"GENERATION_LENGTH"):
                await _generate_stream(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
                )

    async def test_midstream_failure_is_not_retried(self):
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FailingStream(),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_STREAM.*events=1"):
                await _generate_stream(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
                )

        self.assertEqual(calls, 1)

    async def test_shared_mcp_entrypoint_uses_preflight_and_streaming_worker(self):
        requests = []

        async def handler(request):
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})
            return _sse_response(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "--- file.txt\n+++ file.txt\n@@ -1 +1 @@\n-alpha\n+beta\n"
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        class TestClient(httpx.AsyncClient):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "file.txt").write_text("alpha\n", encoding="utf-8")
            with mock.patch.dict(
                "os.environ", {"QWEN_BASE_URL": "http://qwen.test/v1"}
            ), mock.patch("qwen_worker_server.httpx.AsyncClient", TestClient):
                direct = await run_qwen_worker("task", directory, ["file.txt"])
                mcp_result = await qwen_patch("task", directory, ["file.txt"])

        expected = "--- file.txt\n+++ file.txt\n@@ -1 +1 @@\n-alpha\n+beta\n"
        self.assertEqual(direct, expected)
        self.assertEqual(mcp_result, expected)
        self.assertEqual([request.method for request in requests], ["GET", "POST"] * 2)


class WorkerOutputValidationTests(unittest.TestCase):
    def test_valid_diff_and_need_files_are_accepted(self):
        _validate_worker_output("--- a.txt\n+++ a.txt\n@@ -1 +1 @@\n-old\n+new\n")
        _validate_worker_output("NEED_FILES:\npath/to/file.ts")

    def test_malformed_or_truncated_diff_is_rejected(self):
        for output in (
            "not a diff",
            "--- a.txt\n+++ a.txt",
            "--- a.txt\n+++ a.txt\n@@ -1 +1 @@\n-old",
        ):
            with self.subTest(output=output), self.assertRaisesRegex(
                RuntimeError, "GENERATION_INVALID_OUTPUT"
            ):
                _validate_worker_output(output)


if __name__ == "__main__":
    unittest.main()
