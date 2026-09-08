import asyncio
import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import httpx

from qwen_worker_server import (
    DEFAULT_MODEL,
    MAX_ERROR_BODY_BYTES,
    MAX_OUTPUT_TOKENS,
    PREFLIGHT_MAX_CONNECT_ATTEMPTS,
    PRESENCE_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    BoundedSSEParser,
    StreamLimits,
    _build_prompt,
    _configured_base_url,
    _generate_stream,
    _parse_chat_completion,
    _parse_chat_completion_response,
    _parse_chat_completion_stream,
    _preflight,
    _validate_worker_output,
    qwen_patch,
    run_qwen_worker,
)


class EndpointConfigurationTests(unittest.TestCase):
    def configured(self, value):
        with mock.patch.dict("os.environ", {"QWEN_BASE_URL": value}):
            return _configured_base_url()

    def test_loopback_urls_are_accepted(self):
        for value in (
            "http://127.0.0.1:11234/v1",
            "http://localhost:11234/v1",
            "https://[::1]:11234/v1",
            "http://127.0.0.2:11234/v1/",
            "http://[0:0:0:0:0:0:0:1]:11234/v1",
        ):
            with self.subTest(value=value):
                self.assertEqual(self.configured(value), value.rstrip("/"))

    def test_remote_hosts_are_rejected_without_dns_inference(self):
        for value in (
            "http://localhost.evil.example/v1",
            "http://127.0.0.1.evil.example/v1",
            "http://192.168.1.25:11234/v1",
            "https://example.com/v1",
            "http://qwen.test/v1",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                RuntimeError, "endpoint_remote_not_allowed"
            ):
                self.configured(value)

    def test_malformed_and_unsupported_urls_fail_with_controlled_errors(self):
        for value in (
            "http://[::1",
            "http://localhost:bad/v1",
            "http:///v1",
            "localhost:11234/v1",
            "file:///tmp/model",
            "ftp://localhost/model",
            "http://localhost:11234/v1?debug=1",
            "http://localhost:11234/v1#fragment",
            "http://localhost:11234/v1?",
            "http://localhost:11234/v1#",
            " http://localhost:11234/v1",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                RuntimeError, "endpoint_"
            ):
                self.configured(value)

    def test_userinfo_is_rejected_without_echoing_credentials(self):
        for value in (
            "http://user:secret@localhost:11234/v1",
            "http://localhost@evil.example/v1",
            "http://evil.example@127.0.0.1/v1",
        ):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError) as raised:
                    self.configured(value)
                self.assertIn("endpoint_credentials_not_allowed", str(raised.exception))
                self.assertNotIn("secret", str(raised.exception))

    def test_empty_endpoint_is_rejected(self):
        with mock.patch.dict("os.environ", {"QWEN_BASE_URL": "   "}):
            with self.assertRaisesRegex(RuntimeError, "endpoint_not_configured"):
                _configured_base_url()

    def test_patch_prompt_does_not_include_absolute_repository_root(self):
        root = Path("/tmp/unique-qwen-repository-root-sentinel")
        prompt = _build_prompt("task", root, [("src/main.py", "pass\n")], None)

        self.assertNotIn(str(root), prompt)
        self.assertIn("src/main.py", prompt)
        self.assertIn("Only modify files explicitly supplied", prompt)


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


class _DelayedGenerationHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.post_count += 1
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.server.inference_started += 1
        self.server.request_received.set()
        time.sleep(self.server.header_delay)
        body = b'data: {"choices":[{"delta":{"content":"PATCH"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        pass


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def _assert_generation_error_once(self, exception_type, expected_code):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise exception_type("PRIVATE_PROMPT_SENTINEL", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, expected_code):
                await _generate_stream(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    "PRIVATE_PROMPT_SENTINEL",
                    sleep=_no_sleep,
                )

        self.assertEqual(calls, 1)

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

    async def test_preflight_retries_multiple_transient_failures_then_succeeds(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls < 5:
                raise httpx.ConnectError("transient refusal", request=request)
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            attempts = await _preflight(
                client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
            )

        self.assertEqual(attempts, 5)
        self.assertEqual(calls, 5)

    async def test_preflight_read_disconnect_is_retried(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadError("transient disconnect", request=request)
            return httpx.Response(200, json={"data": [{"id": DEFAULT_MODEL}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            attempts = await _preflight(
                client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
            )

        self.assertEqual(attempts, 2)
        self.assertEqual(calls, 2)

    async def test_preflight_read_timeout_is_retried(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("header timed out", request=request)
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
                RuntimeError,
                rf"PREFLIGHT_CONNECT attempts={PREFLIGHT_MAX_CONNECT_ATTEMPTS}",
            ):
                await _preflight(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
                )

        self.assertEqual(calls, PREFLIGHT_MAX_CONNECT_ATTEMPTS)

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

    async def test_generation_pool_timeout_then_safe_retry(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.PoolTimeout("pool busy", request=request)
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

    async def test_generation_read_timeout_is_not_retried(self):
        await self._assert_generation_error_once(
            httpx.ReadTimeout, "GENERATION_READ_TIMEOUT.*attempts=1"
        )

    async def test_generation_write_timeout_is_not_retried(self):
        await self._assert_generation_error_once(
            httpx.WriteTimeout, "GENERATION_WRITE_TIMEOUT.*attempts=1"
        )

    async def test_generation_write_error_is_not_retried(self):
        await self._assert_generation_error_once(
            httpx.WriteError, "GENERATION_WRITE_ERROR.*attempts=1"
        )

    async def test_generation_read_error_is_not_retried(self):
        await self._assert_generation_error_once(
            httpx.ReadError, "GENERATION_READ_ERROR.*attempts=1"
        )

    async def test_generation_protocol_error_is_not_retried(self):
        await self._assert_generation_error_once(
            httpx.RemoteProtocolError, "GENERATION_PROTOCOL_ERROR.*attempts=1"
        )

    async def test_generation_connect_error_is_fail_closed(self):
        await self._assert_generation_error_once(
            httpx.ConnectError, "GENERATION_CONNECT_ERROR.*attempts=1"
        )

    async def test_generation_safe_retry_is_bounded(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ConnectTimeout("connect timed out", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(
                RuntimeError, r"GENERATION_CONNECT_TIMEOUT.*attempts=3"
            ):
                await _generate_stream(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    "prompt",
                    sleep=_no_sleep,
                )

        self.assertEqual(calls, 3)

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

    async def test_generation_429_is_not_retried(self):
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            return httpx.Response(429, text="rate limited")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_HTTP.*status=429"):
                await _generate_stream(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    "prompt",
                    sleep=_no_sleep,
                )

        self.assertEqual(calls, 1)

    async def test_generation_deadline_is_shared_across_safe_retries(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ConnectTimeout("connect timed out", request=request)

        async def slow_backoff(_delay):
            await asyncio.sleep(0.02)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with mock.patch("qwen_worker_server.MAX_GENERATION_SECONDS", 0.01):
                with self.assertRaisesRegex(RuntimeError, "GENERATION_DEADLINE"):
                    await _generate_stream(
                        client,
                        "http://qwen.test/v1",
                        DEFAULT_MODEL,
                        "prompt",
                        sleep=slow_backoff,
                    )

        self.assertEqual(calls, 1)

    async def test_generation_stream_limit_is_not_retried(self):
        calls = 0

        async def handler(_request):
            nonlocal calls
            calls += 1
            return _sse_response(
                {"choices": [{"delta": {"content": "PATCH"}, "finish_reason": "stop"}]}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with mock.patch("qwen_worker_server.MAX_STREAM_BYTES", 1):
                with self.assertRaisesRegex(RuntimeError, "GENERATION_STREAM_LIMIT"):
                    await _generate_stream(
                        client,
                        "http://qwen.test/v1",
                        DEFAULT_MODEL,
                        "prompt",
                    )

        self.assertEqual(calls, 1)

    async def test_generation_retry_telemetry_is_metadata_only(self):
        stderr = io.StringIO()

        async def handler(request):
            raise httpx.ReadTimeout("PRIVATE_PROMPT_SENTINEL", request=request)

        with mock.patch.dict("os.environ", {"QWEN_DEBUG": "1"}), contextlib.redirect_stderr(
            stderr
        ):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler), follow_redirects=True
            ) as client:
                with self.assertRaisesRegex(RuntimeError, "GENERATION_READ_TIMEOUT"):
                    await _generate_stream(
                        client,
                        "http://qwen.test/v1",
                        DEFAULT_MODEL,
                        "PRIVATE_PROMPT_SENTINEL",
                    )

        diagnostic = stderr.getvalue()
        self.assertIn(
            "operation=generation method=POST attempt=1 "
            "error_kind=read_timeout retry=false",
            diagnostic,
        )
        self.assertNotIn("PRIVATE_PROMPT_SENTINEL", diagnostic)

    async def test_preflight_and_generation_do_not_follow_redirects(self):
        preflight_calls = 0

        async def preflight_handler(_request):
            nonlocal preflight_calls
            preflight_calls += 1
            return httpx.Response(
                307, headers={"location": "https://example.com/remote-models"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(preflight_handler), follow_redirects=True
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "PREFLIGHT_HTTP.*status=307"):
                await _preflight(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, sleep=_no_sleep
                )
        self.assertEqual(preflight_calls, 1)

        generation_calls = 0

        async def generation_handler(_request):
            nonlocal generation_calls
            generation_calls += 1
            return httpx.Response(
                307, headers={"location": "https://example.com/remote-completions"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(generation_handler), follow_redirects=True
        ) as client:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_HTTP.*status=307"):
                await _generate_stream(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    "PRIVATE_PROMPT_SENTINEL",
                    sleep=_no_sleep,
                )
        self.assertEqual(generation_calls, 1)

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
            with self.assertRaisesRegex(RuntimeError, "GENERATION_READ_ERROR.*events=1"):
                await _generate_stream(
                    client, "http://qwen.test/v1", DEFAULT_MODEL, "prompt"
                )

        self.assertEqual(calls, 1)

    async def test_done_is_terminal_and_utf8_can_split_across_raw_chunks(self):
        payload = json.dumps(
            {"choices": [{"delta": {"content": "café"}, "finish_reason": "stop"}]},
            ensure_ascii=False,
        ).encode("utf-8")
        body = b"data: " + payload + b"\r\n\r\ndata: [DONE]\r\n\r\n"
        body += b"data: this-must-not-be-parsed\r\n\r\n"
        split = body.index("é".encode("utf-8")) + 1
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((body[:split], body[split:])),
        )
        try:
            text = await _parse_chat_completion_stream(
                response,
                attempts=1,
                generation_started=asyncio.get_running_loop().time(),
                headers_ms=0,
            )
        finally:
            await response.aclose()
        self.assertEqual(text, "café")

    async def test_sse_comments_blank_lines_and_multiple_data_fields_are_supported(self):
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream(
                (b": heartbeat\r\n\r\ndata: first\r\ndata: second\r\n\r\n",)
            ),
        )
        parser = BoundedSSEParser(StreamLimits(max_events=4))
        try:
            values = [value async for value in parser.iter_data(response)]
        finally:
            await response.aclose()
        self.assertEqual(values, ["first\nsecond"])
        self.assertEqual(parser.event_count, 2)

    async def test_raw_stream_byte_limit_counts_heartbeats_and_json(self):
        body = b": heartbeat\n\n" + (
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
        ) + b"data: [DONE]\n\n"
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((body,)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_STREAM_LIMIT"):
                await _parse_chat_completion_stream(
                    response,
                    attempts=1,
                    generation_started=asyncio.get_running_loop().time(),
                    headers_ms=0,
                    limits=StreamLimits(max_stream_bytes=len(body) - 1),
                )
        finally:
            await response.aclose()

    async def test_usage_only_event_is_accepted_after_patch_content(self):
        response = _sse_response(
            {"choices": [{"delta": {"content": "PATCH"}}]},
            {"choices": [], "usage": {"completion_tokens": 7}},
        )
        self.assertEqual(
            await _parse_chat_completion_stream(
                response,
                attempts=1,
                generation_started=asyncio.get_running_loop().time(),
                headers_ms=0,
            ),
            "PATCH",
        )

    async def test_malformed_utf8_is_rejected_before_json_parsing(self):
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((b"data: \xff\n\n",)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_MALFORMED"):
                await _parse_chat_completion_stream(
                    response,
                    attempts=1,
                    generation_started=asyncio.get_running_loop().time(),
                    headers_ms=0,
                )
        finally:
            await response.aclose()

    async def test_single_sse_event_and_event_count_are_bounded(self):
        large_event = b"data: " + b"x" * 100
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((large_event,)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_EVENT_SIZE_LIMIT"):
                await _parse_chat_completion_stream(
                    response,
                    attempts=1,
                    generation_started=asyncio.get_running_loop().time(),
                    headers_ms=0,
                    limits=StreamLimits(max_event_bytes=16),
                )
        finally:
            await response.aclose()

        heartbeat_body = b": heartbeat\n\n" * 4
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ChunkedStream((heartbeat_body,)),
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_EVENT_LIMIT"):
                await _parse_chat_completion_stream(
                    response,
                    attempts=1,
                    generation_started=asyncio.get_running_loop().time(),
                    headers_ms=0,
                    limits=StreamLimits(max_events=3),
                )
        finally:
            await response.aclose()

    async def test_generation_output_and_reasoning_limits_fail_closed(self):
        output_response = _sse_response(
            {"choices": [{"delta": {"content": "12345"}, "finish_reason": "stop"}]}
        )
        with self.assertRaisesRegex(RuntimeError, "GENERATION_OUTPUT_LIMIT"):
            await _parse_chat_completion_stream(
                output_response,
                attempts=1,
                generation_started=asyncio.get_running_loop().time(),
                headers_ms=0,
                limits=StreamLimits(max_final_chars=4),
            )

        reasoning_response = _sse_response(
            {
                "choices": [
                    {"delta": {"reasoning_content": "private"}, "finish_reason": "stop"}
                ]
            }
        )
        with self.assertRaisesRegex(RuntimeError, "GENERATION_REASONING_LIMIT"):
            await _parse_chat_completion_stream(
                reasoning_response,
                attempts=1,
                generation_started=asyncio.get_running_loop().time(),
                headers_ms=0,
                limits=StreamLimits(max_reasoning_chars=3),
            )

    async def test_non_2xx_body_is_read_with_a_byte_cap(self):
        error_stream = _ChunkedStream((b"E" * 512 for _ in range(100)))

        async def handler(_request):
            return httpx.Response(
                503,
                headers={"content-type": "text/plain"},
                stream=error_stream,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(RuntimeError, "GENERATION_HTTP.*status=503") as raised:
                await _generate_stream(
                    client,
                    "http://qwen.test/v1",
                    DEFAULT_MODEL,
                    "prompt",
                    sleep=_no_sleep,
                )
        self.assertLess(len(str(raised.exception)), MAX_ERROR_BODY_BYTES)
        self.assertLess(error_stream.yielded, 100)

    async def test_generation_deadline_covers_a_slow_drip(self):
        async def handler(_request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_ChunkedStream((b": heartbeat\n\n", b": delayed\n\n"), delay=0.05),
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with mock.patch("qwen_worker_server.MAX_GENERATION_SECONDS", 0.01):
                with self.assertRaisesRegex(RuntimeError, "GENERATION_DEADLINE"):
                    await _generate_stream(
                        client,
                        "http://qwen.test/v1",
                        DEFAULT_MODEL,
                        "prompt",
                        sleep=_no_sleep,
                    )

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

        client_options = []

        class TestClient(httpx.AsyncClient):
            def __init__(self, *args, **kwargs):
                client_options.append(dict(kwargs))
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "file.txt").write_text("alpha\n", encoding="utf-8")
            with mock.patch.dict(
                "os.environ", {"QWEN_BASE_URL": "http://127.0.0.1:11234/v1"}
            ), mock.patch("qwen_worker_server.httpx.AsyncClient", TestClient):
                direct = await run_qwen_worker("task", directory, ["file.txt"])
                mcp_result = await qwen_patch("task", directory, ["file.txt"])

        expected = "--- file.txt\n+++ file.txt\n@@ -1 +1 @@\n-alpha\n+beta\n"
        self.assertEqual(direct, expected)
        self.assertEqual(mcp_result, expected)
        self.assertEqual([request.method for request in requests], ["GET", "POST"] * 2)
        self.assertEqual(
            [options["trust_env"] for options in client_options], [False, False]
        )
        self.assertEqual(
            [options["follow_redirects"] for options in client_options], [False, False]
        )


class GenerationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_accept_then_header_read_timeout_starts_one_inference(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _DelayedGenerationHandler)
        server.post_count = 0
        server.inference_started = 0
        server.request_received = threading.Event()
        server.header_delay = 0.1
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        timeout = httpx.Timeout(connect=1.0, pool=1.0, write=1.0, read=0.02)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                with self.assertRaisesRegex(
                    RuntimeError, "GENERATION_READ_TIMEOUT.*attempts=1"
                ):
                    await _generate_stream(
                        client,
                        f"http://{host}:{port}/v1",
                        DEFAULT_MODEL,
                        "PRIVATE_PROMPT_SENTINEL",
                        sleep=_no_sleep,
                    )

            self.assertTrue(server.request_received.wait(0.5))
            self.assertEqual(server.inference_started, 1)
            self.assertEqual(server.post_count, 1)
            await asyncio.sleep(0.12)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1.0)


class WorkerOutputValidationTests(unittest.TestCase):
    @staticmethod
    def validate(output, allowed_files=("src/foo.py",)):
        return _validate_worker_output(output, allowed_files=allowed_files)

    @staticmethod
    def diff(old_path="src/foo.py", new_path="src/foo.py"):
        return (
            f"--- {old_path}\n"
            f"+++ {new_path}\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

    def test_valid_diff_and_need_files_are_accepted(self):
        self.validate(self.diff("a/src/foo.py", "b/src/foo.py"))
        self.validate(
            "diff --git a/src/foo.py b/src/foo.py\n"
            "index 1234567..89abcde 100644\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        self.validate("NEED_FILES:\npath/to/file.ts")

    def test_need_files_relative_path_guard_is_preserved(self):
        for output in ("NEED_FILES:\n/absolute/file.py", "NEED_FILES:\n../outside.py"):
            with self.subTest(output=output), self.assertRaisesRegex(
                RuntimeError, "unsafe NEED_FILES path"
            ):
                self.validate(output)

    def test_valid_multi_hunk_diff_is_accepted(self):
        self.validate(
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "@@ -3 +3 @@\n"
            "-old2\n"
            "+new2\n"
        )

    def test_unauthorized_target_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_TARGET_NOT_ALLOWED"):
            self.validate(self.diff("a/src/bar.py", "b/src/bar.py"))

    def test_sensitive_unrelated_target_is_rejected(self):
        for target in ("a/.env", "b/.github/workflows/release.yml"):
            with self.subTest(target=target), self.assertRaisesRegex(
                RuntimeError, "PATCH_TARGET_NOT_ALLOWED"
            ):
                self.validate(self.diff(target, target))

    def test_traversal_targets_are_rejected(self):
        for old_path, new_path in (
            ("a/../secret.py", "b/../secret.py"),
            ("a/../../secret.py", "b/../../secret.py"),
            ("a/src/../../private.py", "b/src/../../private.py"),
        ):
            with self.subTest(old_path=old_path), self.assertRaisesRegex(
                RuntimeError, "PATCH_TARGET_TRAVERSAL"
            ):
                self.validate(self.diff(old_path, new_path))

    def test_absolute_and_windows_paths_are_rejected(self):
        for target, reason in (
            ("/Users/example/secret.py", "PATCH_TARGET_ABSOLUTE"),
            ("C:\\Users\\example\\secret.py", "PATCH_TARGET_ABSOLUTE"),
        ):
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, reason):
                self.validate(self.diff(target, target))

    def test_prefix_collisions_are_rejected(self):
        for target in ("a/src/foo.py.evil", "a/src/foo.py/../bar.py", "a/src2/foo.py"):
            reason = "PATCH_TARGET_TRAVERSAL" if ".." in target else "PATCH_TARGET_NOT_ALLOWED"
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, reason):
                self.validate(self.diff(target, target))

    def test_authorized_multiple_file_patch_is_accepted(self):
        output = (
            self.diff("a/src/foo.py", "b/src/foo.py")
            + self.diff("a/src/bar.py", "b/src/bar.py")
        )
        self.validate(output, allowed_files=("src/foo.py", "src/bar.py"))

    def test_one_unauthorized_section_rejects_whole_artifact(self):
        output = self.diff("a/src/foo.py", "b/src/foo.py") + self.diff(
            "a/src/bar.py", "b/src/bar.py"
        )
        with self.assertRaisesRegex(RuntimeError, "PATCH_TARGET_NOT_ALLOWED"):
            self.validate(output)

    def test_new_file_is_rejected_by_existing_file_policy(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_NEW_FILE_NOT_ALLOWED"):
            self.validate(self.diff("/dev/null", "b/src/new.py"))

    def test_delete_is_rejected_by_existing_file_policy(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_DELETE_NOT_ALLOWED"):
            self.validate(self.diff("a/src/foo.py", "/dev/null"))

    def test_rename_and_cross_file_targets_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_RENAME_NOT_ALLOWED"):
            self.validate(
                self.diff("a/src/foo.py", "b/src/bar.py"),
                allowed_files=("src/foo.py", "src/bar.py"),
            )

    def test_duplicate_target_sections_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_DUPLICATE_TARGET"):
            self.validate(self.diff() + self.diff())

    def test_null_to_null_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_INVALID_NULL_TARGET"):
            self.validate(self.diff("/dev/null", "/dev/null"))

    def test_nested_authorized_path_and_git_prefixes_are_accepted(self):
        self.validate(
            self.diff("a/src/pkg/module.py", "b/src/pkg/module.py"),
            allowed_files=("src/pkg/module.py",),
        )

    def test_lexical_symlink_alias_does_not_bypass_canonical_allowed_identity(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_TARGET_NOT_ALLOWED"):
            self.validate(
                self.diff("a/allowed/link.py", "b/allowed/link.py"),
                allowed_files=("allowed/real.py",),
            )

    def test_quoted_paths_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "PATCH_TARGET_QUOTED"):
            self.validate(self.diff('"a/src/foo.py"', '"b/src/foo.py"'))

    def test_tab_timestamp_suffix_is_supported(self):
        self.validate(
            self.diff(
                "a/src/foo.py\t2026-09-08 12:00:00",
                "b/src/foo.py\t2026-09-08 12:00:00",
            )
        )

    def test_malformed_or_truncated_diff_is_rejected(self):
        for output in (
            "not a diff",
            "--- a.txt\n+++ a.txt",
            "--- a.txt\n+++ a.txt\n@@ -1 +1 @@\n-old",
        ):
            with self.subTest(output=output), self.assertRaisesRegex(
                RuntimeError, "GENERATION_INVALID_OUTPUT"
            ):
                self.validate(output)


if __name__ == "__main__":
    unittest.main()
