"""Bounded profile adapter for qwen_agent.run_qwen_agent.

Exposes two canonical sampling profiles (non-thinking / thinking) and a thin
async wrapper that applies the active profile to every agent completion while
preserving all other request fields. Reasoning content is never fed into the
action parser; only ordinary delta.content is treated as action JSON.
"""

import asyncio
import contextvars
import hashlib
import importlib
import json
import os
import sqlite3
import stat
import sys
import threading
from pathlib import Path
from typing import Any

import qwen_agent

try:
    qwen_worker_server: Any = importlib.import_module("qwen_worker_server")
except ImportError:  # pragma: no cover - optional at runtime
    qwen_worker_server = None

# ---------------------------------------------------------------------------
# Canonical profiles
# ---------------------------------------------------------------------------

_NON_THINKING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "presence_penalty": 1.5,
    "chat_template_kwargs": {"enable_thinking": False},
}

_THINKING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0,
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "medium",
    },
}

PROFILES = {
    "non-thinking": _NON_THINKING,
    "thinking": _THINKING,
}

_PROFILE_ALIASES = {
    "a": "non-thinking",
    "b": "thinking",
}

_DEFAULT_PROFILE = "non-thinking"
_PROFILE_SANDBOX_BINDINGS: dict[str, str] = {}
_PROFILE_BINDINGS_LOCK = threading.RLock()


def _default_profile_bindings_path() -> Path:
    if sys.platform == "darwin":
        cache_root = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        cache_root = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
    else:
        cache_root = Path.home() / ".cache"
    return cache_root / "qwen-codex-worker" / "profile-bindings.sqlite3"


_PROFILE_BINDINGS_DB_PATH = _default_profile_bindings_path()

# Fields the adapter is allowed to overwrite on a request payload.
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "chat_template_kwargs",
)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_profile(profile):
    """Return the canonical profile name for ``profile``.

    Accepts canonical names and single-letter aliases (case-insensitive).
    ``None`` maps to the default profile. Raises ValueError otherwise.
    """
    if profile is None:
        return _DEFAULT_PROFILE
    if not isinstance(profile, str):
        raise ValueError("unknown profile: %r" % (profile,))
    key = profile.strip().lower()
    if key in _PROFILE_ALIASES:
        return _PROFILE_ALIASES[key]
    if key in PROFILES:
        return key
    raise ValueError("unknown profile: %r" % (profile,))


def apply_profile_payload(request, profile=_DEFAULT_PROFILE):
    """Return a copy of ``request`` with the profile's sampling fields applied.

    Only the four sampling fields plus chat_template_kwargs are replaced; every
    other field (including max_tokens) is preserved untouched.
    """
    name = normalize_profile(profile)
    spec = PROFILES[name]
    payload = dict(request)
    for field in _SAMPLING_FIELDS:
        value = spec[field]
        if isinstance(value, dict):
            payload[field] = dict(value)
        else:
            payload[field] = value
    return payload


def _claim_profile_sandbox(profile: str, sandbox: Any) -> None:
    """Persistently bind one sandbox to a single sampling profile."""
    if not isinstance(sandbox, (str, Path)):
        return
    try:
        root = Path(sandbox).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("profile sandbox must resolve to an existing directory") from exc
    if not root.is_dir():
        raise ValueError("profile sandbox must resolve to an existing directory")
    try:
        metadata = root.stat()
    except OSError:
        raise ValueError("profile sandbox must resolve to an existing directory") from None
    birth_time = getattr(metadata, "st_birthtime_ns", None)
    if birth_time is None and hasattr(metadata, "st_birthtime"):
        birth_time = int(metadata.st_birthtime * 1_000_000_000)
    if metadata.st_ino and birth_time is not None:
        identity = f"{metadata.st_dev}:{metadata.st_ino}:{birth_time}"
    else:
        identity = str(root)
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    with _PROFILE_BINDINGS_LOCK:
        existing = _PROFILE_SANDBOX_BINDINGS.get(key)
        if existing is not None and existing != profile:
            raise _profile_conflict(existing, profile)
        _claim_persistent_profile_binding(key, profile)
        _PROFILE_SANDBOX_BINDINGS[key] = profile


def _profile_conflict(existing: str, requested: str) -> ValueError:
    return ValueError(
        "sandbox is already assigned to the "
        f"{existing} profile; use a clean, separate sandbox for {requested}"
    )


def _assert_private_profile_state(path: Path, *, directory: bool) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError("sandbox profile identity could not be verified") from None
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("sandbox profile identity could not be verified")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise ValueError("sandbox profile identity could not be verified")


def _claim_persistent_profile_binding(key: str, profile: str) -> None:
    database = _PROFILE_BINDINGS_DB_PATH
    try:
        database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _assert_private_profile_state(database.parent, directory=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(database, flags, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        _assert_private_profile_state(database, directory=False)
        connection = sqlite3.connect(str(database), timeout=5)
    except (OSError, sqlite3.Error):
        raise ValueError("sandbox profile identity could not be verified") from None

    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS sandbox_profiles ("
            "sandbox_key TEXT PRIMARY KEY, profile TEXT NOT NULL)"
        )
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT profile FROM sandbox_profiles WHERE sandbox_key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            existing = row[0]
            if existing not in PROFILES:
                raise ValueError("sandbox profile identity could not be verified")
            if existing != profile:
                raise _profile_conflict(existing, profile)
        else:
            connection.execute(
                "INSERT INTO sandbox_profiles (sandbox_key, profile) VALUES (?, ?)",
                (key, profile),
            )
        connection.commit()
    except ValueError:
        connection.rollback()
        raise
    except sqlite3.Error:
        connection.rollback()
        raise ValueError("sandbox profile identity could not be verified") from None
    finally:
        connection.close()

# ---------------------------------------------------------------------------
# Per-invocation context
# ---------------------------------------------------------------------------

_active_profile: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "qwen_agent_profiles.active", default=None
)
_metrics_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "qwen_agent_profiles.metrics", default=None
)

# ---------------------------------------------------------------------------
# Safe one-time wrappers
# ---------------------------------------------------------------------------

_WRAPPED = False
_originals = {}


def _install_wrappers():
    global _WRAPPED
    if _WRAPPED:
        return

    original_start = getattr(qwen_agent, "_start_generation_request", None)
    original_turn = getattr(qwen_agent, "_generate_agent_turn", None)
    original_parse_stream = getattr(qwen_agent, "_parse_agent_stream", None)
    original_parse_action = getattr(qwen_agent, "parse_action", None)

    if original_start is not None:
        def _start_wrapper(*args, **kwargs):
            active = _active_profile.get()
            call_args = args
            if active is not None:
                applied = False
                for i, arg in enumerate(args):
                    if isinstance(arg, dict) and any(
                        k in arg for k in ("messages", "model")
                    ):
                        mutable_args = list(args)
                        mutable_args[i] = apply_profile_payload(arg, active)
                        call_args = tuple(mutable_args)
                        applied = True
                        break
                if not applied:
                    for key, value in kwargs.items():
                        if isinstance(value, dict) and any(
                            k in value for k in ("messages", "model")
                        ):
                            kwargs[key] = apply_profile_payload(value, active)
                            break
            return original_start(*call_args, **kwargs)

        _originals["_start_generation_request"] = original_start
        qwen_agent._start_generation_request = _start_wrapper

    if original_turn is not None:
        def _turn_wrapper(*args, **kwargs):
            return original_turn(*args, **kwargs)

        _originals["_generate_agent_turn"] = original_turn
        qwen_agent._generate_agent_turn = _turn_wrapper

    if original_parse_stream is not None:
        async def _parse_stream_wrapper(response, *args, **kwargs):
            metrics = _metrics_var.get()

            if metrics is not None:
                original_aiter = getattr(response, "aiter_bytes", None)
                if original_aiter is not None:
                    bound = getattr(
                        qwen_worker_server,
                        "MAX_SSE_EVENT_BYTES",
                        1024 * 1024,
                    ) if qwen_worker_server is not None else 1024 * 1024

                    async def _proxied_aiter():
                        event_buffer = b""
                        async for chunk in original_aiter():
                            if not chunk:
                                continue
                            event_buffer += chunk
                            # Forward every original byte unchanged.
                            yield chunk
                            # Process complete SSE events from buffer.
                            while b"\n\n" in event_buffer:
                                event_bytes, event_buffer = (
                                    event_buffer.split(b"\n\n", 1)
                                )
                                _process_sse_event(event_bytes, metrics)
                            # Bound our own event buffer.
                            if len(event_buffer) > bound:
                                event_buffer = b""

                    try:
                        response.aiter_bytes = _proxied_aiter
                    except Exception:
                        pass

            return await original_parse_stream(response, *args, **kwargs)

        _originals["_parse_agent_stream"] = original_parse_stream
        qwen_agent._parse_agent_stream = _parse_stream_wrapper

    if original_parse_action is not None:
        def _parse_action_wrapper(*args, **kwargs):
            metrics = _metrics_var.get()
            try:
                return original_parse_action(*args, **kwargs)
            except Exception:
                if metrics is not None:
                    metrics["protocol_errors"] += 1
                raise

        _originals["parse_action"] = original_parse_action
        qwen_agent.parse_action = _parse_action_wrapper

    _WRAPPED = True


def _process_sse_event(event_bytes, metrics):
    """Inspect a single SSE event for first-token and usage metrics."""
    try:
        text = event_bytes.decode("utf-8", errors="replace")
    except Exception:
        return

    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload_str = line[5:].strip()
        if not payload_str or payload_str == "[DONE]":
            continue
        try:
            obj = json.loads(payload_str)
        except Exception:
            continue

        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                has_content = isinstance(content, str) and content != ""
                has_reasoning = isinstance(reasoning, str) and reasoning != ""
                if (has_content or has_reasoning) and metrics["first_token_ms"] is None:
                    metrics["first_token_ms"] = (
                        asyncio.get_running_loop().time() - metrics["start_time"]
                    ) * 1000.0

        usage = obj.get("usage")
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict) and "reasoning_tokens" in details:
                rt = details["reasoning_tokens"]
                if isinstance(rt, int) and not isinstance(rt, bool) and rt >= 0:
                    metrics["reasoning_call_count"] += 1
                    metrics["reasoning_sum"] += rt
                else:
                    metrics["reasoning_valid"] = False

# ---------------------------------------------------------------------------
# Public async entry point
# ---------------------------------------------------------------------------


async def run_qwen_agent_profile(
    profile: str | None = _DEFAULT_PROFILE, **kwargs: Any
) -> dict[str, Any]:
    """Run qwen_agent.run_qwen_agent with a named sampling profile applied.

    Rejects unknown profile names before any network access. Tracks per-invocation
    metrics (first_token_ms, reasoning_tokens, response_parse_errors) and returns the
    agent result dict augmented with those metrics.
    """
    # Reject unknown profiles before touching the network.
    name = normalize_profile(profile)
    _claim_profile_sandbox(name, kwargs.get("sandbox"))

    _install_wrappers()

    metrics: dict[str, Any] = {
        "start_time": asyncio.get_running_loop().time(),
        "first_token_ms": None,
        "reasoning_valid": True,
        "reasoning_sum": 0,
        "reasoning_call_count": 0,
        "protocol_errors": 0,
    }

    token_profile = _active_profile.set(name)
    token_metrics = _metrics_var.set(metrics)

    try:
        result = await qwen_agent.run_qwen_agent(**kwargs)
    finally:
        _active_profile.reset(token_profile)
        _metrics_var.reset(token_metrics)

    reasoning_tokens = None
    if metrics["reasoning_valid"] and metrics["reasoning_call_count"] > 0:
        reasoning_tokens = metrics["reasoning_sum"]

    out = result.as_dict()
    # Preserve the agent's cumulative protocol error total. The profile adapter's
    # parser wrapper measures a separate subset and must not hide that total.
    out.setdefault("protocol_errors", 0)
    out["profile"] = name
    out["first_token_ms"] = metrics["first_token_ms"]
    out["reasoning_tokens"] = reasoning_tokens
    out["response_parse_errors"] = metrics["protocol_errors"]
    return out
