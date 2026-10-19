"""Load the local oMLX API credential without exposing it to Codex config."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

MAX_SETTINGS_BYTES = 2_000_000


def load_api_key(settings_path: str | Path | None = None) -> str:
    """Read auth.api_key from a private, regular oMLX settings file."""
    configured_path = settings_path or os.environ.get("QWEN_OMLX_SETTINGS_PATH")
    path = (
        Path(configured_path)
        if configured_path
        else Path.home() / ".omlx" / "settings.json"
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise RuntimeError("OMLX_API_KEY_SETTINGS_UNAVAILABLE") from None

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("OMLX_API_KEY_SETTINGS_INVALID")
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError("OMLX_API_KEY_SETTINGS_PERMISSIONS")
        with os.fdopen(descriptor, "rb", closefd=False) as settings_file:
            raw = settings_file.read(MAX_SETTINGS_BYTES + 1)
    finally:
        os.close(descriptor)

    if len(raw) > MAX_SETTINGS_BYTES:
        raise RuntimeError("OMLX_API_KEY_SETTINGS_TOO_LARGE")
    try:
        settings: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("OMLX_API_KEY_SETTINGS_INVALID") from None

    auth = settings.get("auth") if isinstance(settings, dict) else None
    key = auth.get("api_key") if isinstance(auth, dict) else None
    if (
        not isinstance(key, str)
        or not key
        or len(key) > 4096
        or not key.isascii()
        or key != key.strip()
        or any(ord(character) < 33 or ord(character) > 126 for character in key)
    ):
        raise RuntimeError("OMLX_API_KEY_MISSING_OR_INVALID")
    return key


def api_headers(settings_path: str | Path | None = None) -> dict[str, str]:
    """Return only the in-memory Authorization header required by oMLX."""
    return {"Authorization": f"Bearer {load_api_key(settings_path)}"}
