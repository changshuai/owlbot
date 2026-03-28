"""
Persist chat message lists per (agent_id, session_key) under workspace-<id>/chat_sessions/.

Survives process restarts; separate from memory.write (facts) and context_archive (trimmed prefix).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from common.paths import get_agent_workspace


def _persist_enabled() -> bool:
    v = (os.getenv("PERSIST_CHAT_SESSIONS", "1") or "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _safe_session_filename(session_key: str) -> str:
    key = (session_key or "").strip()
    if not key:
        key = "default-session"
    safe = re.sub(r"[^a-zA-Z0-9._@-]+", "_", key).strip("._-")
    return safe or "default-session"


def chat_session_path(agent_id: str, session_key: str) -> Path:
    ws = get_agent_workspace(agent_id)
    d = ws / "chat_sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_safe_session_filename(session_key)}.json"


def load_chat_session(agent_id: str, session_key: str) -> list[dict[str, Any]]:
    """Load saved messages, or [] if missing, disabled, or invalid."""
    if not _persist_enabled():
        return []
    path = chat_session_path(agent_id, session_key)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [m for m in raw if isinstance(m, dict)]
    except Exception:
        return []


def save_chat_session(agent_id: str, session_key: str, messages: list[dict[str, Any]]) -> None:
    if not _persist_enabled():
        return
    path = chat_session_path(agent_id, session_key)
    path.write_text(
        json.dumps(messages, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
