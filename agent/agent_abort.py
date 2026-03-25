"""
Non-core helpers for the agent loop: cooperative abort, session validation, text extraction.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

def abort_requested(signal: Any | None) -> bool:
    """True if cooperative abort was requested (asyncio.Event or any object with is_set() -> bool)."""
    if signal is None:
        return False
    is_set = getattr(signal, "is_set", None)
    if callable(is_set):
        try:
            return bool(is_set())
        except Exception:
            return False
    return False


class AgentAbortController:
    """
    Cooperative cancel for run_agent / _agent_loop.

    Pass ``controller.signal`` as ``abort_signal=...``. Call ``abort()`` from the same
    process to stop streaming (model checks between chunks; loop checks between tools/rounds).

    If you call ``abort()`` from another thread, schedule ``controller.abort()`` on the
    event loop with ``asyncio.get_event_loop().call_soon_threadsafe(controller.abort)``.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def abort(self) -> None:
        self._event.set()

    @property
    def signal(self) -> asyncio.Event:
        return self._event



