from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

def _tool_call_ids_from_assistant(msg: dict[str, Any]) -> list[str]:
    """Collect tool call ids from assistant message content blocks."""
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    ids: list[str] = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "toolCall":
            tid = p.get("id")
            if tid is not None and str(tid) != "":
                ids.append(str(tid))
    return ids


def validate_session_messages(messages: list[dict[str, Any]]) -> None:
    """
    Provider-agnostic session checks before calling any Model:

    - After an assistant message that contains toolCall blocks, each id must be answered
      by a toolResult (same toolCallId) before the next user or assistant message.
    - toolResult must not appear without a matching pending tool call.

    Raises ValueError with a short explanation if the history is inconsistent.
    """
    pending: Counter[str] = Counter()

    def _pending_total() -> int:
        return sum(pending.values())

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{idx}]: expected dict, got {type(msg).__name__}")
        role = msg.get("role")
        if role == "user":
            if _pending_total() > 0:
                raise ValueError(
                    f"messages[{idx}]: user message while tool results are still pending "
                    f"for tool_call_ids: {list(pending.elements())}"
                )
            pending.clear()
        elif role == "assistant":
            if _pending_total() > 0:
                raise ValueError(
                    f"messages[{idx}]: assistant message while tool results are still pending "
                    f"for tool_call_ids: {list(pending.elements())}"
                )
            ids = _tool_call_ids_from_assistant(msg)
            pending = Counter(ids) if ids else Counter()
        elif role == "toolResult":
            tid_raw = msg.get("toolCallId")
            if tid_raw is None or str(tid_raw) == "":
                raise ValueError(f"messages[{idx}]: toolResult missing toolCallId")
            tid = str(tid_raw)
            if _pending_total() == 0:
                raise ValueError(
                    f"messages[{idx}]: toolResult without a preceding assistant tool call"
                )
            if pending.get(tid, 0) <= 0:
                raise ValueError(
                    f"messages[{idx}]: toolResult toolCallId={tid!r} does not match pending calls"
                )
            pending[tid] -= 1
        else:
            if _pending_total() > 0:
                raise ValueError(
                    f"messages[{idx}]: role {role!r} not allowed while tool results are pending"
                )

    if _pending_total() > 0:
        raise ValueError(
            "End of messages: missing toolResult for tool_call_ids: "
            f"{list(pending.elements())}"
        )


def extract_text_from_message(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    content = message.get("content") or []
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
