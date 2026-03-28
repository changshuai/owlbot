"""
Context window helpers: shrink oversized tool results and trim message history.

Single scheme (see Design.txt):
- Tool output over per-message char budget: optional summarizer model, else head+tail truncation.
- History over approximate budget: drop oldest valid prefix, archive to disk, insert user placeholder.
"""
from __future__ import annotations

import os
import time
from typing import Any

from llms import Context, get_env_api_key, get_model
from .message_validator import extract_text_from_message, validate_session_messages

from common.colors import RESET, YELLOW

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")

# --- Tool result shrink ---
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "12000"))
TOOL_SUMMARY_INPUT_MAX_CHARS = int(os.getenv("TOOL_SUMMARY_INPUT_MAX_CHARS", "24000"))

# Optional: lightweight model for summarizing tool output (falls back to truncation if unset or on error).
SUMMARIZER_MODEL_ID = os.getenv("SUMMARIZER_MODEL_ID", "").strip()

# --- History trim (approximate chars: ~4 per token) ---
CONTEXT_BUDGET_RATIO = float(os.getenv("CONTEXT_BUDGET_RATIO", "0.72"))
# Absolute cap on history budget in tokens (system + messages), rough estimate. min(ratio*window, this).
# Set to 0 to disable and use only CONTEXT_BUDGET_RATIO * context_window.
CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "16000"))


def history_budget_chars(model: Any) -> int:
    """Approximate max chars for system+messages before trimming; ~4 chars per token."""
    window = getattr(model, "context_window", 0) or 0
    ratio_budget = max(1024, int(window * CONTEXT_BUDGET_RATIO * 4))
    if CONTEXT_MAX_TOKENS <= 0:
        return ratio_budget
    cap_budget = CONTEXT_MAX_TOKENS * 4
    return min(ratio_budget, cap_budget)


def get_model_for_id(model_id: str):
    """Return LLM Model for the given model id (uses MODEL_PROVIDER for provider)."""
    return get_model(MODEL_PROVIDER, model_id, api_key=get_env_api_key(MODEL_PROVIDER))


def estimate_message_chars(msg: dict[str, Any]) -> int:
    role = msg.get("role")
    if role == "user":
        c = msg.get("content", "")
        if isinstance(c, str):
            return len(c)
        if isinstance(c, list):
            return sum(len(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
        return len(str(c))
    c = msg.get("content") or []
    if isinstance(c, list):
        return sum(len(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    return len(str(c))


def estimate_context_chars(system: str, messages: list[dict[str, Any]]) -> int:
    return len(system or "") + sum(estimate_message_chars(m) for m in messages if isinstance(m, dict))


def truncate_head_tail(text: str, max_chars: int) -> str:
    """Keep the first and last halves of the budget (minus separator), per Design.txt."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    sep = "\n\n... [middle omitted] ...\n\n"
    if max_chars <= len(sep) + 2:
        return text[:max_chars]
    avail = max_chars - len(sep)
    h = avail // 2
    t = avail - h
    return text[:h] + sep + text[-t:]


def _message_suffix_is_valid(suffix: list[dict[str, Any]]) -> bool:
    try:
        validate_session_messages(suffix)
        return True
    except ValueError:
        return False


def latest_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
    return ""


async def shrink_tool_result_text(
    *,
    tool_name: str,
    raw_text: str,
    user_question: str,
) -> tuple[str, str]:
    """
    If raw_text exceeds TOOL_RESULT_MAX_CHARS, summarize via SUMMARIZER_MODEL_ID if set,
    else truncate_head_tail. Returns (text, method) where method is none|summarize|truncate_head_tail.
    """
    if len(raw_text) <= TOOL_RESULT_MAX_CHARS:
        return raw_text, "none"

    data = raw_text
    if len(data) > TOOL_SUMMARY_INPUT_MAX_CHARS:
        half = TOOL_SUMMARY_INPUT_MAX_CHARS // 2
        data = data[:half] + "\n... [middle omitted for summarization] ...\n" + data[-half:]

    if SUMMARIZER_MODEL_ID:
        try:
            model = get_model_for_id(SUMMARIZER_MODEL_ID)
            sys = (
                "You shorten tool execution output for an autonomous agent.\n"
                "Preserve errors, exit codes, file paths, numbers, and key lines.\n"
                "Reply with plain text only (no JSON, no markdown fences)."
            )
            user = (
                f"User request: {user_question}\n\n"
                f"Tool name: {tool_name}\n\n"
                "Tool output:\n=== BEGIN ===\n"
                f"{data}\n"
                "=== END ===\n"
            )
            ctx = Context(
                system_prompt=sys,
                messages=[{"role": "user", "content": user, "timestamp": int(time.time() * 1000)}],
                tools=None,
            )
            folded = await model.invoke(ctx, {"max_tokens": 2048})
            out = extract_text_from_message(folded).strip()
            if out:
                if len(out) > TOOL_RESULT_MAX_CHARS:
                    out = truncate_head_tail(out, TOOL_RESULT_MAX_CHARS)
                return out, "summarize"
        except Exception as exc:
            logger.info(f"{YELLOW}tool result summarization failed: {exc}{RESET}")

    return truncate_head_tail(raw_text, TOOL_RESULT_MAX_CHARS), "truncate_head_tail"


async def maybe_trim_history_for_budget(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    tool_ctx: dict[str, Any] | None = None,
) -> None:
    """
    If system+messages exceed approximate budget, remove the smallest valid prefix (keep recent suffix),
    archive removed messages under memory/, and prepend a user placeholder with the archive path.
    """
    model = get_model_for_id(model_id)
    window = getattr(model, "context_window", 0) or 0
    if window <= 0:
        return

    budget_chars = history_budget_chars(model)
    total = estimate_context_chars(system, messages)
    if total <= budget_chars:
        return

    chosen_start: int | None = None
    for start in range(len(messages) + 1):
        suffix = messages[start:]
        if not _message_suffix_is_valid(suffix):
            continue
        if estimate_context_chars(system, suffix) <= budget_chars:
            chosen_start = start
            break

    if chosen_start is None:
        logger.info(f"{YELLOW}history trim: no valid suffix fits budget; leaving messages unchanged{RESET}")
        return

    if chosen_start == 0:
        return

    removed = messages[:chosen_start]
    archive_ref = ""
    ctx = tool_ctx or {}
    agent_id = ctx.get("agent_id") or "default"
    session_key = ctx.get("session_key") or ""
    try:
        from .memory_store import get_memory_store

        store = get_memory_store(agent_id)
        archive_ref = store.archive_messages(session_key, removed)
    except Exception as exc:
        logger.info(f"{YELLOW}history archive failed: {exc}{RESET}")
        archive_ref = f"(archive failed: {exc})"

    placeholder: dict[str, Any] = {
        "role": "user",
        "content": (
            f"[Earlier conversation was removed from context to stay within the model window "
            f"({len(removed)} messages archived). "
            f"Archive: {archive_ref}]"
        ),
        "timestamp": int(time.time() * 1000),
        "details": {"context_trim": True, "archived_count": len(removed), "archive_ref": archive_ref},
    }

    messages[:] = [placeholder] + messages[chosen_start:]
