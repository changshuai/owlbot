from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, AsyncIterator

from LLMs import Context
from .agent_ import AgentManager
from .agent_abort import abort_requested, AgentAbortController
from .message_validator import extract_text_from_message, validate_session_messages
from .tools import TOOLS_LLM, process_tool_call
from .context_compaction import (
    TOOL_RESULT_MAX_CHARS,
    estimate_context_chars,
    get_model_for_id,
    latest_user_text,
    maybe_trim_history_for_budget,
    shrink_tool_result_text,
)
from common.colors import (
    BLUE,
    BOLD,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RESET,
    YELLOW,
)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
MODEL_ID = os.getenv("MODEL_ID", "deepseek/deepseek-chat")

# Max model↔tool rounds per run (prevents infinite tool loops).
MAX_AGENT_TOOL_ROUNDS = 15

_agent_semaphore: asyncio.Semaphore | None = None


def _get_model_for_id(model_id: str):
    """Delegate to context_compaction so tests can patch `get_model_for_id` on that module."""
    return get_model_for_id(model_id)


def _messages_without_trailing_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def _tool_schema_chars(tools: list[Any] | None) -> int:
    if not tools:
        return 0
    return len(json.dumps(tools, ensure_ascii=False))


def _log_round_token_usage(
    round_no: int,
    system: str,
    messages: list[dict[str, Any]],
    final_message: dict[str, Any],
    *,
    tools: list[Any] | None,
) -> None:
    """Log per model round: API usage when present, else rough token estimate (~4 chars/token)."""
    usage = final_message.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    inp = int(usage.get("input") or 0)
    out_tok = int(usage.get("output") or 0)
    total_api = int(usage.get("totalTokens") or 0)
    estimated = False
    if inp == 0 and out_tok == 0:
        estimated = True
        prompt_msgs = _messages_without_trailing_assistant(messages)
        inp = max(
            1,
            (estimate_context_chars(system, prompt_msgs) + _tool_schema_chars(tools)) // 4,
        )
        comp_chars = len(extract_text_from_message(final_message))
        for block in final_message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                comp_chars += len(json.dumps(block.get("arguments") or {}, ensure_ascii=False))
        out_tok = max(1, comp_chars // 4)
        total_api = inp + out_tok
    elif total_api == 0:
        total_api = inp + out_tok

    tag = f"{DIM}(estimated){RESET}" if estimated else f"{DIM}(API){RESET}"
    logger.info(
        f"{CYAN}{BOLD}[round {round_no}] tokens{RESET} {tag}: "
        f"{GREEN}prompt {inp}{RESET} + {BLUE}completion {out_tok}{RESET} = "
        f"{MAGENTA}{BOLD}total {total_api}{RESET}"
    )


async def run_agent(
    mgr: AgentManager,
    agent_id: str,
    session_key: str,
    user_text: str,
    on_typing: Any = None,
    on_event: Any = None,
    channel: str = "terminal",
    abort_signal: Any | None = None,
) -> str:
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(4)
    agent = mgr.get_agent(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' not found"
    messages = mgr.get_session(session_key, agent_id=agent_id)
    messages.append({"role": "user", "content": user_text, "timestamp": int(time.time() * 1000)})

    # Build dynamic per-turn system prompt for this agent.
    system_prompt = agent.build_system_prompt_for_agent(
        channel=channel,
        last_user_message=user_text,
        session_key=session_key,
    )

    async with _agent_semaphore:
        if on_typing:
            on_typing(agent_id, True)
        try:
            tool_ctx = {
                "agent_id": agent_id,
                "channel": channel,
                "session_key": session_key,
                "role": getattr(agent, "role", "general"),
                "_mgr": mgr,
            }
            return await _run_agent(
                agent.model,
                system_prompt,
                messages,
                tool_ctx=tool_ctx,
                on_event=on_event,
                abort_signal=abort_signal,
            )
        finally:
            if on_typing:
                on_typing(agent_id, False)
            try:
                from .session_store import save_chat_session

                save_chat_session(agent_id, session_key, messages)
            except Exception:
                logger.debug("save_chat_session failed", exc_info=True)


async def _execute_tool_calls(
    assistant_message: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    tool_ctx: dict[str, Any] | None = None,
    abort_signal: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Execute all toolCall blocks from an assistant message and emit tool events."""
    content = assistant_message.get("content") or []
    tool_calls = [c for c in content if isinstance(c, dict) and c.get("type") == "toolCall"]

    # Providers/streams may duplicate the same toolCall block (same id). De-dup by id while
    # preserving order to prevent repeated execution and log spam.
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for c in tool_calls:
        tid = str(c.get("id", "") or "")
        if tid and tid in seen_ids:
            continue
        if tid:
            seen_ids.add(tid)
        deduped.append(c)

    for block in deduped:
        if abort_requested(abort_signal):
            break
        name = str(block.get("name", "") or "").strip()
        bid = block.get("id", "")
        args = block.get("arguments", {}) or {}

        yield {
            "type": "tool_call_start",
            "tool_call_id": bid,
            "tool_name": name,
            "args": args,
        }

        logger.info(f"{YELLOW}calling tool: {name} with args: {args}{RESET}")
        is_error = False
        try:
            body = process_tool_call(name, args, tool_ctx=tool_ctx)
            if isinstance(body, str) and body.startswith("Error:"):
                is_error = True
        except Exception as exc:
            body = f"Error: {exc}"
            is_error = True

        output_body = body
        shrink_method = "none"
        if isinstance(body, str) and len(body) > TOOL_RESULT_MAX_CHARS:
            output_body, shrink_method = await shrink_tool_result_text(
                tool_name=name,
                raw_text=body,
                user_question=latest_user_text(messages),
            )

        tool_result_msg: dict[str, Any] = {
            "role": "toolResult",
            "toolCallId": bid,
            "toolName": name,
            "content": [{"type": "text", "text": output_body}],
            "details": {
                "shrink_method": shrink_method,
            },
            "isError": is_error,
            "timestamp": int(time.time() * 1000),
        }
        messages.append(tool_result_msg)

        yield {
            "type": "tool_call_end",
            "tool_call_id": bid,
            "tool_name": name,
            "result": tool_result_msg,
            "is_error": is_error,
        }


## Agent Loop (event streaming style)
#
# Layers:
# - model_start(run_begin) / model_end — run & round boundaries (when tools / run_done).
# - message_start / message_update / message_end — assistant message streaming (mirrors model.stream).
# - tool_call_start / tool_call_end — tool execution.
async def _agent_loop(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    tool_ctx: dict[str, Any] | None = None,
    abort_signal: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream model/tool events while mutating messages in place."""
    model = _get_model_for_id(model_id)
    yield {"type": "model_start", "run_begin": True}

    for round_no in range(1, MAX_AGENT_TOOL_ROUNDS + 1):
        if abort_requested(abort_signal):
            yield {
                "type": "model_end",
                "run_done": True,
                "messages": messages,
                "aborted": True,
            }
            return

        validate_session_messages(messages)

        await maybe_trim_history_for_budget(model_id, system, messages, tool_ctx=tool_ctx)

        # 判断 message中 最后一条是否是 toolResult, 如果有，添加 toolresutl处理的prompt.
        context = Context(messages=messages, system_prompt=system, tools=TOOLS_LLM)
        logger.info(f"System Prompt:\n {context.system_prompt}")

        stream_opts: dict[str, Any] = {"max_tokens": 40960}
        if abort_signal is not None:
            stream_opts["signal"] = abort_signal

        partial: dict[str, Any] | None = None
        final_message: dict[str, Any] | None = None
        added_partial = False

        async for event in model.stream(context, stream_opts):
            ev_type = event.get("type")
            if ev_type == "start":
                partial = event.get("partial") or {}
                messages.append(partial)
                added_partial = True
                yield {"type": "message_start", "message": dict(partial)}
            elif ev_type in (
                "text_delta",
                "text_start",
                "text_end",
                "thinking_start",
                "thinking_delta",
                "thinking_end",
                "toolcall_start",
                "toolcall_delta",
                "toolcall_end",
            ):
                partial = event.get("partial")
                if partial is not None:
                    if added_partial:
                        messages[-1] = partial
                    out: dict[str, Any] = {
                        "type": "message_update",
                        "message": dict(partial),
                        "assistant_message_event": event,
                    }
                    if ev_type == "text_delta":
                        out["delta"] = event.get("delta", "")
                    yield out
            elif ev_type == "done":
                final_message = event.get("message")
                if final_message is not None:
                    if added_partial:
                        messages[-1] = final_message
                    else:
                        messages.append(final_message)
                    yield {"type": "message_end", "message": final_message}
                break
            elif ev_type == "error":
                err_msg = event.get("error") or {}
                final_message = err_msg if isinstance(err_msg, dict) else {"errorMessage": str(err_msg)}
                if added_partial:
                    messages[-1] = final_message
                else:
                    messages.append(final_message)
                yield {"type": "message_end", "message": final_message}
                aborted = final_message.get("stopReason") == "aborted" or event.get("reason") == "aborted"
                yield {
                    "type": "model_end",
                    "message": final_message,
                    "tool_results": [],
                    "run_done": True,
                    "messages": messages,
                    **({"aborted": True} if aborted else {}),
                }
                return

        if final_message is None:
            break

        _log_round_token_usage(round_no, system, messages, final_message, tools=TOOLS_LLM)

        stop_reason = final_message.get("stopReason") or "stop"
        if stop_reason in ("error", "aborted"):
            yield {
                "type": "model_end",
                "message": final_message,
                "tool_results": [],
                "run_done": True,
                "messages": messages,
                **({"aborted": True} if stop_reason == "aborted" else {}),
            }
            return

        if abort_requested(abort_signal):
            yield {
                "type": "model_end",
                "message": final_message,
                "tool_results": [],
                "run_done": True,
                "messages": messages,
                "aborted": True,
            }
            return

        content_list = final_message.get("content") or []
        logger.info(f"{YELLOW}content_list: {content_list}{RESET}")

        tool_calls = [c for c in content_list if isinstance(c, dict) and c.get("type") == "toolCall"]
        if not tool_calls:
            yield {
                "type": "model_end",
                "message": final_message,
                "tool_results": [],
                "run_done": True,
                "messages": messages,
            }
            return

        tool_results: list[dict[str, Any]] = []
        async for tool_event in _execute_tool_calls(
            final_message, messages, tool_ctx=tool_ctx, abort_signal=abort_signal
        ):
            if tool_event.get("type") == "tool_call_end":
                tool_msg = tool_event.get("result")
                if isinstance(tool_msg, dict) and tool_msg.get("role") == "toolResult":
                    tool_results.append(tool_msg)
            yield tool_event

        if abort_requested(abort_signal):
            yield {
                "type": "model_end",
                "message": final_message,
                "tool_results": tool_results,
                "run_done": True,
                "messages": messages,
                "aborted": True,
            }
            return

        yield {
            "type": "model_end",
            "message": final_message,
            "tool_results": tool_results,
            "run_done": False,
            "messages": messages,
        }

    yield {"type": "model_end", "run_done": True, "messages": messages}


async def _run_agent(
    model_id: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    tool_ctx: dict[str, Any] | None = None,
    on_event: Any = None,
    abort_signal: Any | None = None,
) -> str:
    """
    Compatibility wrapper:
    consume event stream and return final assistant text.
    """
    last_assistant_message: dict[str, Any] | None = None
    aborted = False
    try:
        async for event in _agent_loop(
            model_id, system, messages, tool_ctx=tool_ctx, abort_signal=abort_signal
        ):
            if on_event:
                try:
                    on_event(event)
                except Exception:
                    # Upstream event consumer failures must not break agent execution.
                    pass
            et = event.get("type")
            if et == "model_end" and event.get("aborted"):
                aborted = True
            if et in ("message_end", "model_end"):
                msg = event.get("message")
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    last_assistant_message = msg
    except Exception as exc:
        while messages and messages[-1].get("role") != "user":
            messages.pop()
        if messages:
            messages.pop()
        return f"API Error: {exc}"

    text = extract_text_from_message(last_assistant_message)
    if text:
        return text
    if aborted:
        return "[aborted]"
    if last_assistant_message is None:
        return "[no response]"
    stop_reason = last_assistant_message.get("stopReason")
    if stop_reason:
        return f"[stop={stop_reason}]"
    if last_assistant_message.get("errorMessage"):
        return str(last_assistant_message.get("errorMessage"))
    return "[no text]"
