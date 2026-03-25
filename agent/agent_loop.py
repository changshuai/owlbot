from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator

from LLMs import Context, get_env_api_key, get_model
from .agent_ import AgentManager
from .agent_abort import abort_requested, AgentAbortController
from .message_validator import extract_text_from_message, validate_session_messages
from .tools import TOOLS_LLM, process_tool_call
from common.colors import DIM, RESET, BOLD, GREEN, YELLOW

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
    """Return LLM Model for the given model id (uses MODEL_PROVIDER for provider)."""
    return get_model(MODEL_PROVIDER, model_id, api_key=get_env_api_key(MODEL_PROVIDER))

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
    messages = mgr.get_session(session_key)
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
    for block in tool_calls:
        if abort_requested(abort_signal):
            break
        name = block.get("name", "")
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

        tool_result_msg: dict[str, Any] = {
            "role": "toolResult",
            "toolCallId": bid,
            "toolName": name,
            "content": [{"type": "text", "text": body}],
            "details": {},
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

    for _ in range(MAX_AGENT_TOOL_ROUNDS):
        if abort_requested(abort_signal):
            yield {
                "type": "model_end",
                "run_done": True,
                "messages": messages,
                "aborted": True,
            }
            return

        validate_session_messages(messages)

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
