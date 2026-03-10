"""
Agent loop: stream assistant response, execute tool calls, yield events.
Uses ai_models Model and Context.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable

from ai_models import Context, SimpleStreamOptions
from ai_models.modelbase import Model

from .types import AgentContext, AgentEvent, AgentToolResult, make_tool_result_message, validate_tool_arguments


def _tools_to_llm(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert AgentTool list to ai_models Context.tools format (name, description, parameters)."""
    if not tools:
        return None
    return [
        {"name": t.name, "description": t.description, "parameters": getattr(t, "parameters", {})}
        for t in tools
    ]


async def _execute_tool_calls(
    tools: list[Any] | None,
    assistant_message: dict[str, Any],
) -> AsyncIterator[tuple[AgentEvent, dict[str, Any] | None]]:
    """
    Execute tool calls from an assistant message. Yields tool_execution_start/end and
    returns (event, tool_result_message) for each call. Caller appends tool_result_message to context.
    """
    content = assistant_message.get("content") or []
    tool_calls = [c for c in content if isinstance(c, dict) and c.get("type") == "toolCall"]

    for tc in tool_calls:
        tool_call_id = tc.get("id") or ""
        tool_name = tc.get("name") or ""
        args = tc.get("arguments") or {}

        yield (
            {
                "type": "tool_execution_start",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args": args,
            },
            None,
        )

        tool = None
        if tools:
            for t in tools:
                if getattr(t, "name", None) == tool_name:
                    tool = t
                    break

        result: AgentToolResult
        is_error = False
        try:
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            validated_args = validate_tool_arguments(tool, tc)
            result = await tool.execute(tool_call_id, validated_args)
        except Exception as e:
            result = {
                "content": [{"type": "text", "text": str(e)}],
                "details": {},
            }
            is_error = True

        tool_result_msg = make_tool_result_message(
            tool_call_id,
            tool_name,
            result["content"],
            result["details"],
            is_error=is_error,
        )

        yield (
            {
                "type": "tool_execution_end",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "is_error": is_error,
            },
            tool_result_msg,
        )


async def agent_loop(
    prompts: list[dict[str, Any]],
    context: AgentContext,
    model: Model,
    options: SimpleStreamOptions | None = None,
    *,
    transform_context: Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]] | None = None,
    stream_fn: Callable[[Model, Context, SimpleStreamOptions], AsyncIterator[dict[str, Any]]] | None = None,
    signal: Any | None = None,
    get_steering_messages: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
    get_followup_messages: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
) -> AsyncIterator[AgentEvent]:
    """
    Run the agent loop: add prompts to context, stream assistant response, execute tool calls,
    append tool results, repeat until no more tool calls. Yields agent events.

    transform_context, if provided, is applied to the messages list before each LLM call.
    stream_fn, if provided, is used instead of model.stream(model_context, options).
    """
    opts: SimpleStreamOptions = dict(options or {})
    if signal is not None:
        opts["signal"] = signal

    current_messages = list(context.messages)
    pending_prompts: list[dict[str, Any]] = list(prompts)

    yield {"type": "agent_start"}

    # Outer loop: process initial prompts and any follow-up batches without recursion.
    while True:
        if pending_prompts:
            current_messages.extend(pending_prompts)
            yield {"type": "turn_start"}
            for prompt in pending_prompts:
                yield {"type": "message_start", "message": prompt}
                yield {"type": "message_end", "message": prompt}
            pending_prompts = []

        # Inner loop: one or more LLM turns (stream + tool execution + optional steering).
        while True:
            # Build current context for this turn (agent-level)
            turn_context = AgentContext(
                messages=list(current_messages),
                system_prompt=context.system_prompt,
                tools=context.tools,
            )
            # Apply optional context transform before calling the model
            turn_messages: list[dict[str, Any]] = list(turn_context.messages)
            if transform_context is not None:
                turn_messages = await transform_context(turn_messages)

            llm_context = Context(
                messages=turn_messages,
                system_prompt=turn_context.system_prompt,
                tools=_tools_to_llm(turn_context.tools),
            )

            partial: dict[str, Any] | None = None
            added_partial = False
            final_message: dict[str, Any] | None = None

            stream_iter = stream_fn(model, llm_context, opts) if stream_fn is not None else model.stream(llm_context, opts)
            async for event in stream_iter:
                ev_type = event.get("type")
                if ev_type == "start":
                    partial = event.get("partial") or {}
                    current_messages.append(partial)
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
                            current_messages[-1] = partial
                        yield {"type": "message_update", "message": dict(partial), "assistant_message_event": event}
                elif ev_type == "done":
                    final_message = event.get("message")
                    if final_message is not None:
                        if added_partial:
                            current_messages[-1] = final_message
                        else:
                            current_messages.append(final_message)
                        if not added_partial:
                            yield {"type": "message_start", "message": dict(final_message)}
                        yield {"type": "message_end", "message": final_message}
                    break
                elif ev_type == "error":
                    err_msg = event.get("error") or {}
                    final_message = err_msg
                    if added_partial:
                        current_messages[-1] = final_message
                    else:
                        current_messages.append(final_message)
                    yield {"type": "message_end", "message": final_message}
                    yield {"type": "turn_end", "message": final_message, "tool_results": []}
                    yield {"type": "agent_end", "messages": current_messages}
                    return

            if final_message is None:
                break

            stop_reason = final_message.get("stopReason") or "stop"
            if stop_reason in ("error", "aborted"):
                yield {"type": "turn_end", "message": final_message, "tool_results": []}
                yield {"type": "agent_end", "messages": current_messages}
                return

            tool_calls = [
                c for c in (final_message.get("content") or []) if isinstance(c, dict) and c.get("type") == "toolCall"
            ]
            if not tool_calls:
                yield {"type": "turn_end", "message": final_message, "tool_results": []}
                break

            tool_results: list[dict[str, Any]] = []
            async for ev_and_msg in _execute_tool_calls(context.tools, final_message):
                ev, tool_result_msg = ev_and_msg
                yield ev
                if tool_result_msg is not None:
                    tool_results.append(tool_result_msg)
                    current_messages.append(tool_result_msg)
                    yield {"type": "message_start", "message": tool_result_msg}
                    yield {"type": "message_end", "message": tool_result_msg}

            yield {"type": "turn_end", "message": final_message, "tool_results": tool_results}

            # Steering: if there are queued steering messages, inject them before the next LLM call.
            if get_steering_messages is not None:
                steering = await get_steering_messages()
                if steering:
                    for msg in steering:
                        current_messages.append(msg)
                        yield {"type": "message_start", "message": msg}
                        yield {"type": "message_end", "message": msg}
                    yield {"type": "turn_start"}
                    continue

            yield {"type": "turn_start"}

        # Inner loop ended (no more tool calls). Check for follow-up messages.
        if get_followup_messages is not None:
            followups = await get_followup_messages()
            if followups:
                pending_prompts = followups
                continue
        yield {"type": "agent_end", "messages": current_messages}
        break

