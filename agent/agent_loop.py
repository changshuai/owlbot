from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from LLMs import Context, get_env_api_key, get_model
from .agent_ import AgentManager, Agent
from .tools import TOOLS_LLM, process_tool_call
from common.colors import DIM, RESET, BOLD, GREEN, YELLOW

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
MODEL_ID = os.getenv("MODEL_ID", "deepseek/deepseek-chat")

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
    channel: str = "terminal",
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
            return await _agent_loop(agent.model, system_prompt, messages, tool_ctx=tool_ctx)
        finally:
            if on_typing:
                on_typing(agent_id, False)

## Agent Loop
async def _agent_loop(
    model_id: str,
    system: str,
    messages: list[dict],
    *,
    tool_ctx: dict[str, Any] | None = None,
) -> str:
    """messages is list[Message] in Context format; mutated in place."""
    model = _get_model_for_id(model_id)
    for _ in range(15):
        try:
            context = Context(
                messages=messages,
                system_prompt=system,
                tools=TOOLS_LLM,
            )
            # for tool in context.tools:
                # logger.info(f" {tool}")
            logger.info(f"System Prompt:\n {context.system_prompt}")
            response = await model.invoke(context, {"max_tokens": 40960})

        except Exception as exc:
            while messages and messages[-1].get("role") != "user":
                messages.pop()
            if messages:
                messages.pop()
            return f"API Error: {exc}"
        content_list = response.get("content") or []

        logger.info(f"{YELLOW}content_list: {content_list}{RESET}")

        assistant_msg: dict = {
            "role": "assistant",
            "content": content_list,
            "timestamp": int(time.time() * 1000),
        }
        messages.append(assistant_msg)
        stop_reason = response.get("stopReason", "stop")

        if stop_reason in ("stop", "end_turn"):
            text = "".join(b.get("text", "") for b in content_list if b.get("type") == "text")
            
            # logger.info(f"{GREEN}Assistant Response:\n{text}{RESET}")

            return text or "[no text]"
        if stop_reason == "toolUse":
            for block in content_list:
                if block.get("type") != "toolCall":
                    continue
                name = block.get("name", "")
                bid = block.get("id", "")
                args = block.get("arguments", {}) or {}

                logger.info(f"{YELLOW}calling tool: {name} with args: {args}{RESET}")
                # print(f"  {DIM}[tool: {name}]{RESET}")
                body = process_tool_call(name, args, tool_ctx=tool_ctx)
                messages.append({
                    "role": "toolResult",
                    "toolCallId": bid,
                    "toolName": name,
                    "content": [{"type": "text", "text": body}],
                    "details": {},
                    "isError": False,
                    "timestamp": int(time.time() * 1000),
                })
            continue
        text = "".join(b.get("text", "") for b in content_list if b.get("type") == "text")
        return text or f"[stop={stop_reason}]"
    return "[max iterations reached]"
