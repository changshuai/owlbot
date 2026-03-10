import asyncio, os, threading, time
from typing import Any

from LLMs import Context, get_env_api_key, get_model
from .agent_manager import AgentManager, DIM, RESET
from .tools import TOOLS_LLM, process_tool_call

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
MODEL_ID = os.getenv("MODEL_ID", "deepseek/deepseek-chat")

_agent_semaphore: asyncio.Semaphore | None = None


def _get_model_for_id(model_id: str):
    """Return LLM Model for the given model id (uses MODEL_PROVIDER for provider)."""
    return get_model(MODEL_PROVIDER, model_id, api_key=get_env_api_key(MODEL_PROVIDER))


## Timestamp
def _ts() -> int:
    return int(time.time() * 1000)


## Run Agent
async def run_agent(mgr: AgentManager, agent_id: str, session_key: str,
                    user_text: str, on_typing: Any = None) -> str:
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(4)
    agent = mgr.get_agent(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' not found"
    messages = mgr.get_session(session_key)
    messages.append({"role": "user", "content": user_text, "timestamp": _ts()})
    async with _agent_semaphore:
        if on_typing:
            on_typing(agent_id, True)
        try:
            return await _agent_loop(agent.effective_model, agent.system_prompt(), messages)
        finally:
            if on_typing:
                on_typing(agent_id, False)

## Agent Loop
async def _agent_loop(model_id: str, system: str, messages: list[dict]) -> str:
    """messages is list[Message] in Context format; mutated in place."""
    model = _get_model_for_id(model_id)
    for _ in range(15):
        try:
            context = Context(
                messages=messages,
                system_prompt=system,
                tools=TOOLS_LLM,
            )
            response = await model.invoke(context, {"max_tokens": 40960})
        except Exception as exc:
            while messages and messages[-1].get("role") != "user":
                messages.pop()
            if messages:
                messages.pop()
            return f"API Error: {exc}"
        content_list = response.get("content") or []
        assistant_msg: dict = {
            "role": "assistant",
            "content": content_list,
            "timestamp": _ts(),
        }
        messages.append(assistant_msg)
        stop_reason = response.get("stopReason", "stop")
        if stop_reason in ("stop", "end_turn"):
            text = "".join(b.get("text", "") for b in content_list if b.get("type") == "text")
            return text or "[no text]"
        if stop_reason == "toolUse":
            for block in content_list:
                if block.get("type") != "toolCall":
                    continue
                name = block.get("name", "")
                bid = block.get("id", "")
                args = block.get("arguments", {}) or {}
                print(f"  {DIM}[tool: {name}]{RESET}")
                body = process_tool_call(name, args)
                messages.append({
                    "role": "toolResult",
                    "toolCallId": bid,
                    "toolName": name,
                    "content": [{"type": "text", "text": body}],
                    "details": {},
                    "isError": False,
                    "timestamp": _ts(),
                })
            continue
        text = "".join(b.get("text", "") for b in content_list if b.get("type") == "text")
        return text or f"[stop={stop_reason}]"
    return "[max iterations reached]"

