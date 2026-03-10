"""
SimpleAgent: small stateful agent built on top of agent_loop.

Features:
- Tracks system prompt and message history.
- Supports basic tool execution via AgentTool.
- Exposes a simple prompt() returning the final assistant message.

No steering/follow-up queues or rich AgentEvent state – this is a thinner,
less opinionated wrapper than the full Agent class.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List

from ai_models import SimpleStreamOptions
from ai_models.modelbase import Model
from ai_models.types import Message

from .agent_loop import agent_loop
from .types import AgentContext


def _make_user_message(text: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": text,
        "timestamp": int(time.time() * 1000),
    }


@dataclass
class SimpleAgent:
    """
    Minimal stateful agent that uses agent_loop for a single-model, single-context chat.

    - Keeps a list of messages (user / assistant / toolResult) in ai_models.Message format.
    - Supports tools via the same AgentTool/Context wiring as the full Agent.
    - Does not expose AgentEvent; prompt() hides the loop and only returns the final assistant message.
    """

    system_prompt: str
    model: Model
    tools: list[Any] = field(default_factory=list)
    messages: List[Message] = field(default_factory=list)

    async def prompt(
        self,
        text: str,
        options: SimpleStreamOptions | None = None,
    ) -> Message:
        """
        Append a user message, run a full agent_loop turn (including tool calls),
        and return the final assistant message. History is updated in-place.
        """
        user = _make_user_message(text)
        context = AgentContext(
            messages=list(self.messages),
            system_prompt=self.system_prompt,
            tools=self.tools or None,
        )
        opts = options or {}

        async for event in agent_loop([user], context, self.model, opts):
            if event.get("type") == "agent_end":
                msgs = event.get("messages") or []
                # Update history
                self.messages = list(msgs)  # type: ignore[list-item]
                break

        # Find the last assistant message in history
        for msg in reversed(self.messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg  # type: ignore[return-value]

        raise RuntimeError("SimpleAgent: no assistant message produced")

    async def stream(
        self,
        text: str,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream AgentEvents for a single prompt.

        This is a thin wrapper over agent_loop; the caller receives AgentEvent dicts
        (agent_start/turn_start/message_*/tool_execution_*/turn_end/agent_end).
        On agent_end, history is updated.
        """
        user = _make_user_message(text)
        context = AgentContext(
            messages=list(self.messages),
            system_prompt=self.system_prompt,
            tools=self.tools or None,
        )
        opts = options or {}

        async for event in agent_loop([user], context, self.model, opts):
            if event.get("type") == "agent_end":
                msgs = event.get("messages") or []
                self.messages = list(msgs)  # type: ignore[list-item]
            yield event

    def clear_messages(self) -> None:
        """Drop all conversation history."""
        self.messages.clear()

