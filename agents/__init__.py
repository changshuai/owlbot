"""
Python agent: stateful agent with tool execution and event streaming.
Built on ai_models (Model, Context, stream/invoke).
"""
from __future__ import annotations

from .agent import Agent
from .agent_loop import agent_loop
from .simple_agent import SimpleAgent
from .types import (
    AgentContext,
    AgentEvent,
    AgentTool,
    AgentToolResult,
    make_tool_result_message,
    make_user_message,
)

__all__ = [
    "Agent",
    "SimpleAgent",
    "AgentContext",
    "AgentEvent",
    "AgentTool",
    "AgentToolResult",
    "agent_loop",
    "make_tool_result_message",
    "make_user_message",
]
