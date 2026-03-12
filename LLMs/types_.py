"""
Unified types for the ai_models layer.
Aligns with packages/ai TypeScript types where practical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Union

# --- API / Provider identifiers ---
Api = str
Provider = str

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

# --- Content blocks ---
class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ThinkingContent(TypedDict):
    type: Literal["thinking"]
    thinking: str


class ImageContent(TypedDict):
    type: Literal["image"]
    data: str  # base64
    mimeType: str


class ToolCallContent(TypedDict):
    type: Literal["toolCall"]
    id: str
    name: str
    arguments: dict[str, Any]


# --- Messages ---
class UserMessage(TypedDict):
    role: Literal["user"]
    content: str | list[TextContent | ImageContent]
    timestamp: int


class Usage(TypedDict):
    input: int
    output: int
    cacheRead: int
    cacheWrite: int
    totalTokens: int
    cost: dict[str, float]


StopReason = Literal["stop", "length", "toolUse", "error", "aborted"]


class AssistantMessage(TypedDict, total=False):
    role: Literal["assistant"]
    content: list[TextContent | ThinkingContent | ToolCallContent]
    api: Api
    provider: Provider
    model: str
    usage: Usage
    stopReason: StopReason
    errorMessage: str
    timestamp: int


class ToolResultMessage(TypedDict):
    role: Literal["toolResult"]
    toolCallId: str
    toolName: str
    content: list[TextContent | ImageContent]
    details: dict[str, Any]
    isError: bool
    timestamp: int


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]

# --- Tool (for context) ---
class Tool(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


# --- Context ---
@dataclass
class Context:
    messages: list[Message]
    system_prompt: str = ""
    tools: list[Tool] | None = None


# --- Stream options ---
class StreamOptions(TypedDict, total=False):
    temperature: float
    max_tokens: int
    signal: Any  # AbortSignal-like
    session_id: str
    headers: dict[str, str]
    max_retry_delay_ms: int


class SimpleStreamOptions(StreamOptions, total=False):
    reasoning: ThinkingLevel
    thinking_budgets: dict[str, int]
