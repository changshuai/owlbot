"""
Agent types: tools, events, context.
Aligns with packages/agent TypeScript types where practical.
"""

import inspect
import time
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, Union, get_args, get_origin, get_type_hints

# Re-export Message types from ai_models for agent use
from ai_models.types import Message, ToolResultMessage

try:
    import jsonschema
    from jsonschema import ValidationError
except ImportError:  # pragma: no cover - optional dependency
    jsonschema = None  # type: ignore[assignment]
    ValidationError = Exception  # type: ignore[assignment]


class AgentToolResult(TypedDict):
    """Result of a tool execution. content is list of text/image blocks; details for UI/logging."""

    content: list[dict[str, Any]]  # [{ "type": "text", "text": "..." }] or image
    details: dict[str, Any]


class AgentTool(Protocol):
    """Tool that the agent can call. Must have name, description, parameters, and execute()."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
    ) -> AgentToolResult:
        """Run the tool. Raise an error on failure; the agent will report it as a tool error."""
        ...


@dataclass
class AgentContext:
    """Context for the agent loop: messages, system prompt, optional tools."""

    messages: list[Message]
    system_prompt: str = ""
    tools: list[Any] | None = None  # list[AgentTool] without importing AgentTool at runtime


# --- Agent events (for subscribers) ---

AgentEvent = dict[str, Any]
# Known event types:
#   agent_start: {}
#   agent_end: { messages: list[Message] }
#   turn_start: {}
#   turn_end: { message: Message, tool_results: list[ToolResultMessage] }
#   message_start: { message: Message }
#   message_update: { message: Message (partial), assistant_message_event: dict }
#   message_end: { message: Message }
#   tool_execution_start: { tool_call_id, tool_name, args }
#   tool_execution_end: { tool_call_id, tool_name, result: AgentToolResult, is_error: bool }


def make_user_message(content: str) -> dict[str, Any]:
    """Build a user message for the agent."""
    return {"role": "user", "content": content, "timestamp": int(time.time() * 1000)}


def make_tool_result_message(
    tool_call_id: str,
    tool_name: str,
    content: list[dict[str, Any]],
    details: dict[str, Any],
    is_error: bool = False,
) -> ToolResultMessage:
    """Build a tool result message for the agent context."""
    return {
        "role": "toolResult",
        "toolCallId": tool_call_id,
        "toolName": tool_name,
        "content": content,
        "details": details,
        "isError": is_error,
        "timestamp": int(time.time() * 1000),
    }


def validate_tool_arguments(tool: AgentTool, tool_call: dict[str, Any]) -> dict[str, Any]:
    """
    Validate tool_call['arguments'] against tool.parameters using JSON Schema, if available.

    This relies on the tool.parameters dict being a valid JSON Schema. If jsonschema is not
    installed or the parameters are not a dict, arguments are returned as-is.
    """
    schema = getattr(tool, "parameters", None)
    args = tool_call.get("arguments") or {}
    if not isinstance(schema, dict) or jsonschema is None:
        return args

    try:
        jsonschema.validate(instance=args, schema=schema)  # type: ignore[arg-type]
    except ValidationError as e:
        # Normalize jsonschema's error into a simpler ValueError message
        raise ValueError(f"Invalid arguments for tool {tool.name}: {e.message}") from e

    return args


def _annotation_to_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """
    Map a Python type annotation to a JSON Schema fragment.
    Returns (schema, is_optional).
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] / Union[T, None]
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if len(non_none) == 1 and len(args) == 2:
            inner_schema, _ = _annotation_to_schema(non_none[0])
            return inner_schema, True

    if annotation is str:
        return {"type": "string"}, False
    if annotation is int:
        return {"type": "integer"}, False
    if annotation is float:
        return {"type": "number"}, False
    if annotation is bool:
        return {"type": "boolean"}, False

    if origin in (list, tuple):
        item_ann = args[0] if args else Any
        item_schema, _ = _annotation_to_schema(item_ann)
        return {"type": "array", "items": item_schema}, False

    # Fallback: no specific constraints
    return {}, False


def agent_tool(name: str | None = None, description: str | None = None) -> Any:
    """
    Decorator to turn an async function into an AgentTool instance.

    Example:
        @agent_tool()
        async def echo(message: str) -> AgentToolResult:
            \"\"\"Echo back the message.\"\"\"
            return {
                \"content\": [{\"type\": \"text\", \"text\": message}],
                \"details\": {},
            }
    """

    def decorator(fn: Any) -> AgentTool:
        tool_name = name or fn.__name__
        doc = inspect.getdoc(fn) or ""
        first_line = doc.splitlines()[0] if doc else ""
        tool_description = description or first_line

        sig = inspect.signature(fn)
        # Use typing.get_type_hints for Python 3.9 compatibility
        try:
            type_hints = get_type_hints(fn)
        except Exception:
            type_hints = {}

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            ann = type_hints.get(param_name, Any)
            schema, is_optional = _annotation_to_schema(ann)
            properties[param_name] = schema
            if param.default is inspect._empty and not is_optional:
                required.append(param_name)

        parameters_schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters_schema["required"] = required

        class FunctionTool:
            name = tool_name
            description = tool_description
            parameters = parameters_schema

            async def execute(self, tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
                # The wrapped function is responsible for returning a valid AgentToolResult.
                return await fn(**params)

        return FunctionTool()  # type: ignore[return-value]

    return decorator
