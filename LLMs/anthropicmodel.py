"""
Anthropic model: stream/invoke via Anthropic Messages SDK.
"""
from typing import Any, AsyncIterator

from .envapikeys import get_env_api_key
from .modelbase import Model, make_empty_assistant_message
from .types_ import Context, SimpleStreamOptions


def _messages_to_anthropic(context: Context) -> tuple[str | None, list[dict[str, Any]]]:
    system: str | None = context.system_prompt or None
    messages: list[dict[str, Any]] = []
    for msg in context.messages:
        if msg["role"] == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                blocks = []
                for part in content:
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image":
                        blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.get("mimeType", "image/png"),
                                "data": part["data"],
                            },
                        })
                if blocks:
                    messages.append({"role": "user", "content": blocks})
        elif msg["role"] == "assistant":
            content = msg.get("content", [])
            blocks = []
            for p in content:
                if p.get("type") == "text":
                    blocks.append({"type": "text", "text": p.get("text", "")})
                elif p.get("type") == "toolCall":
                    blocks.append({
                        "type": "tool_use",
                        "id": p["id"],
                        "name": p["name"],
                        "input": p.get("arguments") or {},
                    })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
        elif msg["role"] == "toolResult":
            text = ""
            if isinstance(msg.get("content"), list):
                text = "".join(
                    p.get("text", "") for p in msg["content"] if p.get("type") == "text"
                )
            else:
                text = str(msg.get("content", ""))
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg["toolCallId"],
                    "content": text,
                    "is_error": msg.get("isError", False),
                }],
            })
    return system, messages


def _tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in tools
    ]


class AnthropicModel(Model):
    """Model that calls Anthropic Messages API."""

    async def stream(
        self,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ImportError("Anthropic model requires: pip install anthropic") from e

        api_key = self.api_key or get_env_api_key(self.provider)
        if not api_key:
            raise ValueError(f"No API key for provider: {self.provider}")

        client = AsyncAnthropic(api_key=api_key)
        system, messages = _messages_to_anthropic(context)
        anthropic_tools = _tools_to_anthropic(context.tools)
        kwargs: dict[str, Any] = {
            "model": self.id,
            "messages": messages,
            "max_tokens": options.get("max_tokens") or min(self.max_tokens, 4096),
            "temperature": options.get("temperature"),
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            kwargs["tool_choice"] = "auto"

        output = make_empty_assistant_message(self.api, self.provider, self.id)
        abort_signal = options.get("signal")
        yield {"type": "start", "partial": dict(output)}

        try:
            async with client.messages.stream(**kwargs) as api_stream:
                content_index = 0
                async for event in api_stream:
                    if abort_signal is not None and getattr(abort_signal, "is_set", None) and abort_signal.is_set():
                        output["stopReason"] = "aborted"
                        yield {"type": "error", "reason": "aborted", "error": output}
                        return
                    if getattr(event, "type", None) == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "text":
                            content_index = len(output["content"])
                    elif getattr(event, "type", None) == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", None) == "text_delta":
                            text = getattr(delta, "text", None) or ""
                            if not any(c.get("type") == "text" for c in output["content"]):
                                output["content"].append({"type": "text", "text": text})
                            else:
                                for c in output["content"]:
                                    if c.get("type") == "text":
                                        c["text"] = c.get("text", "") + text
                                        break
                            yield {
                                "type": "text_delta",
                                "contentIndex": content_index,
                                "delta": text,
                                "partial": dict(output),
                            }
                    elif getattr(event, "type", None) == "message_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "stop_reason", None):
                            output["stopReason"] = (
                                "toolUse" if getattr(delta, "stop_reason") == "tool_use" else "stop"
                            )
                final = await api_stream.get_final_message()
                for block in getattr(final, "content", []) or []:
                    if getattr(block, "type", None) == "tool_use":
                        output["content"].append({
                            "type": "toolCall",
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "arguments": getattr(block, "input", {}) or {},
                        })
            yield {"type": "done", "reason": output.get("stopReason", "stop"), "message": output}
        except Exception as e:
            output["stopReason"] = "error"
            output["errorMessage"] = str(e)
            yield {"type": "error", "reason": "error", "error": output}
