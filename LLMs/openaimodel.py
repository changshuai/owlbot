"""
OpenAI model: stream/invoke via OpenAI chat completions SDK.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from .envapikeys import get_env_api_key
from .modelbase import Model, make_empty_assistant_message
from .types_ import Context, SimpleStreamOptions


def _messages_to_openai(context: Context) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if context.system_prompt:
        messages.append({"role": "system", "content": context.system_prompt})
    for msg in context.messages:
        if msg["role"] == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image":
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{part['mimeType']};base64,{part['data']}"},
                        })
                if parts:
                    messages.append({"role": "user", "content": parts})
        elif msg["role"] == "assistant":
            parts = [p for p in msg.get("content", []) if p.get("type") == "text"]
            text = "".join(p.get("text", "") for p in parts)
            if text:
                messages.append({"role": "assistant", "content": text})
            tool_calls = [p for p in msg.get("content", []) if p.get("type") == "toolCall"]
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc.get("arguments") or {})},
                        }
                        for tc in tool_calls
                    ],
                })
        elif msg["role"] == "toolResult":
            messages.append({
                "role": "tool",
                "tool_call_id": msg["toolCallId"],
                "content": (
                    "".join(p["text"] for p in msg["content"] if p.get("type") == "text")
                    if isinstance(msg.get("content"), list)
                    else str(msg.get("content", ""))
                ),
            })
    return messages


def _tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


class OpenAIModel(Model):
    """Model that calls OpenAI (or OpenAI-compatible) chat completions API."""

    async def stream(
        self,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("OpenAI model requires: pip install openai") from e

        api_key = self.api_key or get_env_api_key(self.provider)
        if not api_key:
            raise ValueError(f"No API key for provider: {self.provider}")

        client = AsyncOpenAI(api_key=api_key, base_url=self.base_url or None)
        messages = _messages_to_openai(context)
        openai_tools = _tools_to_openai(context.tools)
        kwargs: dict[str, Any] = {
            "model": self.id,
            "messages": messages,
            "stream": True,
            "temperature": options.get("temperature"),
            "max_tokens": options.get("max_tokens") or min(self.max_tokens, 4096),
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        output = make_empty_assistant_message(self.api, self.provider, self.id)
        abort_signal = options.get("signal")
        yield {"type": "start", "partial": dict(output)}

        try:
            response_stream = await client.chat.completions.create(**kwargs)
            content_index = 0
            current_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            async for chunk in response_stream:
                if abort_signal is not None and getattr(abort_signal, "is_set", None) and abort_signal.is_set():
                    output["stopReason"] = "aborted"
                    yield {"type": "error", "reason": "aborted", "error": output}
                    return
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    current_text += delta.content or ""
                    output["content"] = [{"type": "text", "text": current_text}]
                    yield {
                        "type": "text_delta",
                        "contentIndex": content_index,
                        "delta": delta.content or "",
                        "partial": dict(output),
                    }
                if delta.tool_calls:
                    for tc in delta.tool_calls or []:
                        idx = tc.index if tc.index is not None else 0
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments
                if chunk.choices[0].finish_reason:
                    reason = chunk.choices[0].finish_reason
                    output["stopReason"] = "toolUse" if reason == "tool_calls" else "stop"
                    if tool_calls_acc:
                        for idx in sorted(tool_calls_acc):
                            t = tool_calls_acc[idx]
                            args_str = t.get("arguments") or "{}"
                            try:
                                args = json.loads(args_str)
                            except json.JSONDecodeError:
                                args = {}
                            output["content"].append({
                                "type": "toolCall",
                                "id": t.get("id") or f"call_{idx}",
                                "name": t.get("name") or "",
                                "arguments": args,
                            })
            yield {"type": "done", "reason": output.get("stopReason", "stop"), "message": output}
        except Exception as e:
            output["stopReason"] = "error"
            output["errorMessage"] = str(e)
            yield {"type": "error", "reason": "error", "error": output}
