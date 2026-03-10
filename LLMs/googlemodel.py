"""
Google Gemini model: stream/invoke via Google Generative AI SDK.
Uses sync stream in a thread to yield async.
"""
import asyncio
import queue
import threading
from typing import Any, AsyncIterator

from .envapikeys import get_env_api_key
from .modelbase import Model, make_empty_assistant_message
from .types_ import Context, SimpleStreamOptions


def _messages_to_gemini(context: Context) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in context.messages:
        if msg["role"] == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                contents.append({"role": "user", "parts": [{"text": content}]})
            else:
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image":
                        parts.append({
                            "inline_data": {
                                "mime_type": part.get("mimeType", "image/png"),
                                "data": part["data"],
                            },
                        })
                if parts:
                    contents.append({"role": "user", "parts": parts})
        elif msg["role"] == "assistant":
            parts = []
            for p in msg.get("content", []):
                if p.get("type") == "text":
                    parts.append({"text": p.get("text", "")})
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif msg["role"] == "toolResult":
            text = ""
            if isinstance(msg.get("content"), list):
                text = "".join(
                    p.get("text", "") for p in msg["content"] if p.get("type") == "text"
                )
            else:
                text = str(msg.get("content", ""))
            contents.append({
                "role": "user",
                "parts": [{"text": f"[Tool result for {msg.get('toolName', '')}]: {text}"}],
            })
    return contents


def _run_sync_stream(
    model_id: str,
    api_key: str,
    history: list[dict[str, Any]],
    out_queue: queue.Queue,
) -> None:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model_id)
    if not history or history[-1].get("role") != "user":
        out_queue.put(("done", None))
        return
    chat = gemini.start_chat(history=history[:-1])
    parts = history[-1].get("parts", [])
    if not parts:
        prompt = ""
    elif len(parts) == 1 and "text" in parts[0]:
        prompt = parts[0]["text"]
    else:
        prompt = parts
    try:
        for chunk in chat.send_message(prompt, stream=True):
            if chunk.text:
                out_queue.put(("delta", chunk.text))
        out_queue.put(("done", None))
    except Exception as e:
        out_queue.put(("error", str(e)))


class GoogleModel(Model):
    """Model that calls Google Generative AI (Gemini) API."""

    async def stream(
        self,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Google model requires: pip install google-generativeai"
            ) from e

        api_key = self.api_key or get_env_api_key(self.provider)
        if not api_key:
            raise ValueError(f"No API key for provider: {self.provider}")

        history = _messages_to_gemini(context)
        output = make_empty_assistant_message(self.api, self.provider, self.id)
        abort_signal = options.get("signal")
        yield {"type": "start", "partial": dict(output)}

        out_queue: queue.Queue = queue.Queue()
        thread = threading.Thread(
            target=_run_sync_stream,
            args=(self.id, api_key, history, out_queue),
            daemon=True,
        )
        thread.start()
        content_index = 0
        current_text = ""
        try:
            while True:
                try:
                    event_type, data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: out_queue.get(timeout=300),
                    )
                except queue.Empty:
                    break
                if abort_signal is not None and getattr(abort_signal, "is_set", None) and abort_signal.is_set():
                    output["stopReason"] = "aborted"
                    yield {"type": "error", "reason": "aborted", "error": output}
                    return
                if event_type == "delta":
                    current_text += data
                    output["content"] = [{"type": "text", "text": current_text}]
                    yield {
                        "type": "text_delta",
                        "contentIndex": content_index,
                        "delta": data,
                        "partial": dict(output),
                    }
                elif event_type == "done":
                    break
                elif event_type == "error":
                    output["stopReason"] = "error"
                    output["errorMessage"] = data
                    yield {"type": "error", "reason": "error", "error": output}
                    return
            yield {"type": "done", "reason": "stop", "message": output}
        except Exception as e:
            output["stopReason"] = "error"
            output["errorMessage"] = str(e)
            yield {"type": "error", "reason": "error", "error": output}
