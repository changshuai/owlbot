"""
Base Model: abstract class for all model implementations.

Upper layer only sees Model. Subclasses (OpenAIModel, etc.) call vendor SDKs directly
and are instantiated by get_model() according to provider.
"""
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from .types_ import Api, Context, Provider, SimpleStreamOptions


def make_empty_assistant_message(
    api: Api,
    provider: Provider,
    model_id: str,
) -> dict[str, Any]:
    """Build a minimal assistant message dict for stream output."""
    return {
        "role": "assistant",
        "content": [],
        "api": api,
        "provider": provider,
        "model": model_id,
        "usage": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {
                "input": 0.0,
                "output": 0.0,
                "cacheRead": 0.0,
                "cacheWrite": 0.0,
                "total": 0.0,
            },
        },
        "stopReason": "stop",
        "timestamp": int(time.time() * 1000),
    }


class Model(ABC):
    """
    Abstract base for all models. Upper layer uses only this type.

    Subclasses implement stream() by calling the vendor SDK directly.
    invoke() has a default implementation that consumes stream(); subclasses may override.
    """

    def __init__(
        self,
        id: str,
        name: str,
        provider: Provider,
        base_url: str,
        api_key: str | None,
        context_window: int,
        max_tokens: int,
        api: Api,
    ) -> None:
        self.id = id
        self.name = name
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.api = api

    @abstractmethod
    def stream(
        self,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream assistant message events. Implemented by subclasses using vendor SDK."""
        ...

    async def invoke(
        self,
        context: Context,
        options: SimpleStreamOptions,
    ) -> dict[str, Any]:
        """Consume stream() and return the final assistant message (or raise on error)."""
        final_message = None
        async for event in self.stream(context, options):
            if event.get("type") == "done":
                final_message = event.get("message")
                break
            elif event.get("type") == "error":
                err_msg = event.get("error", {})
                err_text = err_msg.get("errorMessage", str(err_msg)) if isinstance(err_msg, dict) else str(err_msg)
                raise RuntimeError(err_text)
        if final_message is None:
            raise RuntimeError("Stream ended without done or error event")
        return final_message
