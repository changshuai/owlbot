"""
ai_models: Unified multi-provider LLM API.

Upper layer only sees Model and get_model. Model encapsulates stream() and invoke();
subclasses (OpenAIModel, AnthropicModel, GoogleModel) call vendor SDKs directly.
"""
from .envapikeys import get_env_api_key
from .modelbase import Model
from .models import get_model
from .types_ import Context, SimpleStreamOptions, StreamOptions

__all__ = [
    "Context",
    "Model",
    "SimpleStreamOptions",
    "StreamOptions",
    "get_env_api_key",
    "get_model",
]
