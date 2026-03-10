"""Build Model instances for known providers. Only Model (base type) is exposed to upper layer."""
from .anthropicmodel import AnthropicModel
from .googlemodel import GoogleModel
from .modelbase import Model
from .openaimodel import OpenAIModel

DEFAULT_MODELS: dict[tuple[str, str], dict[str, int]] = {
    ("openai", "gpt-4o-mini"): {"context_window": 128000, "max_tokens": 16384},
    ("openai", "gpt-4o"): {"context_window": 128000, "max_tokens": 16384},
    ("anthropic", "claude-sonnet-4-20250514"): {"context_window": 200000, "max_tokens": 8192},
    ("anthropic", "claude-3-5-sonnet-20241022"): {"context_window": 200000, "max_tokens": 8192},
    ("google", "gemini-1.5-flash"): {"context_window": 1000000, "max_tokens": 8192},
    ("google", "gemini-1.5-pro"): {"context_window": 2000000, "max_tokens": 8192},
    ("openrouter", "anthropic/claude-sonnet-4"): {"context_window": 200000, "max_tokens": 8192},
    ("openrouter", "openai/gpt-4o-mini"): {"context_window": 128000, "max_tokens": 16384},
    ("openrouter", "google/gemini-2.0-flash-001"): {"context_window": 1000000, "max_tokens": 8192},
}

API_BY_PROVIDER: dict[str, str] = {
    "openai": "openai-responses",
    "anthropic": "anthropic-messages",
    "google": "google-generative-ai",
    "openrouter": "openai-responses",
}

BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "google": "https://generativelanguage.googleapis.com",
    "openrouter": "https://openrouter.ai/api/v1",
}


def get_model(
    provider: str,
    model_id: str,
    name: str | None = None,
    api_key: str | None = None,
) -> Model:
    """Return a Model for the given provider and model id. Only the base Model type is exposed.

    If api_key is set, it is used and the provider's environment variable is not required.
    If api_key is None, the implementation falls back to get_env_api_key(provider).
    """
    api = API_BY_PROVIDER.get(provider, provider)
    base_url = BASE_URLS.get(provider, "")
    key = (provider, model_id)
    info = DEFAULT_MODELS.get(key, {"context_window": 128000, "max_tokens": 4096})

    if provider in ("openai", "openrouter"):
        return OpenAIModel(
            id=model_id,
            name=name or model_id,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            context_window=info["context_window"],
            max_tokens=info["max_tokens"],
            api=api,
        )
    if provider == "anthropic":
        return AnthropicModel(
            id=model_id,
            name=name or model_id,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            context_window=info["context_window"],
            max_tokens=info["max_tokens"],
            api=api,
        )
    if provider == "google":
        return GoogleModel(
            id=model_id,
            name=name or model_id,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            context_window=info["context_window"],
            max_tokens=info["max_tokens"],
            api=api,
        )
    # Default to OpenAI-compatible for unknown provider
    return OpenAIModel(
        id=model_id,
        name=name or model_id,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        context_window=info["context_window"],
        max_tokens=info["max_tokens"],
        api=api,
    )
