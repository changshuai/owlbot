"""
Resolve API key from environment by provider.
"""
import os
from pathlib import Path

# Load .env from project root when this module is first imported
_env_loaded = False


def _load_dotenv_if_available() -> None:
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    try:
        from dotenv import load_dotenv
        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env")
    except ImportError:
        pass  # python-dotenv not installed; rely on real env vars only


def get_env_api_key(provider: str) -> str | None:
    """Return the API key for the given provider from environment, or None."""
    _load_dotenv_if_available()
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if provider == "google":
        return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if provider == "openrouter":
        return os.environ.get("OPENROUTER_API_KEY")
    key = os.environ.get(f"{provider.upper().replace('-', '_')}_API_KEY")
    if key:
        return key
    return os.environ.get("API_KEY")
