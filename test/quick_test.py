"""
Run from python/:  python test/quick_test.py
Or with PYTHONPATH:  PYTHONPATH=. python test/quick_test.py  (from python/)
"""
import asyncio
import sys
from pathlib import Path

# Allow importing ai_models when run as script without PYTHONPATH
_src = Path(__file__).resolve().parent.parent
if _src not in sys.path:
    sys.path.insert(0, str(_src))

from LLMs import Context, get_env_api_key, get_model


async def main() -> None:
    # If openai/gpt-4o-mini returns 403 (not available in region), try e.g.:
    # google/gemini-2.0-flash-001, deepseek/deepseek-chat, meta-llama/llama-3.3-70b-instruct
    model = get_model("openrouter", "deepseek/deepseek-chat", api_key=get_env_api_key("openrouter"))
    context = Context(messages=[{"role": "user", "content": "who are you?", "timestamp": 0}])
    async for ev in model.stream(context, {}):
        if ev.get("type") == "text_delta":
            print(ev.get("delta", ""), end="", flush=True)
        elif ev.get("type") == "error":
            print("Error:", ev.get("error", {}).get("errorMessage"))


if __name__ == "__main__":
    asyncio.run(main())