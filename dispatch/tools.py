from typing import Any
from datetime import datetime, timezone
from pathlib import Path
import time

MAX_TOOL_OUTPUT = 30000

TOOLS = [
    {"name": "read_file", "description": "Read the contents of a file.",
     "input_schema": {"type": "object", "required": ["file_path"],
                      "properties": {"file_path": {"type": "string", "description": "Path to the file."}}}},
    {"name": "get_current_time", "description": "Get the current date and time in UTC.",
     "input_schema": {"type": "object", "properties": {}}},
]
# LLM layer expects "parameters" (JSON Schema)
TOOLS_LLM: list[dict[str, Any]] = [
    {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
    for t in TOOLS
]

def _tool_read(file_path: str) -> str:
    try:
        p = Path(file_path).resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"
        content = p.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            return content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except Exception as exc:
        return f"Error: {exc}"

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": lambda file_path: _tool_read(file_path),
    "get_current_time": lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
}

def process_tool_call(name: str, inp: dict) -> str:
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Error: Unknown tool '{name}'"
    try:
        return handler(**inp)
    except Exception as exc:
        return f"Error: {name} failed: {exc}"






