from typing import Any
from datetime import datetime, timezone
from pathlib import Path

from .memory_store import get_memory_store
from common.paths import WORKSPACE_DIR
from common.logs import print_tool
import subprocess
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


MAX_TOOL_OUTPUT = 30000

TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for system commands, git, package managers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "fileOps",
        "description": (
            "File operation: read, write, or edit. "
            "read: return file content. write: overwrite file (creates dirs). "
            "edit: replace one exact old_string with new_string (must be unique)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "edit"],
                    "description": "read | write | edit",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path relative to working directory.",
                },
                "content": {"type": "string", "description": "Required for write."},
                "old_string": {"type": "string", "description": "Required for edit; must appear exactly once."},
                "new_string": {"type": "string", "description": "Required for edit."},
            },
            "required": ["action", "file_path"],
        },
    },
    {
        "name": "memory",
        "description": (
            "Long-term memory: write saves a fact; search returns relevant snippets by similarity. "
            "Use write when you learn something worth remembering; use search to recall past context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "search"],
                    "description": "write | search",
                },
                "content": {
                    "type": "string",
                    "description": "For write: the fact to remember.",
                },
                "category": {
                    "type": "string",
                    "description": "For write: preference, fact, context, project, etc. Default general.",
                },
                "query": {"type": "string", "description": "For search: search query."},
                "top_k": {
                    "type": "integer",
                    "description": "For search: max results. Default 5.",
                },
            },
            "required": ["action"],
        },
    },
]

# LLM layer expects "parameters" (JSON Schema)
TOOLS_LLM: list[dict[str, Any]] = [
    {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
    for t in TOOLS
]


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def safe_path(raw: str) -> Path:
    """Resolve path, block traversal outside WORKSPACE_DIR."""
    target = (WORKSPACE_DIR / raw).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR)):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKSPACE_DIR")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


# ---------------------------------------------------------------------------
# Bash: allowed cwd is WORKSPACE_DIR only; dangerous patterns are blocked.
# ---------------------------------------------------------------------------
BASH_MAX_TIMEOUT = 60
BASH_DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "> /dev/sd",
    "dd if=",
    "sudo",
    " su ",
    "su\n",
    "chmod 777",
    "chown ",
    "> /etc",
    ">> /etc",
    "> /usr",
    ">> /usr",
    "> /bin",
    ">> /bin",
    "> /sbin",
    ">> /sbin",
    "| nc ",
    "| bash",
    "| sh ",
    "curl | sh",
    "wget -O- | sh",
    "mknod ",
    "/dev/sd",
    "eval ",
]
# Commands that would escape workspace when cwd=WORKSPACE_DIR (relative is ok)
BASH_FORBIDDEN_PREFIXES = ["cd /", "cd ~", "cd ..", "> /", ">> /"]


def _tool_bash(command: str, timeout: int = 30, tool_ctx: dict[str, Any] | None = None) -> str:
    """Run a shell command in WORKSPACE_DIR with safety checks. No sudo, no write outside workspace."""
    if not command or not command.strip():
        return "Error: Empty command."

    raw = command.strip()
    for pattern in BASH_DANGEROUS_PATTERNS:
        if pattern in raw:
            return f"Error: Refused to run command containing forbidden pattern: '{pattern.strip()}'"
    for prefix in BASH_FORBIDDEN_PREFIXES:
        if raw.startswith(prefix) or f" {prefix}" in raw:
            return f"Error: Refused to run command that escapes workspace: '{prefix}'"

    effective_timeout = min(max(1, timeout), BASH_MAX_TIMEOUT)

    print_tool("bash", command)
    try:
        result = subprocess.run(
            raw,
            shell=True,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=str(WORKSPACE_DIR),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


def _tool_file(
    action: str,
    file_path: str,
    content: str = "",
    old_string: str = "",
    new_string: str = "",
    tool_ctx: dict[str, Any] | None = None,
) -> str:
    """Single file tool: read | write | edit. Paths are relative to WORKSPACE_DIR."""
    act = (action or "").strip().lower()
    if act not in ("read", "write", "edit"):
        return f"Error: file action must be read, write, or edit; got '{action}'"

    print_tool("file", f"{act} {file_path}")
    try:
        target = safe_path(file_path)
        if act == "read":
            if not target.exists():
                return f"Error: File not found: {file_path}"
            if not target.is_file():
                return f"Error: Not a file: {file_path}"
            return truncate(target.read_text(encoding="utf-8"))
        if act == "write":
            if content is None:
                return "Error: file write requires 'content'."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} chars to {file_path}"
        if act == "edit":
            if old_string is None or (isinstance(old_string, str) and not old_string.strip()):
                return "Error: file edit requires non-empty 'old_string'."
            if not target.exists():
                return f"Error: File not found: {file_path}"
            text = target.read_text(encoding="utf-8")
            count = text.count(old_string)
            if count == 0:
                return "Error: old_string not found in file. Make sure it matches exactly."
            if count > 1:
                return (
                    f"Error: old_string found {count} times. "
                    "It must be unique. Provide more surrounding context."
                )
            target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
            return f"Successfully edited {file_path}"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"
    return "Error: unexpected action"

def _tool_memory(
    action: str,
    content: str = "",
    category: str = "general",
    query: str = "",
    top_k: int = 5,
    tool_ctx: dict[str, Any] | None = None,
) -> str:
    """Single memory tool: write | search. Uses tool_ctx['agent_id'] for per-agent store."""
    act = (action or "").strip().lower()
    if act not in ("write", "search"):
        return f"Error: memory action must be write or search; got '{action}'"

    ctx = tool_ctx or {}
    agent_id = ctx.get("agent_id") or "default"
    store = get_memory_store(agent_id)

    if act == "write":
        if not (content or "").strip():
            return "Error: memory write requires non-empty 'content'."
        print_tool("memory", f"write [{category}] {content[:60]}...")
        return store.write_memory(content.strip(), category or "general")
    # search
    if not (query or "").strip():
        return "Error: memory search requires non-empty 'query'."
    print_tool("memory", f"search {query[:60]}...")
    results = store.hybrid_search(query.strip(), top_k=max(1, min(top_k, 20)))
    if not results:
        return "No relevant memories found."
    lines = [f"[{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results]
    return "\n".join(lines)


TOOL_HANDLERS: dict[str, Any] = {
    "fileOps": _tool_file,
    "bash": _tool_bash,
    "memory": _tool_memory,
}


def process_tool_call(name: str, inp: dict, tool_ctx: dict[str, Any] | None = None) -> str:
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Error: Unknown tool '{name}'"
    try:
        return handler(tool_ctx=tool_ctx, **inp)
    except TypeError as exc:
        logger.info(f"Error: Invalid arguments for {name}: {exc}")
        return f"Error: Invalid arguments for {name}: {exc}"
    except Exception as exc:
        logger.info(f"Error: {name} failed: {exc}")
        return f"Error: {name} failed: {exc}"

