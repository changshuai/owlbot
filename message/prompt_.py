from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.paths import WORKSPACE_DIR
from message.agent_ import Agent
from .memory_store import get_memory_store
from .skill_manager import SkillsManager
from .bootstrap_loader import BootstrapLoader
from common.paths import get_agent_workspace

def _format_recalled(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines: list[str] = []
    for r in results:
        lines.append(f"- [{r['path']}] {r['snippet']}")
    return "\n".join(lines)

def build_system_prompt(
    *,
    mode: str = "full",
    bootstrap: dict[str, str] | None = None,
    skills_block: str = "",
    memory_context: str = "",
    agent_id: str = "main",
    channel: str = "terminal",
    model_id: str | None = None,
) -> str:
    """Assemble the layered system prompt."""
    if bootstrap is None:
        bootstrap = {}
    sections: list[str] = []

    # Identity
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful personal AI assistant.")

    # Soul (personality)
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # Tools guidance
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # Skills
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # Memory (evergreen + auto-recalled)
    if mode == "full":
        mem_md = bootstrap.get("MEMORY.md", "").strip()
        parts: list[str] = []
        if mem_md:
            parts.append(f"### Evergreen Memory\n\n{mem_md}")
        if memory_context:
            parts.append(f"### Recalled Memories (auto-searched)\n\n{memory_context}")
        if parts:
            sections.append("## Memory\n\n" + "\n".join(parts))

    # Bootstrap context (remaining files)
    if mode in ("full", "minimal"):
        for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
            content = bootstrap.get(name, "").strip()
            if content:
                sections.append(f"## {name.replace('.md', '')}\n\n{content}")

    # Runtime context
    # now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    # model_label = model_id or os.getenv("MODEL_ID", "unknown-model")
    # sections.append(
    #     "## Runtime Context\n\n"
    #     f"- Agent ID: {agent_id}\n"
    #     f"- Model: {model_label}\n"
    #     f"- Channel: {channel}\n"
    #     f"- Current time: {now}\n"
    #     f"- Prompt mode: {mode}"
    # )

    # Response channel hints
    hints = {
        "terminal": "You are responding via a terminal REPL. Markdown is supported.",
        "whatsapp_web": "You are responding via WhatsApp Web. Keep messages concise.",
        "discord": "You are responding via Discord. Keep messages under 2000 characters.",
    }

    sections.append(f"## Channel\n\n{hints.get(channel, f'You are responding via {channel}.')}")

    return "\n\n".join(sections)


def build_system_prompt_for_agent(
    agent: Agent,
    agent_id: str,
    *,
    channel: str = "terminal",
    model_id: str | None = None,
    last_user_message: str = "",
) -> str:
    """
    Build a rich, per-turn system prompt for the given agent.

    - Uses per-agent workspace for bootstrap files / skills / memory.
    - Falls back to AgentConfig.system_prompt() semantics when no IDENTITY.md.
    """
    workspace_dir = get_agent_workspace(agent_id)

    # Load bootstrap files from per-agent workspace
    loader = BootstrapLoader(workspace_dir)
    bootstrap = loader.load_all(mode="full")

    # Skills
    skills_mgr = SkillsManager(workspace_dir)
    extra_dirs = [WORKSPACE_DIR / "skills"]
    skills_mgr.discover(extra_dirs=[d for d in extra_dirs if d.is_dir()])
    skills_block = skills_mgr.format_prompt_block()

    # Memory auto-recall based on the latest user message
    memory_store = get_memory_store(agent_id)
    memory_context = ""
    if last_user_message:
        recalled = memory_store.hybrid_search(last_user_message, top_k=3)
        memory_context = _format_recalled(recalled)

    # Build final layered system prompt
    prompt = build_system_prompt(
        mode="full",
        bootstrap=bootstrap,
        skills_block=skills_block,
        memory_context=memory_context,
        agent_id=agent_id,
        channel=channel,
        model_id=model_id,
    )
    return prompt


