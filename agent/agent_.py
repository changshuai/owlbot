from dataclasses import dataclass
from common.paths import AGENTS_DIR, WORKSPACE_DIR, SKILLS_DIR
import re
from pathlib import Path
from typing import Optional, Any
from agent.memory_store import get_memory_store
from agent.skill_manager import SkillLoader
from config.bootstrap_loader import BootstrapLoader

## Valid ID Regex
VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
DEFAULT_AGENT_ID = "main"

def normalize_agent_id(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return DEFAULT_AGENT_ID
    if VALID_ID_RE.match(trimmed):
        return trimmed.lower()
    cleaned = INVALID_CHARS_RE.sub("-", trimmed.lower()).strip("-")[:64]
    return cleaned or DEFAULT_AGENT_ID

def _format_recalled(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines: list[str] = []
    for r in results:
        lines.append(f"- [{r['path']}] {r['snippet']}")
    return "\n".join(lines)

@dataclass
class Agent:
    id: str
    name: str
    model: str = ""
    capacity: str = "full"
    role: str = "general"

    def __init__(self, id: str, name: str, personality: str = "", model: str = "", role: str = "general") -> None:
        self.id = normalize_agent_id(id)
        self.name = name
        self.model = model or "deepseek/deepseek-chat"
        self.role = role
        workspace_dir = self.get_agent_workspace()
        self.bootstrap_loader = BootstrapLoader(workspace_dir)

        # Global skills + role-based skills + per-agent private skills.
        role_skills_dir = SKILLS_DIR / "roles" / self.role
        private_skills_dir = workspace_dir / "skills"
        extra_dirs = [role_skills_dir, private_skills_dir]
        self.skill_loader = SkillLoader(extra_dirs=extra_dirs)

    def system_prompt(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)
    
    def get_agent_workspace(self) -> Path:
        """Return per-agent workspace dir (created by AgentManager.register)."""
        ws = WORKSPACE_DIR / f"workspace-{self.id}"
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def _build_memory_prompt(
        self, last_user_message: str = "", session_key: str = ""
    ) -> str:
        # Memory auto-recall based on the latest user message
        memory_store = get_memory_store(self.id)
        memory_context = ""
        if last_user_message:
            recalled = memory_store.hybrid_search(
                last_user_message,
                top_k=3,
                session_key=session_key,
            )
            memory_context = _format_recalled(recalled)

        sections: list[str] = []
        # Memory (evergreen + auto-recalled)
        if self.capacity == "full":
            mem_md = self.bootstrap_loader.load_file("MEMORY.md").strip()
            parts: list[str] = []
            if mem_md:
                parts.append(f"### Evergreen Memory\n{mem_md}")
            if memory_context:
                parts.append(f"### Recalled Memories (auto-searched)\n\n{memory_context}")
            if parts:
                sections.append("## Memory\n" + "\n".join(parts))
            sections.append(
                "## Memory Instructions\n\n"
                "Save durable facts with `memory.write` or MEMORY.md (not the full chat log). "
                "Older turns are often dropped after ~10–15+ rounds of chat; if nothing important was saved lately, write it now. "
                "Write key points before long threads. Trimmed chat is archived on disk—not in `memory.search`. "
                "Sections above are MEMORY.md + diary only; use `memory.search` for more. "
                "Use recalled facts naturally.\n"
            )
        return sections

    def build_system_prompt_for_agent(
        self,
        *,
        channel: str = "cli",
        last_user_message: str = "",
        session_key: str = "",
    ) -> str:
        """
        Build a rich, per-turn system prompt for the given agent.

        - Uses per-agent workspace for bootstrap files / skills / memory.
        - Falls back to AgentConfig.system_prompt() semantics when no IDENTITY.md.
        """
        bootstrap = self.bootstrap_loader.load_all(self.capacity)
        sections: list[str] = []

        # Identity
        identity = bootstrap.get("IDENTITY.md", "").strip()
        sections.append(identity if identity else "You are a helpful personal AI assistant.")

        # Soul (personality)
        if self.capacity == "full":
            soul = bootstrap.get("SOUL.md", "").strip()
            if soul:
                sections.append(f"## Personality\n{soul}")

        # Tools guidance
        # tools_md = bootstrap.get("TOOLS.md", "").strip()
        # if tools_md:
        #     sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

        # Skills
        skills_descriptions = self.skill_loader.get_descriptions()
        if self.capacity == "full" and skills_descriptions:
            sections.append(
                "## Skills available:\n"
                f"{skills_descriptions}\n\n"
                "When you need the detailed instructions for a specific skill, "
                "call the `skill` tool with the exact skill name from the list above."
            )

        # Memory
        sections.extend(
            self._build_memory_prompt(
                last_user_message,
                session_key=session_key,
            )
        )

        # # Bootstrap context (remaining files)
        # if self.capacity in ("full", "minimal"):
        #     for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
        #         content = bootstrap.get(name, "").strip()
        #         if content:
        #             sections.append(f"## {name.replace('.md', '')}\n\n{content}")

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
            "REPL": "You are responding via a terminal REPL. Markdown is supported.",
            "cli": "You are responding via a CLI. Markdown is supported.",
            "whatsapp_web": "You are responding via WhatsApp Web. Keep messages concise.",
            "discord": "You are responding via Discord. Keep messages under 2000 characters.",
        }

        sections.append(f"## {hints.get(channel)}")

        return "\n\n".join(sections)

class AgentManager:
    def __init__(self, agents_base: Optional[Path] = None) -> None:
        self._agents: dict[str, Agent] = {}
        self._agents_base = agents_base or AGENTS_DIR
        self._sessions: dict[str, list[dict]] = {}
        # Session-scoped ephemeral state (not persisted): extracted key variables, summaries, etc.
        self._session_state: dict[str, dict[str, Any]] = {}

    def register(self, agent: Agent) -> None:
        aid = normalize_agent_id(agent.id)
        agent.id = aid
        self._agents[aid] = agent
        agent_dir = self._agents_base / aid
        (agent_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (WORKSPACE_DIR / f"workspace-{aid}").mkdir(parents=True, exist_ok=True)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def get_session(self, session_key: str, agent_id: str = "") -> list[dict]:
        if session_key not in self._sessions:
            if agent_id:
                from .session_store import load_chat_session

                self._sessions[session_key] = load_chat_session(agent_id, session_key)
            else:
                self._sessions[session_key] = []
        return self._sessions[session_key]

    def get_session_state(self, session_key: str) -> dict[str, Any]:
        """
        Get a mutable, session-scoped state dict.

        This is intended for short-term working memory (key variables extracted from summaries,
        artifact refs, counters) and is not persisted across process restarts.
        """
        if session_key not in self._session_state:
            self._session_state[session_key] = {}
        return self._session_state[session_key]

    def list_sessions(self, agent_id: str = "") -> dict[str, int]:
        aid = normalize_agent_id(agent_id) if agent_id else ""
        return {k: len(v) for k, v in self._sessions.items()
                if not aid or k.startswith(f"agent:{aid}:")}