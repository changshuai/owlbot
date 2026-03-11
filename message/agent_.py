from dataclasses import dataclass
from common.paths import AGENTS_DIR, WORKSPACE_DIR
import re
from pathlib import Path

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

@dataclass
class Agent:
    id: str
    name: str
    personality: str = ""
    model: str = ""
    dm_scope: str = "per-peer"

    @property
    def effective_model(self) -> str:
        return self.model or "deepseek/deepseek-chat"

    def system_prompt(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)

class AgentManager:
    def __init__(self, agents_base: Path | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        self._agents_base = agents_base or AGENTS_DIR
        self._sessions: dict[str, list[dict]] = {}

    def register(self, agent: Agent) -> None:
        aid = normalize_agent_id(agent.id)
        agent.id = aid
        self._agents[aid] = agent
        agent_dir = self._agents_base / aid
        (agent_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (WORKSPACE_DIR / f"workspace-{aid}").mkdir(parents=True, exist_ok=True)

    def get_agent(self, agent_id: str) -> Agent | None:
        return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def get_session(self, session_key: str) -> list[dict]:
        if session_key not in self._sessions:
            self._sessions[session_key] = []
        return self._sessions[session_key]

    def list_sessions(self, agent_id: str = "") -> dict[str, int]:
        aid = normalize_agent_id(agent_id) if agent_id else ""
        return {k: len(v) for k, v in self._sessions.items()
                if not aid or k.startswith(f"agent:{aid}:")}