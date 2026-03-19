"""Project paths. Single source of truth for workspace and related dirs."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
AGENTS_DIR = WORKSPACE_DIR / ".agents"
STATE_DIR = WORKSPACE_DIR / ".state"

MEMORY_FILE = WORKSPACE_DIR / "MEMORY.md"

PLAN_FILE = WORKSPACE_DIR / "PLAN.md"
REFLECTION_FILE = WORKSPACE_DIR / "REFLECTION.md"
SKILLS_DIR = WORKSPACE_DIR / "skills"

WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def get_agent_workspace(agent_id: str) -> Path:
    """Return per-agent workspace dir (created by AgentManager.register)."""
    ws = WORKSPACE_DIR / f"workspace-{agent_id}"
    ws.mkdir(parents=True, exist_ok=True)
    return ws