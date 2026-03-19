from __future__ import annotations

from pathlib import Path
import re
import common.paths

class SkillLoader:
    def __init__(self, extra_dirs: list[Path] | None = None):
        """
        Load skills from the global skills dir plus any extra dirs (e.g. per-agent private skills).

        Dirs loaded later in the list can override skills from earlier dirs with the same name.
        """
        # base_dir is the global, role-agnostic skills root (skills/general).
        # Role-scoped skills live under skills/roles/<role> and are provided via extra_dirs.
        self.base_dir = common.paths.SKILLS_DIR / "general"
        self.extra_dirs = extra_dirs or []
        self.skills: dict[str, dict] = {}
        self._load_all()

    def _load_all(self):
        """Load all SKILL.md files from the configured directories."""
        dirs: list[Path] = [self.base_dir] + list(self.extra_dirs)
        for idx, skills_dir in enumerate(dirs):
            if not skills_dir.exists():
                continue
            for f in sorted(skills_dir.rglob("SKILL.md")):
                text = f.read_text()
                meta, body = self._parse_frontmatter(text)
                name = meta.get("name", f.parent.name)
                # Later dirs (e.g. role-specific or per-agent) override earlier ones.
                self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """Parse YAML frontmatter between --- delimiters."""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text
        meta = {}
        for line in match.group(1).strip().splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                meta[key.strip()] = val.strip()
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1: short descriptions for the system prompt."""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: full skill body returned in tool_result."""
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
