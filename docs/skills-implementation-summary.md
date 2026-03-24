# Skills Implementation Summary

This document summarizes the current skill system implementation in Owlbot, including global skills, role-based skills, private skills, and tool integration.

## 1) Current Skill Directory Design

Skills are now organized into three layers:

- Global shared skills:
  - `workspace/skills/general/**/SKILL.md`
- Role-based skills:
  - `workspace/skills/roles/<role>/**/SKILL.md`
- Agent private skills:
  - `workspace/workspace-<agent-id>/skills/**/SKILL.md`

Examples currently present:

- Global:
  - `workspace/skills/general/agent-builder/SKILL.md`
  - `workspace/skills/general/pdf/SKILL.md`
  - `workspace/skills/general/test-runner/SKILL.md`
- Role:
  - `workspace/skills/roles/coder/basic_coding/SKILL.md`
- Private:
  - `workspace/workspace-main/skills/auto_summary/SKILL.md`

## 2) How Skills Are Loaded

Main code: `agent/skill_manager.py`

- `SkillLoader` loads from:
  - base dir: `workspace/skills/general`
  - extra dirs: provided by caller (`extra_dirs`)
- Loading order:
  1. global (`general`)
  2. role dir (if configured)
  3. private dir (if configured)
- If skill names conflict, later-loaded directories override earlier ones.

This allows:

- Shared defaults in `general`
- Role specialization in `roles/<role>`
- Agent-specific override/customization in `workspace-<agent-id>/skills`

## 3) Agent-Side Integration

Main code: `agent/agent_.py`

- `Agent` now supports a `role` field (default: `"general"`).
- In `Agent.__init__`, `SkillLoader` is initialized with:
  - role dir: `workspace/skills/roles/<agent.role>`
  - private dir: `workspace/workspace-<agent-id>/skills`
- System prompt includes:
  - a “Skills available” section (from `get_descriptions()`)
  - instruction to call `skill` tool for full detail.

## 4) Tool-Side Integration

Main code: `agent/tools.py`

- Added tool: `skill`
  - input: `{ "name": "<skill-name>" }`
  - returns: full skill body wrapped as:
    - `<skill name="..."> ... </skill>`
- `_tool_skill` reads `tool_ctx` to get:
  - `agent_id`
  - `role`
- `_tool_skill` loads from:
  - global `general`
  - `roles/<role>`
  - `workspace-<agent-id>/skills`

## 5) Runtime Context Flow

Main code: `agent/agent_loop.py`

- During each run, `tool_ctx` now includes:
  - `agent_id`
  - `channel`
  - `session_key`
  - `role`
- This ensures tool loading behavior is consistent with the current agent identity and role.

## 6) Runtime Config Status

Main code: `config/config_runtime.py`, config file: `workspace/runtime_config.json`

- Current runtime config loader creates agents with:
  - `id`, `name`, `model`
- It does not yet pass `role` from JSON into `Agent(...)`.

Current effect:

- If `role` is not passed at creation time, agent role defaults to `"general"`.

Recommended small follow-up:

- Update runtime config loading to pass `role`:
  - `role=a.get("role", "general")`
- Add `"role"` in `workspace/runtime_config.json` for each agent.

## 7) Demo Skill for Execution Test

A simple executable skill was added:

- Skill:
  - `workspace/skills/general/test-runner/SKILL.md`
- Script:
  - `workspace/skills/general/test-runner/scripts/hello.py`

Purpose:

- Demonstrate that a skill can guide the agent to run a local script via tools (`bash`), not just provide static guidance.

## 8) Known Constraints and Notes

- `SKILL.md` is instruction text; it is not executable by itself.
- Real execution requires tool calls (`bash`, `fileOps`, etc.).
- Missing dependencies/path assumptions in referenced scripts can still cause runtime failures.

## 9) Practical Usage Pattern

Suggested pattern when adding new skills:

1. Put shared capability in `workspace/skills/general/<skill-name>/SKILL.md`
2. Put role-specific behavior in `workspace/skills/roles/<role>/<skill-name>/SKILL.md`
3. Let agents generate personal/private learnings in:
   - `workspace/workspace-<agent-id>/skills/<skill-name>/SKILL.md`
4. Keep names stable; use private skill names to override global behavior when needed.

