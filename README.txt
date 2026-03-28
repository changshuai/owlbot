# Owlbot

## Version 0.1 (preview)

This is an **early 0.1** drop: **core agent behavior is working end-to-end** (model loop, tools, routing, basic context handling). **Only the CLI and WhatsApp** paths have been exercised in practice; other channels exist in code but are not validated here.

**Run the app:** `python main.py` (from the project root, with `workspace/runtime_config.json` and API keys configured as needed).

**Origins:** Owlbot was built by **learning from and improving on [OpenClaw](https://github.com/openclaw/openclaw)** (ideas and architecture—not a fork). This codebase is Python and evolves separately.

**What’s next:** Remaining work—long-horizon tasks, subagents, MCP, more skills, sandboxed execution, and finer context/memory policy—is captured in **`Design.txt`** (design notes and backlog). Treat that file as the living product/design checklist after 0.1.

---

Owlbot is a multi-channel personal AI assistant framework. It connects LLMs to sessions, tools, bootstrap files, and memory, and routes inbound messages to the correct agent.

## Agent layer

- **Agents** use a per-id **workspace** (`workspace-<agent_id>/`), model id, and optional **role-based skills**.
- **System prompt** is built from **IDENTITY / SOUL / MEMORY.md**, skill descriptions, **memory recall**, and channel hints (CLI, WhatsApp Web, Discord, etc.).
- **`run_agent`** runs an async **tool loop**: stream the model → execute tool calls → append results → repeat until the model stops or a round limit is hit.
- **Tools** include shell (**bash**), **fileOps** (read/write/edit under workspace), **memory** (write/search), and **skill** loading.
- **Context management**
  - Large **tool outputs** are shrunk (optional summarizer via `SUMMARIZER_MODEL_ID`, or head/tail truncation).
  - **History** is trimmed when estimated context exceeds budget (`CONTEXT_BUDGET_RATIO`, `CONTEXT_MAX_TOKENS`, etc.). Removed prefix is **archived** under `memory/<session>/context_archive_*.jsonl` with a **placeholder** message; archives are **not** indexed by `memory.search`.
- **Long-term memory**: `memory.write` / `memory.search`, **MEMORY.md**, and per-session diary **JSONL**; hybrid-style recall for the prompt.
- **Chat persistence**: full message lists can be saved under `workspace-<agent_id>/chat_sessions/` and reloaded after restart. Set `PERSIST_CHAT_SESSIONS=0` to disable.
- **Logging**: per model round, **token usage** when the API returns it (OpenAI-compatible streaming with `include_usage`); otherwise a rough estimate.

## Messaging and channels

- **Message center** takes inbound messages from **channels**, resolves **bindings** → `agent_id` + `session_key`, runs `run_agent`, sends the reply on the same channel.
- **Routing**: `session_key` encodes agent + channel + account + peer so context stays isolated.
- **Channels** (via `runtime_config.json` and **ChannelManager**): CLI, WhatsApp Web (neonize), WhatsApp Cloud API, Telegram, and others.
- **JSON-RPC gateway** can invoke `run_agent`, list sessions, abort runs, etc.

## Configuration

- **`workspace/runtime_config.json`**: agents, bindings, channel accounts, `auto_bridge`.
- **Environment**: API keys per provider (`MODEL_PROVIDER`, `MODEL_ID`, etc.). See `LLMs/` and `docs/context-compaction.md` for context-related env vars.

## Roadmap (see `Design.txt`)

The authoritative backlog and design rationale live in **`Design.txt`**. Highlights still open or in progress after 0.1 include:

- Long-horizon / multi-step tasks  
- Subagents  
- MCP integration  
- More skills  
- **Sandbox environment**: run user scripts and return outputs directly (not only show code)  

*(Design also records context/memory tuning notes, message-center role, and items already landed—e.g. chat history persistence work—alongside future ideas.)*

## Layout (short)

- `agent/` — agent loop, tools, memory, context compaction, session store
- `message/` — routing, message center, gateway
- `channels/` — channel implementations
- `config/` — runtime config loader
- `workspace/` — default workspace, `runtime_config.json`, per-agent workspaces
