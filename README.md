# Owlbot

**Multi-channel personal AI assistant (Python).**  
Connects llms to **sessions**, **tools**, **bootstrap files**, and **memory**; routes inbound messages to the right **agent** via bindings.

### Why this repo exists (learning project)

Owlbot is intentionally a **learning-oriented** codebase: the primary aim is to **follow and internalize current agent techniques**—tool-use loops, long-context handling, memory, routing across channels, and related patterns—and to **implement them in a runnable system**. Progress is measured both by **what works in practice** and by **what you learn building it**, not only by feature completeness.

---

## Quick start

| Step | What to do |
|------|------------|
| 1 | From the repo root: `python main.py` |
| 2 | Configure `workspace/runtime_config.json` (agents, bindings, channels) |
| 3 | Set API keys for your provider (`MODEL_PROVIDER`, `MODEL_ID`, etc.) |

---

## Version 0.1 (preview)

| | |
|---|--|
| **Scope** | Core **agent path** works end-to-end: model loop, tools, routing, basic context handling. |
| **Tested** | **CLI** and **WhatsApp** only. Other channels exist in code but are **not** validated in this release. |
| **Entry** | `python main.py` |
| **Lineage** | Learned from **[OpenClaw](https://github.com/openclaw/openclaw)** (ideas & architecture)—**not** a fork; this tree is Python and evolves on its own. |
| **What’s next** | Backlog and design notes live in **`Design.txt`**. |

---

## Features

### Agent runtime

- Per-agent **workspace**: `workspace-<agent_id>/`, model id, optional **role-based skills**.
- **System prompt** from IDENTITY / SOUL / **MEMORY.md**, skill blurbs, **memory recall**, and channel hints.
- **`run_agent`**: async **tool loop**—stream → tool calls → results → repeat until stop or round cap.
- **Tools**: `bash`, `fileOps` (read/write/edit under workspace), `memory` (write/search), `skill` loader.

### Context & memory

- **Large tool outputs**: shrink via optional **`SUMMARIZER_MODEL_ID`** or head/tail truncation (`TOOL_RESULT_MAX_CHARS`, etc.).
- **History trim**: when estimated context exceeds budget (`CONTEXT_BUDGET_RATIO`, **`CONTEXT_MAX_TOKENS`**, …). Trimmed prefix → **`context_archive_*.jsonl`** + placeholder; archives **excluded** from `memory.search`.
- **Long-term memory**: `memory.write` / `memory.search`, **MEMORY.md**, session **diary JSONL** + hybrid recall in the prompt. Optional **semantic** vector leg when **`MEMORY_EMBEDDING_MODEL`** is set (see below).
- **Chat persistence**: full message list under `workspace-<agent_id>/chat_sessions/`. Disable with **`PERSIST_CHAT_SESSIONS=0`**.
- **Logging**: per round, **token usage** from API when available (OpenAI-compatible `include_usage`); else rough estimate.

### Messaging & channels

- **Message center**: inbound message → **bindings** → `agent_id` + **`session_key`** → `run_agent` → reply on same channel.
- **Routing**: `session_key` = agent + channel + account + peer (isolated context).
- **Channels** (via `runtime_config.json` + **ChannelManager**): CLI, WhatsApp Web (neonize), WhatsApp Cloud API, Telegram, …
- **JSON-RPC gateway**: `run_agent`, session listing, abort, …

---

## Configuration

| Piece | Role |
|-------|------|
| **`workspace/runtime_config.json`** | Agents, bindings, channel accounts, `auto_bridge` |
| **Environment** | Provider keys, `MODEL_PROVIDER`, `MODEL_ID`, … |
| **Context tuning** | See **`docs/context-compaction.md`** |
| **Semantic memory** | Optional: **`MEMORY_EMBEDDING_MODEL`** + `requirements-memory-semantic.txt` (see below) |

### Semantic memory (optional)

Hybrid `memory.search` / prompt recall can use **Sentence-Transformers** embeddings when **`MEMORY_EMBEDDING_MODEL`** is set to a **non-empty** string. You do **not** need a pre-downloaded model on disk unless you are fully offline.

| What to set | Meaning |
|-------------|---------|
| **Hugging Face model id** | e.g. `paraphrase-multilingual-MiniLM-L12-v2`, `BAAI/bge-small-zh-v1.5`. First run **downloads** weights into the local Hugging Face cache; later runs use the cache only (no separate embedding server). |
| **Local directory path** | Absolute path to an already unpacked model folder (`config.json`, weights, etc.). Use when there is **no outbound network** or you only allow vendored models. |

**Setup**

1. Install optional deps: `pip install -r requirements-memory-semantic.txt`
2. Export the variable, for example:  
   `export MEMORY_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2`  
   or for Chinese-heavy text: `BAAI/bge-small-zh-v1.5`
3. Embeddings for each session are cached under that agent workspace as **`memory/<session>/.owlbot_semantic.npz`** (invalidated when chunk text changes).

**Cache location (HF downloads)** defaults to something like `~/.cache/huggingface/hub`. Override with **`HF_HOME`** (or other Hugging Face cache env vars) if you want a different disk.

If **`MEMORY_EMBEDDING_MODEL`** is unset, `sentence-transformers` is missing, or the model fails to load, recall **falls back** to the built-in hash-based vector leg plus keyword search (same as before).

---

## Roadmap

Authoritative list and rationale: **`Design.txt`**.

Planned / in-progress themes after 0.1:

- Long-horizon / multi-step tasks  
- Subagents  
- MCP  
- More skills  
- **Sandbox**: run user scripts and return outputs (not only show code)  

*(Design also holds context/memory tuning notes and what’s already landed.)*

---

## Repository layout

```
agent/       # loop, tools, memory, context compaction, session store
message/     # routing, message center, gateway
channels/    # channel implementations
config/      # runtime config loader
workspace/   # default workspace, runtime_config.json, per-agent workspaces
llms/        # model providers & streaming
docs/        # e.g. context-compaction.md
```
