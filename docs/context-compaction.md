# Context compaction (single scheme)

Implementation: `agent/context_compaction.py`; tool wiring: `agent/agent_loop.py::_execute_tool_calls()`; history trim before each model call: `_agent_loop()`.

## 1. Tool results

- **Threshold:** `TOOL_RESULT_MAX_CHARS` (default 12000), env-configurable.
- If the raw tool output string is longer:
  - If **`SUMMARIZER_MODEL_ID`** is set, call that model to shorten the text (plain text reply).
  - Otherwise **head + tail truncation** with a `[middle omitted]` separator (each half uses half of the available character budget after the separator).
- `details.shrink_method` on the `toolResult` message is `none` | `summarize` | `truncate_head_tail`.

## 2. Message history

- **Budget:** approximate character budget = `min(max(1024, context_window * CONTEXT_BUDGET_RATIO * 4), CONTEXT_MAX_TOKENS * 4)` when `CONTEXT_MAX_TOKENS > 0` (rough token→char, ~4 chars/token). If `CONTEXT_MAX_TOKENS` is `0`, only the ratio-based budget applies.
- If `system_prompt + messages` exceeds the budget, find the **smallest index** `start` such that the suffix `messages[start:]` is **valid** for `message_validator.validate_session_messages` and fits under budget.
- **Removed** prefix is written as JSONL via `MemoryStore.archive_messages()` under `memory/<safe_session>/context_archive_<timestamp>.jsonl`.
- A **user** placeholder is prepended with the archive path; `system_prompt` is not stored in `messages` and is unchanged.

## Environment

| Variable | Default | Meaning |
|----------|---------|---------|
| `TOOL_RESULT_MAX_CHARS` | 12000 | Max chars kept for a tool result after shrink |
| `TOOL_SUMMARY_INPUT_MAX_CHARS` | 24000 | Max chars fed to summarizer before middle-cut |
| `SUMMARIZER_MODEL_ID` | (empty) | Optional model id for summarization; uses `MODEL_PROVIDER` |
| `CONTEXT_BUDGET_RATIO` | 0.72 | Fraction of `context_window` (tokens) used as budget (×4 for chars) |
| `CONTEXT_MAX_TOKENS` | 16000 | Absolute cap (tokens) on system+messages budget; combined with ratio via `min`. Use `0` to disable |

## 3. Memory recall vs context archives

- **`context_archive_*.jsonl`** files are **not** indexed by `MemoryStore.hybrid_search` / `_load_all_chunks` (only diary `YYYY-MM-DD.jsonl` + MEMORY.md).
- System prompt **Memory Instructions** (`agent/agent_.py`) tell the model to persist durable facts via `memory.write` and not to rely on retrieving trimmed raw dialogue.
