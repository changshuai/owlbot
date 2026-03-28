"""Unit tests for agent.context_compaction helpers."""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from llms.modelbase import make_empty_assistant_message

from agent import context_compaction as cc


class TestEstimateAndTruncate(unittest.TestCase):
    def test_estimate_message_chars_user_string(self) -> None:
        m = {"role": "user", "content": "abc"}
        self.assertEqual(cc.estimate_message_chars(m), 3)

    def test_estimate_message_chars_assistant(self) -> None:
        m = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}, {"type": "toolCall", "id": "1", "name": "x", "arguments": {}}],
        }
        self.assertEqual(cc.estimate_message_chars(m), 5)

    def test_estimate_context_chars(self) -> None:
        msgs = [
            {"role": "user", "content": "aa"},
            {"role": "assistant", "content": [{"type": "text", "text": "bb"}]},
        ]
        self.assertEqual(cc.estimate_context_chars("sys", msgs), len("sys") + 2 + 2)

    def test_truncate_head_tail(self) -> None:
        s = "a" * 40 + "MID" + "b" * 40
        out = cc.truncate_head_tail(s, 80)
        self.assertIn("[middle omitted]", out)

    def test_truncate_head_tail_short_max(self) -> None:
        self.assertEqual(cc.truncate_head_tail("hello", 100), "hello")

    def test_latest_user_text(self) -> None:
        msgs = [
            {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
            {"role": "user", "content": "last"},
        ]
        self.assertEqual(cc.latest_user_text(msgs), "last")

    def test_message_suffix_is_valid(self) -> None:
        good = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": [{"type": "toolCall", "id": "1", "name": "x", "arguments": {}}]},
            {"role": "toolResult", "toolCallId": "1", "toolName": "x", "content": [{"type": "text", "text": "r"}]},
        ]
        self.assertTrue(cc._message_suffix_is_valid(good))
        bad = [{"role": "toolResult", "toolCallId": "1", "toolName": "x", "content": [{"type": "text", "text": "r"}]}]
        self.assertFalse(cc._message_suffix_is_valid(bad))


class TestShrinkToolResult(unittest.IsolatedAsyncioTestCase):
    async def test_noop_when_short(self) -> None:
        text, method = await cc.shrink_tool_result_text(
            tool_name="bash",
            raw_text="short",
            user_question="q",
        )
        self.assertEqual(text, "short")
        self.assertEqual(method, "none")

    async def test_truncates_without_summarizer(self) -> None:
        with patch.object(cc, "SUMMARIZER_MODEL_ID", ""):
            long = "Z" * (cc.TOOL_RESULT_MAX_CHARS + 500)
            text, method = await cc.shrink_tool_result_text(
                tool_name="bash",
                raw_text=long,
                user_question="q",
            )
        self.assertEqual(method, "truncate_head_tail")
        self.assertEqual(len(text), cc.TOOL_RESULT_MAX_CHARS)
        self.assertIn("[middle omitted]", text)

    async def test_summarize_when_configured(self) -> None:
        folded = make_empty_assistant_message("api", "p", "m")
        folded["content"] = [{"type": "text", "text": "compact summary line"}]

        fake_model = MagicMock()
        fake_model.invoke = AsyncMock(return_value=folded)

        long = "Q" * (cc.TOOL_RESULT_MAX_CHARS + 100)
        with (
            patch.object(cc, "SUMMARIZER_MODEL_ID", "small/model"),
            patch.object(cc, "get_model_for_id", return_value=fake_model),
        ):
            text, method = await cc.shrink_tool_result_text(
                tool_name="bash",
                raw_text=long,
                user_question="q",
            )
        self.assertEqual(method, "summarize")
        self.assertEqual(text, "compact summary line")


class TestHistoryBudgetChars(unittest.TestCase):
    def test_caps_at_context_max_tokens(self) -> None:
        fake = MagicMock()
        fake.context_window = 200_000
        with patch.object(cc, "CONTEXT_BUDGET_RATIO", 0.72), patch.object(cc, "CONTEXT_MAX_TOKENS", 16000):
            b = cc.history_budget_chars(fake)
        self.assertEqual(b, 16000 * 4)

    def test_ratio_smaller_than_cap_uses_ratio(self) -> None:
        fake = MagicMock()
        fake.context_window = 4000
        with patch.object(cc, "CONTEXT_BUDGET_RATIO", 0.72), patch.object(cc, "CONTEXT_MAX_TOKENS", 16000):
            b = cc.history_budget_chars(fake)
        expected = max(1024, int(4000 * 0.72 * 4))
        self.assertEqual(b, expected)
        self.assertLess(b, 16000 * 4)

    def test_zero_max_tokens_disables_cap(self) -> None:
        fake = MagicMock()
        fake.context_window = 10_000
        with patch.object(cc, "CONTEXT_BUDGET_RATIO", 0.5), patch.object(cc, "CONTEXT_MAX_TOKENS", 0):
            b = cc.history_budget_chars(fake)
        self.assertEqual(b, max(1024, int(10_000 * 0.5 * 4)))


class TestMaybeTrimHistory(unittest.IsolatedAsyncioTestCase):
    async def test_skips_when_no_window(self) -> None:
        fake = MagicMock()
        fake.context_window = 0
        msgs = [{"role": "user", "content": "x" * 1000}]
        with patch.object(cc, "get_model_for_id", return_value=fake):
            await cc.maybe_trim_history_for_budget("m", "sys", msgs)
        self.assertEqual(len(msgs), 1)

    async def test_skips_when_below_budget(self) -> None:
        fake = MagicMock()
        fake.context_window = 100_000
        msgs = [{"role": "user", "content": "hi"}]
        with patch.object(cc, "get_model_for_id", return_value=fake):
            await cc.maybe_trim_history_for_budget("m", "sys", msgs)
        self.assertEqual(len(msgs), 1)

    async def test_trims_prefix_archives_placeholder(self) -> None:
        fake = MagicMock()
        fake.context_window = 200
        mgr = MagicMock()
        store = MagicMock()
        store.archive_messages.return_value = "memory/sess/context_archive_x.jsonl"
        mgr.get_memory_store = None
        tool_ctx: dict[str, Any] = {
            "_mgr": mgr,
            "session_key": "s1",
            "agent_id": "ag1",
        }

        chunk = "y" * 200
        msgs = [{"role": "user", "content": chunk} for _ in range(4)]

        with (
            patch.object(cc, "get_model_for_id", return_value=fake),
            patch.object(cc, "CONTEXT_BUDGET_RATIO", 0.5),
            patch("agent.memory_store.get_memory_store", return_value=store),
        ):
            await cc.maybe_trim_history_for_budget("m", "s" * 400, msgs, tool_ctx=tool_ctx)

        self.assertGreater(len(msgs), 0)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("Archive:", msgs[0]["content"])
        self.assertTrue(msgs[0].get("details", {}).get("context_trim"))
        store.archive_messages.assert_called_once()


if __name__ == "__main__":
    unittest.main()
