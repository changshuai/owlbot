from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agent import agent_loop as agent_loop_module
from agent.tools import process_tool_call


class AgentLoopToolExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, assistant_message: dict, messages: list[dict], **kwargs):
        out = []
        async for ev in agent_loop_module._execute_tool_calls(assistant_message, messages, **kwargs):
            out.append(ev)
        return out

    async def test_dedups_duplicate_toolcall_id(self) -> None:
        assistant = {
            "content": [
                {"type": "toolCall", "id": "c1", "name": "fileOps", "arguments": {"action": "read", "file_path": "x"}},
                {"type": "toolCall", "id": "c1", "name": "fileOps", "arguments": {"action": "read", "file_path": "x"}},
            ]
        }
        messages: list[dict] = []
        with patch.object(agent_loop_module, "process_tool_call", return_value="ok") as proc:
            events = await self._collect(assistant, messages)
        self.assertEqual(proc.call_count, 1)
        self.assertEqual(len([e for e in events if e["type"] == "tool_call_start"]), 1)
        self.assertEqual(len([e for e in events if e["type"] == "tool_call_end"]), 1)
        self.assertEqual(len(messages), 1)

    async def test_marks_error_when_tool_returns_error_prefix(self) -> None:
        assistant = {"content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "x"}}]}
        messages: list[dict] = []
        with patch.object(agent_loop_module, "process_tool_call", return_value="Error: failed"):
            await self._collect(assistant, messages)
        self.assertTrue(messages[0]["isError"])

    async def test_marks_error_when_tool_raises(self) -> None:
        assistant = {"content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "x"}}]}
        messages: list[dict] = []
        with patch.object(agent_loop_module, "process_tool_call", side_effect=RuntimeError("boom")):
            await self._collect(assistant, messages)
        self.assertTrue(messages[0]["isError"])
        self.assertIn("Error: boom", messages[0]["content"][0]["text"])

    async def test_strips_whitespace_in_tool_name(self) -> None:
        assistant = {
            "content": [
                {"type": "toolCall", "id": "c1", "name": "bash  ", "arguments": {"command": "echo hi"}},
            ]
        }
        messages: list[dict] = []
        with patch.object(agent_loop_module, "process_tool_call", return_value="ok") as proc:
            await self._collect(assistant, messages)
        proc.assert_called_once()
        self.assertEqual(proc.call_args[0][0], "bash")
        self.assertEqual(messages[0]["toolName"], "bash")

    async def test_shrinks_oversized_text_via_truncate(self) -> None:
        assistant = {"content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "x"}}]}
        messages: list[dict] = [{"role": "user", "content": "q"}]
        long = "x" * 100
        with (
            patch.object(agent_loop_module, "TOOL_RESULT_MAX_CHARS", 40),
            patch.object(agent_loop_module, "shrink_tool_result_text", new_callable=AsyncMock) as shrink,
            patch.object(agent_loop_module, "process_tool_call", return_value=long),
        ):
            shrink.return_value = ("shrunk", "truncate_head_tail")
            await self._collect(assistant, messages)
        shrink.assert_awaited_once()
        self.assertEqual(messages[1]["content"][0]["text"], "shrunk")
        self.assertEqual(messages[1]["details"]["shrink_method"], "truncate_head_tail")

    async def test_abort_signal_breaks_before_processing(self) -> None:
        assistant = {"content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "x"}}]}
        messages: list[dict] = []
        abort = asyncio.Event()
        abort.set()
        with patch.object(agent_loop_module, "process_tool_call") as proc:
            events = await self._collect(assistant, messages, abort_signal=abort)
        proc.assert_not_called()
        self.assertEqual(events, [])
        self.assertEqual(messages, [])


class TestProcessToolCallNameNormalization(unittest.TestCase):
    def test_strips_whitespace_for_lookup(self) -> None:
        from agent.tools import TOOL_HANDLERS

        def fake_bash(tool_ctx=None, **kwargs):
            return "ok"

        with patch.dict(TOOL_HANDLERS, {"bash": fake_bash}):
            out = process_tool_call("  bash  ", {"command": "echo"})
        self.assertEqual(out, "ok")


if __name__ == "__main__":
    unittest.main()
