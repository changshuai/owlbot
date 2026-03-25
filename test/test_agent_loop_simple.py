from __future__ import annotations

import asyncio
import unittest
from typing import Any, AsyncIterator

from LLMs.modelbase import make_empty_assistant_message
from agent import agent_loop as agent_loop_module
from agent.agent_abort import AgentAbortController


class FakeStreamModelText:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, context: Any, options: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.calls += 1
        msg = make_empty_assistant_message("test-api", "test", "fake-model")
        yield {"type": "start", "partial": dict(msg)}
        msg["content"] = [{"type": "text", "text": "hello"}]
        yield {"type": "text_delta", "delta": "hello", "partial": dict(msg)}
        yield {"type": "done", "reason": "stop", "message": msg}


class FakeStreamModelToolThenText:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, context: Any, options: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self.calls += 1
        msg = make_empty_assistant_message("test-api", "test", "fake-model")
        yield {"type": "start", "partial": dict(msg)}
        if self.calls == 1:
            msg["stopReason"] = "toolUse"
            msg["content"] = [
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": "unknown_tool",
                    "arguments": {"x": 1},
                }
            ]
            yield {"type": "done", "reason": "toolUse", "message": msg}
        else:
            msg["stopReason"] = "stop"
            msg["content"] = [{"type": "text", "text": "final"}]
            yield {"type": "done", "reason": "stop", "message": msg}


class AgentLoopSimpleTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_agent_streams_text_and_returns_reply(self) -> None:
        fake_model = FakeStreamModelText()
        original_get_model = agent_loop_module._get_model_for_id
        agent_loop_module._get_model_for_id = lambda _model_id: fake_model
        try:
            messages = [{"role": "user", "content": "hi"}]
            seen_events: list[dict[str, Any]] = []

            async for ev in agent_loop_module._agent_loop("fake-model", "sys", messages):
                seen_events.append(ev)

            result = await agent_loop_module._run_agent("fake-model", "sys", [{"role": "user", "content": "hi"}])
        finally:
            agent_loop_module._get_model_for_id = original_get_model

        self.assertTrue(any(e.get("type") == "message_start" for e in seen_events))
        self.assertTrue(any(e.get("type") == "message_update" for e in seen_events))
        self.assertTrue(any(e.get("type") == "message_end" for e in seen_events))
        self.assertEqual(seen_events[-1].get("type"), "model_end")
        self.assertEqual(result, "hello")

    async def test_run_agent_handles_tool_call_then_finishes(self) -> None:
        fake_model = FakeStreamModelToolThenText()
        original_get_model = agent_loop_module._get_model_for_id
        agent_loop_module._get_model_for_id = lambda _model_id: fake_model
        try:
            messages = [{"role": "user", "content": "hi"}]
            seen_events: list[dict[str, Any]] = []

            async for ev in agent_loop_module._agent_loop("fake-model", "sys", messages):
                seen_events.append(ev)
        finally:
            agent_loop_module._get_model_for_id = original_get_model

        event_types = [e.get("type") for e in seen_events]
        self.assertIn("tool_call_start", event_types)
        self.assertIn("tool_call_end", event_types)
        self.assertEqual(fake_model.calls, 2)
        self.assertEqual(seen_events[-1].get("type"), "model_end")

    async def test_abort_signal_stops_before_model_stream(self) -> None:
        """Pre-set abort: run_begin then model_end(aborted); fake model.stream never runs."""
        fake_model = FakeStreamModelText()
        original_get_model = agent_loop_module._get_model_for_id
        agent_loop_module._get_model_for_id = lambda _model_id: fake_model

        abort = asyncio.Event()
        abort.set()

        try:
            messages = [{"role": "user", "content": "hi"}]
            seen_events: list[dict[str, Any]] = []

            async for ev in agent_loop_module._agent_loop(
                "fake-model", "sys", messages, abort_signal=abort
            ):
                seen_events.append(ev)
        finally:
            agent_loop_module._get_model_for_id = original_get_model

        self.assertEqual(fake_model.calls, 0)
        self.assertGreaterEqual(len(seen_events), 2)
        self.assertEqual(seen_events[0].get("type"), "model_start")
        self.assertTrue(seen_events[0].get("run_begin"))
        self.assertEqual(seen_events[-1].get("type"), "model_end")
        self.assertTrue(seen_events[-1].get("aborted"))
        self.assertTrue(seen_events[-1].get("run_done"))

    async def test_run_agent_returns_aborted_when_signal_set(self) -> None:
        fake_model = FakeStreamModelText()
        original_get_model = agent_loop_module._get_model_for_id
        agent_loop_module._get_model_for_id = lambda _model_id: fake_model

        ctrl = AgentAbortController()
        ctrl.abort()

        try:
            out = await agent_loop_module._run_agent(
                "fake-model",
                "sys",
                [{"role": "user", "content": "hi"}],
                abort_signal=ctrl.signal,
            )
        finally:
            agent_loop_module._get_model_for_id = original_get_model

        self.assertEqual(fake_model.calls, 0)
        self.assertEqual(out, "[aborted]")


if __name__ == "__main__":
    unittest.main()
