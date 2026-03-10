from __future__ import annotations

import asyncio
import sys
from pathlib import Path
import unittest
from typing import Any, AsyncIterator

# Allow importing ai_models/agent when running tests directly from python/
_src = Path(__file__).resolve().parent.parent
if _src not in sys.path:
    sys.path.insert(0, str(_src))

from ai_models.modelbase import Model, make_empty_assistant_message

from agent import Agent
from agent.agent_loop import agent_loop
from agent.simple_agent import SimpleAgent
from agent.types import AgentContext, AgentTool, AgentToolResult, agent_tool, make_user_message


class FakeModelNoTools(Model):
    """Fake model that returns a single assistant message with text only."""

    async def stream(
        self,
        context,  # Context type not imported to keep test simple
        options,
    ) -> AsyncIterator[dict[str, Any]]:
        output = make_empty_assistant_message(self.api, self.provider, self.id)
        output["content"] = [{"type": "text", "text": "hi"}]
        yield {"type": "start", "partial": dict(output)}
        yield {"type": "done", "reason": "stop", "message": output}


class FakeModelToolCall(Model):
    """Fake model that first issues a toolCall, then returns plain text."""

    def __init__(self, *args: Any, tool_name: str = "echo", tool_args: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.tool_name = tool_name
        self.tool_args = dict(tool_args or {})
        self.calls = 0

    async def stream(
        self,
        context,
        options,
    ) -> AsyncIterator[dict[str, Any]]:
        self.calls += 1
        output = make_empty_assistant_message(self.api, self.provider, self.id)
        if self.calls == 1:
            # First call: request a tool execution
            output["stopReason"] = "toolUse"
            output["content"] = [
                {
                    "type": "toolCall",
                    "id": "call-1",
                    "name": self.tool_name,
                    "arguments": dict(self.tool_args),
                }
            ]
            yield {"type": "start", "partial": dict(output)}
            yield {"type": "done", "reason": "toolUse", "message": output}
        else:
            # Second and subsequent calls: plain answer, no more tool calls
            output["stopReason"] = "stop"
            output["content"] = [{"type": "text", "text": "done"}]
            yield {"type": "start", "partial": dict(output)}
            yield {"type": "done", "reason": "stop", "message": output}


class EchoTool:
    name = "echo"
    description = "Echo the given message"
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
        "required": ["message"],
    }

    async def execute(self, tool_call_id: str, params: dict[str, Any]) -> AgentToolResult:
        return {
            "content": [{"type": "text", "text": params.get("message", "")}],
            "details": {"tool_call_id": tool_call_id},
        }


@agent_tool()
async def decorated_echo(message: str) -> AgentToolResult:
    """Echo back the given message."""
    return {
        "content": [{"type": "text", "text": message}],
        "details": {},
    }


class FakeModelCountingCalls(FakeModelNoTools):
    """Fake model that returns different text per call count (for multi-turn tests)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    async def stream(self, context, options) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        self.calls += 1
        output = make_empty_assistant_message(self.api, self.provider, self.id)
        output["content"] = [{"type": "text", "text": f"response-{self.calls}"}]
        yield {"type": "start", "partial": dict(output)}
        yield {"type": "done", "reason": "stop", "message": output}


class FakeModelStopReasonError(Model):
    """Fake model that yields done with stopReason error/aborted."""

    def __init__(self, stop_reason: str = "error", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stop_reason = stop_reason

    async def stream(self, context, options) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        output = make_empty_assistant_message(self.api, self.provider, self.id)
        output["stopReason"] = self._stop_reason
        output["content"] = [{"type": "text", "text": "failed"}]
        yield {"type": "start", "partial": dict(output)}
        yield {"type": "done", "reason": self._stop_reason, "message": output}


def _make_model(**kwargs: Any) -> Model:
    defaults = {
        "id": "test-model",
        "name": "Test",
        "provider": "test",
        "base_url": "",
        "api_key": None,
        "context_window": 1024,
        "max_tokens": 256,
        "api": "test-api",
    }
    defaults.update(kwargs)
    return FakeModelNoTools(**defaults)


def _make_tool_call_model(**kwargs: Any) -> FakeModelToolCall:
    defaults = {
        "id": "test-model",
        "name": "Test",
        "provider": "test",
        "base_url": "",
        "api_key": None,
        "context_window": 1024,
        "max_tokens": 256,
        "api": "test-api",
    }
    defaults.update(kwargs)
    return FakeModelToolCall(**defaults)


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    """Tests for agent_loop (event order, steering, follow-up, error paths)."""

    async def test_event_sequence_no_tools(self) -> None:
        model = _make_model()
        ctx = AgentContext(messages=[], system_prompt="", tools=None)
        prompts = [make_user_message("hi")]

        events: list[dict[str, Any]] = []
        async for ev in agent_loop(prompts, ctx, model):
            events.append(ev)

        types = [e.get("type") for e in events]
        self.assertEqual(events[0].get("type"), "agent_start")
        self.assertIn("turn_start", types)
        self.assertIn("message_start", types)
        self.assertIn("message_end", types)
        self.assertIn("turn_end", types)
        self.assertEqual(events[-1].get("type"), "agent_end")
        self.assertIn("messages", events[-1])

    async def test_steering_injects_before_next_llm_call(self) -> None:
        model = _make_tool_call_model(tool_name="echo", tool_args={"message": "ok"})
        ctx = AgentContext(messages=[], system_prompt="", tools=[EchoTool()])
        prompts = [make_user_message("call echo")]

        steering_returned = []

        async def get_steering() -> list[dict[str, Any]]:
            if len(steering_returned) == 0:
                steering_returned.append(1)
                return [{"role": "user", "content": "continue"}]
            return []

        events: list[dict[str, Any]] = []
        async for ev in agent_loop(
            prompts,
            ctx,
            model,
            get_steering_messages=get_steering,
        ):
            events.append(ev)

        self.assertEqual(model.calls, 2)
        turn_starts = [e for e in events if e.get("type") == "turn_start"]
        self.assertGreaterEqual(len(turn_starts), 2)
        user_contents = [
            e.get("message", {}).get("content")
            for e in events
            if e.get("type") == "message_start" and e.get("message", {}).get("role") == "user"
        ]
        self.assertIn("continue", user_contents)
        self.assertEqual(events[-1].get("type"), "agent_end")

    async def test_followup_processes_after_turn_ends(self) -> None:
        model = FakeModelCountingCalls(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        ctx = AgentContext(messages=[], system_prompt="", tools=None)
        prompts = [make_user_message("first")]

        followup_sent: list[dict[str, Any]] = []

        async def get_followup() -> list[dict[str, Any]]:
            if not followup_sent:
                followup_sent.append(1)
                return [make_user_message("second question")]
            return []

        events: list[dict[str, Any]] = []
        async for ev in agent_loop(
            prompts,
            ctx,
            model,
            get_followup_messages=get_followup,
        ):
            events.append(ev)

        self.assertEqual(model.calls, 2)
        turn_starts = [e for e in events if e.get("type") == "turn_start"]
        self.assertEqual(len(turn_starts), 2)
        final = events[-1]
        self.assertEqual(final.get("type"), "agent_end")
        messages = final.get("messages") or []
        assistant_texts = [
            c.get("text", "")
            for m in messages
            if m.get("role") == "assistant"
            for c in (m.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        self.assertIn("response-1", assistant_texts)
        self.assertIn("response-2", assistant_texts)

    async def test_error_stop_reason_ends_loop(self) -> None:
        model = FakeModelStopReasonError(
            stop_reason="error",
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        ctx = AgentContext(messages=[], system_prompt="", tools=None)
        prompts = [make_user_message("hi")]

        events: list[dict[str, Any]] = []
        async for ev in agent_loop(prompts, ctx, model):
            events.append(ev)

        self.assertEqual(events[-2].get("type"), "turn_end")
        self.assertEqual(events[-1].get("type"), "agent_end")

    async def test_aborted_stop_reason_ends_loop(self) -> None:
        model = FakeModelStopReasonError(
            stop_reason="aborted",
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        ctx = AgentContext(messages=[], system_prompt="", tools=None)
        prompts = [make_user_message("hi")]

        events: list[dict[str, Any]] = []
        async for ev in agent_loop(prompts, ctx, model):
            events.append(ev)

        self.assertEqual(events[-1].get("type"), "agent_end")


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_no_tools(self) -> None:
        model = FakeModelNoTools(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        agent = Agent(system_prompt="You are helpful.", model=model)

        events: list[dict[str, Any]] = []
        agent.subscribe(lambda ev: events.append(ev))

        messages = await agent.prompt("Hello")

        # Expect at least one assistant message with text
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        self.assertGreaterEqual(len(assistant_messages), 1)
        last = assistant_messages[-1]
        self.assertEqual(last.get("content")[0].get("type"), "text")
        self.assertIn("hi", last.get("content")[0].get("text"))

        # Basic event sanity: agent_start and agent_end should be present
        types = [e.get("type") for e in events]
        self.assertIn("agent_start", types)
        self.assertIn("agent_end", types)

    async def test_tool_execution_success_with_validation(self) -> None:
        model = FakeModelToolCall(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
            tool_name="echo",
            tool_args={"message": "hello from tool"},
        )
        agent = Agent(system_prompt="You are helpful.", model=model, tools=[EchoTool()])

        messages = await agent.prompt("Call echo")

        tool_results = [m for m in messages if m.get("role") == "toolResult"]
        self.assertEqual(len(tool_results), 1)
        tr = tool_results[0]
        self.assertFalse(tr.get("isError"))
        self.assertEqual(tr.get("toolName"), "echo")
        content = tr.get("content") or []
        self.assertEqual(content[0].get("type"), "text")
        self.assertIn("hello from tool", content[0].get("text"))

    async def test_tool_execution_validation_error(self) -> None:
        # Missing required "message" argument triggers validation error
        model = FakeModelToolCall(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
            tool_name="echo",
            tool_args={},  # missing "message"
        )
        agent = Agent(system_prompt="You are helpful.", model=model, tools=[EchoTool()])

        messages = await agent.prompt("Call echo with bad args")

        tool_results = [m for m in messages if m.get("role") == "toolResult"]
        self.assertEqual(len(tool_results), 1)
        tr = tool_results[0]
        self.assertTrue(tr.get("isError"))
        content = tr.get("content") or []
        self.assertEqual(content[0].get("type"), "text")
        # Error message should mention a required property
        self.assertIn("required property", content[0].get("text"))

    async def test_continue_uses_existing_context(self) -> None:
        """When last message is user, continue_() runs the loop with no new prompts."""
        class CountingModel(FakeModelNoTools):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.calls = 0

            async def stream(self, context, options) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
                self.calls += 1
                async for ev in super().stream(context, options):
                    yield ev

        model = CountingModel(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        agent = Agent(system_prompt="You are helpful.", model=model)
        agent.replace_messages([make_user_message("First")])

        await agent.continue_()
        self.assertEqual(model.calls, 1)
        self.assertGreaterEqual(len(agent.messages), 2)

    async def test_agent_tool_decorator_generates_schema_and_executes(self) -> None:
        # Schema should have message: string and required ["message"]
        self.assertEqual(decorated_echo.parameters["type"], "object")
        self.assertIn("message", decorated_echo.parameters["properties"])
        self.assertEqual(
            decorated_echo.parameters["properties"]["message"].get("type"),
            "string",
        )
        self.assertIn("message", decorated_echo.parameters.get("required", []))

        # Use a fake model that calls the decorated tool
        model = FakeModelToolCall(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
            tool_name=decorated_echo.name,
            tool_args={"message": "from decorated tool"},
        )
        agent = Agent(system_prompt="You are helpful.", model=model, tools=[decorated_echo])

        messages = await agent.prompt("Call decorated echo")

        tool_results = [m for m in messages if m.get("role") == "toolResult"]
        self.assertEqual(len(tool_results), 1)
        tr = tool_results[0]
        self.assertFalse(tr.get("isError"))
        content = tr.get("content") or []
        self.assertEqual(content[0].get("type"), "text")
        self.assertIn("from decorated tool", content[0].get("text"))

    async def test_prompt_accepts_dict_message(self) -> None:
        model = _make_model()
        agent = Agent(system_prompt="", model=model)
        messages = await agent.prompt({"role": "user", "content": "hello"})
        assistant = [m for m in messages if m.get("role") == "assistant"]
        self.assertGreaterEqual(len(assistant), 1)
        self.assertIn("hi", (assistant[-1].get("content") or [{}])[0].get("text", ""))

    async def test_prompt_accepts_list_of_messages(self) -> None:
        model = _make_model()
        agent = Agent(system_prompt="", model=model)
        messages = await agent.prompt([
            make_user_message("first"),
            make_user_message("second"),
        ])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        self.assertGreaterEqual(len(user_msgs), 2)
        assistant = [m for m in messages if m.get("role") == "assistant"]
        self.assertGreaterEqual(len(assistant), 1)

    async def test_follow_up_processes_after_prompt_ends(self) -> None:
        model = FakeModelCountingCalls(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        agent = Agent(system_prompt="", model=model)
        agent.follow_up(make_user_message("second"))
        messages = await agent.prompt("first")
        assistant_texts = [
            c.get("text", "")
            for m in messages
            if m.get("role") == "assistant"
            for c in (m.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        self.assertIn("response-1", assistant_texts)
        self.assertIn("response-2", assistant_texts)

    async def test_continue_from_user_with_empty_prompts(self) -> None:
        """When last message is user, continue_() runs with empty prompts and model is invoked."""
        model = FakeModelCountingCalls(
            id="test-model",
            name="Test",
            provider="test",
            base_url="",
            api_key=None,
            context_window=1024,
            max_tokens=256,
            api="test-api",
        )
        agent = Agent(system_prompt="", model=model)
        agent.replace_messages([make_user_message("only one")])
        self.assertEqual(model.calls, 0)
        messages = await agent.continue_()
        self.assertEqual(model.calls, 1)
        self.assertGreaterEqual(len(messages), 2)
        assistant = [m for m in messages if m.get("role") == "assistant"]
        self.assertEqual(len(assistant), 1)
        self.assertIn("response-1", (assistant[0].get("content") or [{}])[0].get("text", ""))

    async def test_continue_from_assistant_without_queues_raises(self) -> None:
        model = _make_model()
        agent = Agent(system_prompt="", model=model)
        await agent.prompt("hi")
        with self.assertRaises(RuntimeError) as ctx:
            await agent.continue_()
        self.assertIn("assistant", str(ctx.exception))

    async def test_set_tools_rejects_invalid_tool(self) -> None:
        model = _make_model()
        agent = Agent(system_prompt="", model=model)

        class BadTool:
            name = "bad"
            description = "no execute"

        with self.assertRaises(TypeError):
            agent.set_tools([BadTool()])


class SimpleAgentTests(unittest.IsolatedAsyncioTestCase):
    """Tests for SimpleAgent (prompt, stream, clear_messages)."""

    async def test_prompt_returns_last_assistant_message(self) -> None:
        model = _make_model()
        agent = SimpleAgent(system_prompt="", model=model)
        msg = await agent.prompt("hello")
        self.assertEqual(msg.get("role"), "assistant")
        content = msg.get("content") or []
        self.assertGreater(len(content), 0)
        self.assertIn("hi", content[0].get("text", ""))

    async def test_prompt_updates_history(self) -> None:
        model = _make_model()
        agent = SimpleAgent(system_prompt="", model=model)
        self.assertEqual(len(agent.messages), 0)
        await agent.prompt("first")
        self.assertGreater(len(agent.messages), 0)
        await agent.prompt("second")
        user_count = sum(1 for m in agent.messages if m.get("role") == "user")
        self.assertGreaterEqual(user_count, 2)

    async def test_stream_yields_events_and_updates_history(self) -> None:
        model = _make_model()
        agent = SimpleAgent(system_prompt="", model=model)
        events: list[dict[str, Any]] = []
        async for ev in agent.stream("hi"):
            events.append(ev)
        types = [e.get("type") for e in events]
        self.assertIn("agent_start", types)
        self.assertIn("agent_end", types)
        self.assertEqual(events[-1].get("type"), "agent_end")
        self.assertGreater(len(agent.messages), 0)

    async def test_clear_messages_clears_history(self) -> None:
        model = _make_model()
        agent = SimpleAgent(system_prompt="", model=model)
        await agent.prompt("hello")
        self.assertGreater(len(agent.messages), 0)
        agent.clear_messages()
        self.assertEqual(len(agent.messages), 0)

    async def test_simple_agent_with_tool(self) -> None:
        model = _make_tool_call_model(tool_name="echo", tool_args={"message": "tool said this"})
        agent = SimpleAgent(system_prompt="", model=model, tools=[EchoTool()])
        msg = await agent.prompt("call echo")
        self.assertEqual(msg.get("role"), "assistant")
        content = msg.get("content") or []
        self.assertIn("done", content[0].get("text", ""))
        tool_results = [m for m in agent.messages if m.get("role") == "toolResult"]
        self.assertEqual(len(tool_results), 1)
        self.assertIn("tool said this", (tool_results[0].get("content") or [{}])[0].get("text", ""))


if __name__ == "__main__":
    asyncio.run(unittest.main())  # type: ignore[arg-type]

