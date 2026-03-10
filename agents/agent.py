"""
Stateful agent with tool execution and event streaming. Built on ai_models.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any, Awaitable, Callable

from ai_models import Context
from ai_models.modelbase import Model
from ai_models.types import SimpleStreamOptions, ThinkingLevel

from .agent_loop import agent_loop
from .types import AgentContext, AgentEvent, make_user_message


def _validate_tool_object(tool: Any) -> None:
    """
    Perform a lightweight runtime validation of a tool passed to Agent.

    Requirements:
    - has string attributes: name, description
    - has dict attribute: parameters (JSON Schema)
    - has async callable: execute(tool_call_id: str, params: dict) -> AgentToolResult
    """
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("Agent tools must have a non-empty 'name' string attribute")

    description = getattr(tool, "description", None)
    if not isinstance(description, str):
        raise TypeError(f"Agent tool '{name}' must have a 'description' string attribute")

    parameters = getattr(tool, "parameters", None)
    if not isinstance(parameters, dict):
        raise TypeError(f"Agent tool '{name}' must have a 'parameters' dict (JSON Schema)")

    execute = getattr(tool, "execute", None)
    if execute is None or not callable(execute):
        raise TypeError(f"Agent tool '{name}' must define an async 'execute(tool_call_id, params)' method")
    if not inspect.iscoroutinefunction(execute):
        raise TypeError(f"Agent tool '{name}'.execute must be an async function")


class Agent:
    """
    Agent with message history, optional tools, and event streaming.
    Uses ai_models Model for stream/invoke.
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Model,
        tools: list[Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        thinking_level: ThinkingLevel = "off",
        thinking_budgets: dict[str, int] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._tools: list[Any] = []
        if tools:
            # Validate tool objects eagerly to catch misconfiguration early.
            self.set_tools(tools)
        self._messages = list(messages) if messages else []
        self._subscribers: list[Callable[[AgentEvent], None]] = []
        self._is_streaming = False
        self._stream_message: dict[str, Any] | None = None
        self._pending_tool_calls: set[str] = set()
        self._error: str | None = None
        self._thinking_level: ThinkingLevel = thinking_level
        self._thinking_budgets: dict[str, int] = dict(thinking_budgets or {})
        self._session_id: str | None = session_id
        self._transform_context: Callable[
            [list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]
        ] | None = None
        # Custom stream function; should return an async iterator of events
        self._stream_fn: Callable[[Model, Context, SimpleStreamOptions], Any] | None = None
        self._current_task: asyncio.Task[Any] | None = None
        self._debug: bool = os.environ.get("PI_PY_AGENT_DEBUG") == "1"
        # Queues for steering and follow-up messages (for higher-level UIs)
        self._steering_queue: list[dict[str, Any]] = []
        self._followup_queue: list[dict[str, Any]] = []
        self._abort_signal: asyncio.Event | None = None

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def model(self) -> Model:
        return self._model

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def stream_message(self) -> dict[str, Any] | None:
        return self._stream_message

    @property
    def pending_tool_calls(self) -> set[str]:
        return self._pending_tool_calls

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    @property
    def thinking_budgets(self) -> dict[str, int]:
        return self._thinking_budgets

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def set_system_prompt(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    def set_model(self, model: Model) -> None:
        self._model = model

    def set_tools(self, tools: list[Any]) -> None:
        validated: list[Any] = []
        for t in tools:
            _validate_tool_object(t)
            validated.append(t)
        self._tools = validated

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._thinking_level = level

    def set_thinking_budgets(self, budgets: dict[str, int]) -> None:
        self._thinking_budgets = dict(budgets)

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = session_id

    def append_message(self, message: dict[str, Any]) -> None:
        self._messages.append(message)

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)

    def clear_messages(self) -> None:
        self._messages.clear()

    def reset(self) -> None:
        self.clear_messages()
        self._error = None
        self.clear_all_queues()

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> Callable[[], None]:
        """Subscribe to agent events. Returns an unsubscribe function."""

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        self._subscribers.append(callback)
        return unsubscribe

    def _emit(self, event: AgentEvent) -> None:
        if self._debug:
            et = event.get("type")
            msg = event.get("message")
            print(f"[Agent DEBUG] event={et}, role={getattr(msg, 'get', lambda *_: None)('role') if isinstance(msg, dict) else None}")  # noqa: T201,E501
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

    # --- Steering / follow-up queues (for richer Agent UIs) ---

    def steer(self, message: dict[str, Any]) -> None:
        """Queue a steering message to be injected on the next turn."""
        self._steering_queue.append(message)

    def follow_up(self, message: dict[str, Any]) -> None:
        """Queue a follow-up message to be processed after the current work finishes."""
        self._followup_queue.append(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_followup_queue(self) -> None:
        self._followup_queue.clear()

    def clear_all_queues(self) -> None:
        self._steering_queue.clear()
        self._followup_queue.clear()

    def set_transform_context(
        self,
        fn: Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]] | None,
    ) -> None:
        """
        Set a coroutine that transforms the Agent message list before each LLM call.
        Used for context window management, external context injection, etc.
        """
        self._transform_context = fn

    def set_stream_fn(
        self,
        fn: Callable[[Model, Context, SimpleStreamOptions], Any] | None,
    ) -> None:
        """
        Set a custom stream function. If set, called instead of model.stream(context, options).
        Useful for proxy backends.
        """
        self._stream_fn = fn

    def abort(self) -> None:
        """Cooperatively abort the current run (and underlying model stream) if any."""
        if self._abort_signal is not None:
            self._abort_signal.set()

    async def wait_for_idle(self) -> None:
        """Wait until the agent is no longer streaming."""
        task = self._current_task
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _build_options(self, options: SimpleStreamOptions | None) -> SimpleStreamOptions:
        opts: SimpleStreamOptions = dict(options or {})
        if self._thinking_level:
            opts.setdefault("reasoning", self._thinking_level)
        if self._thinking_budgets:
            opts.setdefault("thinking_budgets", self._thinking_budgets)
        if self._session_id:
            opts.setdefault("session_id", self._session_id)
        return opts

    async def _run_loop(
        self,
        prompts: list[dict[str, Any]],
        options: SimpleStreamOptions | None = None,
    ) -> list[dict[str, Any]]:
        self._current_task = asyncio.current_task()
        self._abort_signal = asyncio.Event()

        async def _get_steering_messages() -> list[dict[str, Any]]:
            if not self._steering_queue:
                return []
            # One-at-a-time semantics
            msg = self._steering_queue.pop(0)
            return [msg]

        async def _get_followup_messages() -> list[dict[str, Any]]:
            if not self._followup_queue:
                return []
            msg = self._followup_queue.pop(0)
            return [msg]

        context = AgentContext(
            messages=list(self._messages),
            system_prompt=self._system_prompt,
            tools=self._tools if self._tools else None,
        )
        opts = self._build_options(options)
        self._is_streaming = True
        self._error = None

        try:
            async for event in agent_loop(
                prompts,
                context,
                self._model,
                opts,
                transform_context=self._transform_context,
                stream_fn=self._stream_fn,
                signal=self._abort_signal,
                get_steering_messages=_get_steering_messages,
                get_followup_messages=_get_followup_messages,
            ):
                ev_type = event.get("type")
                if ev_type == "message_start":
                    msg = event.get("message")
                    if msg and msg.get("role") == "assistant":
                        self._stream_message = msg
                elif ev_type == "message_update":
                    self._stream_message = event.get("message")
                elif ev_type == "message_end":
                    self._stream_message = None
                elif ev_type == "tool_execution_start":
                    self._pending_tool_calls.add(event.get("tool_call_id", ""))
                elif ev_type == "tool_execution_end":
                    self._pending_tool_calls.discard(event.get("tool_call_id", ""))
                elif ev_type == "turn_end":
                    msg = event.get("message")
                    if msg and msg.get("role") == "assistant" and msg.get("errorMessage"):
                        self._error = msg.get("errorMessage")
                elif ev_type == "agent_end":
                    # Sync state with final messages from event
                    final = event.get("messages")
                    if final is not None:
                        self._messages = list(final)
                self._emit(event)
        finally:
            self._is_streaming = False
            self._stream_message = None
            self._current_task = None
            self._abort_signal = None

        return self._messages

    async def prompt(
        self,
        input: str | dict[str, Any] | list[dict[str, Any]],
        options: SimpleStreamOptions | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add one or more messages and run the agent loop (stream assistant, execute tools, repeat).
        Emits events to subscribers. Returns the updated messages list when done.

        input: A string (single user message), a single message dict, or a list of message dicts.
        """
        if self._is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer/followUp to queue messages, or wait for completion."
            )
        if not self._model:
            raise RuntimeError("No model configured")

        if isinstance(input, str):
            prompts_list: list[dict[str, Any]] = [make_user_message(input)]
        elif isinstance(input, dict):
            prompts_list = [input]
        else:
            prompts_list = list(input)

        return await self._run_loop(prompts_list, options)

    async def continue_(self, options: SimpleStreamOptions | None = None) -> list[dict[str, Any]]:
        """
        Continue from current context (used for retries and queued messages).

        - If last message is user/toolResult: continue without adding a new message.
        - If last message is assistant:
            - If steering queue has messages, run a new turn with those messages.
            - Else if follow-up queue has messages, run a new turn with those messages.
            - Else, raise an error (cannot continue from assistant).
        """
        if self._is_streaming:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")
        if not self._model:
            raise RuntimeError("No model configured")

        messages = self._messages
        if not messages:
            raise RuntimeError("No messages to continue from")
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant":
            # Prefer steering messages when present
            if self._steering_queue:
                prompts = [self._steering_queue.pop(0)]
                return await self._run_loop(prompts, options)
            # Then fall back to follow-up messages
            if self._followup_queue:
                prompts = [self._followup_queue.pop(0)]
                return await self._run_loop(prompts, options)
            raise RuntimeError("Cannot continue from message role: assistant")

        # Last message is user/toolResult: just continue with existing context
        return await self._run_loop([], options)

