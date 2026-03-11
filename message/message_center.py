"""
Channel bridge: poll registered channels for InboundMessage, resolve route (binding),
run agent, send reply back to the same channel. Use with Gateway so that when the
main program starts, channels (e.g. WhatsApp Web) are monitored and only binding-
matched conversations are handled; unconfigured group messages are not processed
(channel filters them before enqueue).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, List

from channels.types_ import Channel, InboundMessage
from .route_ import AgentManager, BindingTable, resolve_route
from .agent_ import Agent, AgentManager
from .agent_loop import run_agent

logger = logging.getLogger(__name__)


def run_async(coro):
    """Default: run in new loop. Override with main's run_async for shared loop."""
    import asyncio
    return asyncio.run(coro)


class MessageCenter:
    """
    Polls a list of channels; for each InboundMessage, resolves agent via bindings,
    runs the agent, and sends the reply back to the channel. Only processes messages
    that the channel has already filtered (e.g. WhatsApp Web: only DMs + allowed_groups).
    """

    def __init__(
        self,
        mgr: AgentManager,
        bindings: BindingTable,
        channels: List[Channel],
        run_async_fn: Callable | None = None,
        poll_interval: float = 3.0,
    ) -> None:
        self._mgr = mgr
        self._bindings = bindings
        self._channels = list(channels)
        self._run_async = run_async_fn or run_async
        self._poll_interval = max(0.1, poll_interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("ChannelBridge started with %d channel(s)", len(self._channels))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        import time
        while not self._stop.is_set():
            for ch in self._channels:
                self.dispatch(ch)
            time.sleep(self._poll_interval)

    def dispatch(self, ch: Channel) -> None:
        """
        Poll one channel once: receive a message (if any) and handle it.
        Used by the background loop; CLI can call handle_message() directly.
        """
        try:
            msg = ch.receive()
        except Exception as e:
            logger.debug("channel %s receive error: %s", getattr(ch, "name", ch), e)
            return
        if not msg or not isinstance(msg, InboundMessage):
            return

        self.handle_message(msg, ch)

    def handle_message(self, msg: InboundMessage, ch: Channel) -> None:
        """
        Core dispatch logic for a single already-received InboundMessage.
        This is reusable from CLI or tests without going through receive().
        """
        try:
            agent_id, session_key = resolve_route(
                self._bindings,
                self._mgr,
                channel=msg.channel,
                peer_id=msg.peer_id,
                account_id=msg.account_id or "",
                guild_id="",
            )
            reply = self._run_async(
                run_agent(self._mgr, agent_id, session_key, msg.text)
            )
            ch.send(msg.peer_id, reply or "")
        except Exception as e:
            logger.exception("MessageDispatcher dispatch %s: %s", msg.peer_id, e)
            try:
                ch.send(msg.peer_id, f"Error: {e}")
            except Exception:
                pass
