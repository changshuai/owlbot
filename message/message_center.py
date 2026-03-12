"""
Channel bridge: poll registered channels for InboundMessage, resolve route (binding),
run agent, send reply back to the same channel. Use with Gateway so that when the
main program starts, channels (e.g. WhatsApp Web) are monitored and only binding-
matched conversations are handled; unconfigured group messages are not processed
(channel filters them before enqueue).
"""
import logging
import threading
import queue
from typing import Callable, List, Optional

from channels.types_ import Channel, InboundMessage
from .route_ import AgentManager, BindingTable, resolve_route
from agent.agent_ import Agent, AgentManager
from agent.agent_loop import run_agent

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
        run_async_fn: Optional[Callable] = None,
    ) -> None:
        self._mgr = mgr
        self._bindings = bindings
        self._channels = list(channels)
        self._run_async = run_async_fn or run_async
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Global inbound queue: channels act as producers, MessageCenter as consumer.
        self._queue: "queue.Queue[tuple[InboundMessage, Channel]]" = queue.Queue()

        # Register callbacks so push-style channels can enqueue messages directly.
        for ch in self._channels:
            if hasattr(ch, "set_inbound_callback"):
                try:
                    ch.set_inbound_callback(self._on_channel_inbound)  # type: ignore[arg-type]
                except Exception:
                    # If a channel doesn't support callbacks, it will still be polled via receive().
                    logger.debug("Channel %s does not support inbound callback", getattr(ch, "name", ch))
                # 显式启动有长期连接需求的渠道
            if hasattr(ch, "ensure_started"):
                try:
                    ch.ensure_started()
                except Exception as exc:
                    logger.debug("ensure_started failed for %s: %s", getattr(ch, "name", ch), exc)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("ChannelBridge started with %d channel(s)", len(self._channels))

    def stop(self) -> None:
        self._stop.set()
        self._queue.put((None, None))
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            msg, ch = self._queue.get()  # 无 timeout，纯阻塞
            if msg and ch:
                self.handle_message(msg, ch)

    def _on_channel_inbound(self, msg: InboundMessage, ch: Channel) -> None:
        """Callback invoked by channels when they receive a new InboundMessage."""
        self._queue.put((msg, ch))

    def dispatch(self, ch: Channel) -> None:
        """
        Poll one channel once: receive a message (if any) and handle it.
        Used by the background loop; CLI can call handle_message() directly.
        """
        try:
            msg = ch.receive()
            logger.info(f"========= {ch.name} received message: {msg}")
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

            # Adapt channel's send_typing(chat_id) to run_agent's on_typing(agent_id, is_typing)
            on_typing_cb = None
            if getattr(ch, "send_typing", None):
                def on_typing_cb(_agent_id: str, is_typing: bool, peer_id: str = msg.peer_id, channel: Channel = ch):
                    if is_typing:
                        try:
                            channel.send_typing(peer_id)
                        except Exception:
                            # Typing indicator failures should not break message handling
                            logger.debug("send_typing failed for %s", peer_id)

            reply = self._run_async(
                run_agent(
                    self._mgr,
                    agent_id,
                    session_key,
                    msg.text,
                    on_typing=on_typing_cb,
                    channel=msg.channel,
                )
            )
            ch.send(msg.peer_id, "[OwlBot]: " + (reply or ""))
        except Exception as e:
            logger.exception("MessageDispatcher dispatch %s: %s", msg.peer_id, e)
            try:
                ch.send(msg.peer_id, f"Error: {e}")
            except Exception:
                pass
