from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from abc import ABC, abstractmethod
from common.colors import CYAN, GREEN, RESET, BOLD
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

@dataclass
class InboundMessage:
    """All channels normalize into this. The agent loop only sees InboundMessage."""
    text: str
    sender_id: str
    channel: str = ""
    account_id: str = ""
    peer_id: str = ""
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

@dataclass
class ChannelConfig:
    """Per-bot configuration. One channel type can run multiple bots."""
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    return f"agent:main:direct:{channel}:{peer_id}"


## Channel manager -> channel instance from ChannelConfig
class Channel(ABC):
    name: str = "unknown"
    channel_config: ChannelConfig = field(default_factory=ChannelConfig)

    def __init__(self) -> None:
        # Optional callback set by MessageCenter; channels call this when they
        # have an InboundMessage ready (push-style).
        self._inbound_cb: Optional[Callable[[InboundMessage, "Channel"], None]] = None
        self._connected_cb: Optional[Callable[["Channel"], None]] = None

    def set_inbound_callback(self, cb: Callable[[InboundMessage, "Channel"], None]) -> None:
        """Register a callback that will be invoked when this channel receives a message."""
        self._inbound_cb = cb

    def _emit_inbound(self, msg: InboundMessage) -> None:
        """Helper for channel implementations to push a new inbound message."""
        if self._inbound_cb:
            self._inbound_cb(msg, self)

    ## Connected callback
    def set_connected_callback(self, cb: Callable[["Channel"], None]) -> None:
        """Register a callback that will be invoked when this channel is connected."""
        self._connected_cb = cb

    def _emit_connected(self, ch: Channel) -> None:
        """Helper for channel implementations to push a new connected event."""
        if self._connected_cb:
            self._connected_cb(ch)

    @abstractmethod
    def receive(self) -> Optional[InboundMessage]: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...

    def close(self) -> None:
        pass

# ---------------------------------------------------------------------------
# CLIChannel
# ---------------------------------------------------------------------------




# class ChannelManager:
#     def __init__(self) -> None:
#         self.channels: dict[str, Channel] = {}
#         self.accounts: list[ChannelConfig] = []

#     def register(self, channel: Channel) -> None:
#         self.channels[channel.name] = channel
#         print_channel(f"  [+] Channel registered: {channel.name}")

#     def list_channels(self) -> list[str]:
#         return list(self.channels.keys())

#     def get(self, name: str) -> Channel | None:
#         return self.channels.get(name)

#     def close_all(self) -> None:
#         for ch in self.channels.values():
#             ch.close()
