from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod
from common.colors import CYAN, GREEN, YELLOW, DIM, RESET, BOLD, RED, BLUE
from common.logs import print_assistant
from common.logs import print_channel
  
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
class ChannelAccount:
    """Per-bot configuration. One channel type can run multiple bots."""
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    return f"agent:main:direct:{channel}:{peer_id}"


class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...

    def close(self) -> None:
        pass

# ---------------------------------------------------------------------------
# CLIChannel
# ---------------------------------------------------------------------------

class CLIChannel(Channel):
    name = "cli"

    def __init__(self) -> None:
        self.account_id = "cli-local"

    def receive(self) -> InboundMessage | None:
        try:
            text = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            return None
        return InboundMessage(
            text=text, sender_id="cli-user", channel="cli",
            account_id=self.account_id, peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        print_assistant(text)
        return True


# class ChannelManager:
#     def __init__(self) -> None:
#         self.channels: dict[str, Channel] = {}
#         self.accounts: list[ChannelAccount] = []

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
