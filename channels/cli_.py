from __future__ import annotations
from typing import Optional, Any
from common.colors import CYAN, GREEN, RESET, BOLD
from common.paths import PROJECT_ROOT
from dotenv import load_dotenv
import sys
from .types_ import Channel, InboundMessage, ChannelConfig

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

class CLIChannel(Channel):
    name = "cli"
    channel_config: ChannelConfig = ChannelConfig(channel="cli", account_id="cli-local")

    def __init__(self, account: ChannelConfig) -> None:
        self.channel_config = account
        super().__init__()


    def receive(self) -> Optional[InboundMessage]:
        try:
            text = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            return None

        inbound = InboundMessage(
            text=text, 
            sender_id="cli-user", 
            channel=self.channel_config.channel,
            account_id=self.channel_config.account_id,
            peer_id="cli-user",
        )

        self._emit_inbound(inbound)
        print(f"{GREEN}[CLI] -> {text}{RESET}")

        return inbound
    
    def handle_message(self, message: str) -> None:
        """Handle a message from the CLI."""
        inbound = InboundMessage(
            text=message, 
            sender_id="cli-user", 
            channel="cli",
            account_id=self.channel_config.account_id,
            peer_id="cli-user",
        )

        self._emit_inbound(inbound)

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        print(f"{GREEN}[CLI]-> {text}{RESET}")
        return True