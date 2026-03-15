from __future__ import annotations

"""
WhatsApp Web channel using neonize: QR login, session persistence (SQLite),
message events → InboundMessage. Only DMs and allowed_groups are enqueued.

Flow: client.connect() runs in a daemon thread; MessageEv handler puts
InboundMessage in _inbox; receive() returns from _inbox; send() uses
client.send_message(chat_jid, text). Requires: neonize (and libmagic on system).
"""

import re
import threading
import time
from pathlib import Path
from typing import Any, Optional
import logging
from common.colors import RED, GREEN, RESET, YELLOW
from common.paths import STATE_DIR
from channels.types_ import Channel, ChannelConfig, InboundMessage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False

STATE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from neonize.client import NewClient
    from neonize.events import ConnectedEv, MessageEv
    from neonize.utils.jid import Jid2String, build_jid
    HAS_NEONIZE = True
except ImportError as _e:
    HAS_NEONIZE = False
    _NEONIZE_ERROR = _e
    _NEONIZE_PYTHON = __import__("sys").executable

MAX_MSG_LEN = 4096


def _peer_from_jid(jid) -> str:
    """JID -> peer_id string (user@server)."""
    return Jid2String(jid) if jid else ""

def _jid_from_peer_id(peer_id: str):
    """peer_id string (user@server or digits) -> JID for send_message."""
    peer_id = (peer_id or "").strip()
    if "@" in peer_id:
        user, _, server = peer_id.partition("@")
        return build_jid(user.strip(), server.strip() or "s.whatsapp.net")
    digits = re.sub(r"\D", "", peer_id)
    return build_jid(digits, "s.whatsapp.net")


class WhatsAppWebChannel(Channel):
    """
    WhatsApp via neonize (QR + SQLite session). Processes only:
    - All DMs, or only allowed_chats if set.
    - Group messages only if allowed_groups is set and group id is in it.
    """
    name = "whatsapp_web"
    MAX_MSG_LEN = MAX_MSG_LEN

    def __init__(self, account: ChannelConfig) -> None:
        super().__init__()
        if not HAS_NEONIZE:
            raise RuntimeError(
                "WhatsAppWebChannel requires neonize. "
                f"Current Python: {_NEONIZE_PYTHON}. "
                "pip install neonize (and e.g. brew install libmagic on macOS)."
            )
        self.account_id = account.account_id
        logger.info(f"{GREEN}[whatsapp_web] Account ID: {self.account_id}{RESET}")

        session_dir = account.config.get("session_path") or ""
        if not session_dir:
            session_dir = str(STATE_DIR / "whatsapp_web" / f"session-{self.account_id}")
        self._session_path = Path(session_dir)
        self._session_path.mkdir(parents=True, exist_ok=True)
        # neonize uses a DB path (e.g. sqlite file)
        self._store_path = str(self._session_path / "store.sqlite3")

        # need config from account.config
        raw_chats = (account.config.get("allowed_chats") or "").strip()
        self.allowed_chats = {c.strip() for c in raw_chats.split(",") if c.strip()} if raw_chats else set()
        raw_groups = (account.config.get("allowed_groups") or "").strip()
        self.allowed_groups = {g.strip() for g in raw_groups.split(",") if g.strip()} if raw_groups else set()
        self._log_messages = bool(account.config.get("log_messages", False))
        # Reconnect cap: avoid infinite retry loops (0/None treated as default).
        try:
            self._max_reconnect_attempts = int(account.config.get("reconnect_max_attempts", 10))
        except Exception:
            self._max_reconnect_attempts = 10
        if self._max_reconnect_attempts <= 0:
            self._max_reconnect_attempts = 10
        self._gave_up = False

        self._client: Optional[NewClient] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._client_lock = threading.Lock()

    def _should_accept(self, chat_jid, is_group: bool) -> bool:
        peer_str = _peer_from_jid(chat_jid)
        if is_group:
            group_id = getattr(chat_jid, "User", "") or ""
            return bool(self.allowed_groups and group_id in self.allowed_groups)
        if self.allowed_chats:
            return peer_str in self.allowed_chats
        return True

    def _on_connected(self, client: NewClient, _) -> None:
        self._ready.set()
        logger.info(f"{GREEN}[whatsapp_web] Connected (session: {self._session_path}, account: {self.account_id}){RESET}")
        if not self.account_id:
            self.account_id = client.get_me().User
            logger.info(f"{GREEN}[whatsapp_web] Account ID: {self.account_id}{RESET}")
            self.channel_config.account_id = self.account_id
        
        self._emit_connected(self)

    def _on_message(self, client: NewClient, event) -> None:
        try:
            # 按 README 示例: event.Message / event.Info.MessageSource
            msg_obj = event.Message
            text = getattr(msg_obj, "conversation", None) or ""
            if not text and getattr(msg_obj, "extendedTextMessage", None):
                text = getattr(msg_obj.extendedTextMessage, "text", None) or ""
            text = (text or "").strip()
            if not text:
                return

            info = event.Info
            src = info.MessageSource
            chat_jid = src.Chat
            sender_jid = src.Sender
            is_group = bool(getattr(src, "IsGroup", False))
            if not self._should_accept(chat_jid, is_group):
                return

            peer_id = _peer_from_jid(chat_jid)
            sender_id = _peer_from_jid(sender_jid)
            inbound = InboundMessage(
                text=text,
                sender_id=sender_id,
                channel=self.name,
                account_id=self.account_id,
                peer_id=peer_id,
                is_group=is_group,
                raw={},
            )
            if self._log_messages:
                logger.info(f"{GREEN}[whatsapp_web] from {peer_id}: {text[:50]}{'...' if len(text) > 50 else ''}{RESET}")

            # Push-style: emit inbound message via MessageCenter callback (if set)
            self._emit_inbound(inbound)
        except Exception as e:
            logger.info(f"{RED}[whatsapp_web] on_message: {e}{RESET}")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._gave_up:
            logger.info(f"{YELLOW}[whatsapp_web] Reconnect attempts exhausted; not retrying.{RESET}")
            return
        self._stop.clear()
        self._ready.clear()

        def run_connect_forever():
            backoff = 1.0
            attempts = 0
            while not self._stop.is_set():
                if attempts >= self._max_reconnect_attempts:
                    self._gave_up = True
                    logger.info(
                        f"  {RED}[whatsapp_web] connect: gave up after {attempts} attempt(s).{RESET}"
                    )
                    break
                try:
                    client = NewClient(self._store_path)
                    # QR 由 neonize 默认在命令行中输出，无需保存图片
                    client.event(ConnectedEv)(self._on_connected)
                    client.event(MessageEv)(self._on_message)
                    with self._client_lock:
                        self._client = client
                    attempts += 1
                    client.connect()
                    # connect() 返回（断线/退出）也视为需要重连
                    if not self._stop.is_set():
                        logger.info(f"{YELLOW}[whatsapp_web] disconnected; retrying...{RESET}")
                except Exception as e:
                    if not self._stop.is_set():
                        logger.info(f"{RED}[whatsapp_web] connect error: {e}{RESET}")
                finally:
                    with self._client_lock:
                        self._client = None
                    self._ready.clear()
                if self._stop.is_set():
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)

        self._thread = threading.Thread(target=run_connect_forever, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=90)
        if not self._ready.is_set():
            logger.info(f"{YELLOW}[whatsapp_web] Waiting for QR scan / connection.{RESET}")

    def ensure_started(self) -> None:
        if self._gave_up:
            return
        if self._thread is None or not self._thread.is_alive():
            self.start()
        self._ready.wait(timeout=0.5)

    def receive(self) -> Optional[InboundMessage]:
        """
        Backward-compatible pull API. For WhatsAppWebChannel we primarily rely on
        push-style callbacks via _emit_inbound; receive() is kept as a no-op
        that ensures the client is started but does not block.
        """
        self.ensure_started()
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        to = to.split(":topic:")[0] if ":topic:" in to else to
        if not self._client:
            self.ensure_started()
        with self._client_lock:
            client = self._client
        if not client:
            return False
        try:
            chat_jid = _jid_from_peer_id(to)

            logger.info(f"{GREEN}[whatsapp_web]-> {text}{RESET}")

            for chunk in self._chunk(text):
                client.send_message(chat_jid, chunk)
            return True
        except Exception as e:
            logger.info(f"{RED}[whatsapp_web]: {e}{RESET}")
            return False

    def _chunk(self, text: str) -> list[str]:
        if len(text) <= self.MAX_MSG_LEN:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.MAX_MSG_LEN:
                chunks.append(text)
                break
            cut = text.rfind("\n", 0, self.MAX_MSG_LEN)
            if cut <= 0:
                cut = self.MAX_MSG_LEN
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def send_typing(self, chat_id: str) -> None:
        with self._client_lock:
            client = self._client
        if not client:
            return
        try:
            from neonize.utils import ChatPresence, ChatPresenceMedia
            jid = _jid_from_peer_id(chat_id)
            client.send_chat_presence(
                jid,
                ChatPresence.CHAT_PRESENCE_COMPOSING,
                ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
            )
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        self._gave_up = False
        with self._client_lock:
            client = self._client
        if client and hasattr(client, "disconnect"):
            try:
                client.disconnect()
            except Exception:
                pass
        with self._client_lock:
            self._client = None
