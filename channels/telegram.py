from __future__ import annotations

from typing import Any
from common.colors import RED, RESET
from channels.types_ import Channel, ChannelConfig, InboundMessage
import time
import httpx
from pathlib import Path

from common.paths import WORKSPACE_DIR, STATE_DIR

STATE_DIR.mkdir(parents=True, exist_ok=True)

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

def save_offset(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(offset))

def load_offset(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return 0

class TelegramChannel(Channel):
    name = "telegram"
    MAX_MSG_LEN = 4096

    def __init__(self, account: ChannelConfig) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("TelegramChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.base_url = f"https://api.telegram.org/bot{account.token}"
        self._http = httpx.Client(timeout=35.0)
        raw = account.config.get("allowed_chats", "")
        self.allowed_chats = {c.strip() for c in raw.split(",") if c.strip()} if raw else set()

        self._offset_path = STATE_DIR / "telegram" / f"offset-{self.account_id}.txt"
        self._offset = load_offset(self._offset_path)

        # Simple dedup: set of seen update IDs, cleared periodically
        self._seen: set[int] = set()

        # Media group buffer: group_id -> {ts, entries}
        self._media_buf: dict[str, dict] = {}

        # Text coalesce buffer: (peer, sender) -> {text, msg, ts}
        self._text_buf: dict[tuple[str, str], dict] = {}

    def _api(self, method: str, **params: Any) -> dict:
        filtered = {k: v for k, v in params.items() if v is not None}
        try:
            resp = self._http.post(f"{self.base_url}/{method}", json=filtered)
            data = resp.json()
            if not data.get("ok"):
                print(f"  {RED}[telegram] {method}: {data.get('description', '?')}{RESET}")
                return {}
            return data.get("result", {})
        except Exception as exc:
            print(f"  {RED}[telegram] {method}: {exc}{RESET}")
            return {}

    def send_typing(self, chat_id: str) -> None:
        self._api("sendChatAction", chat_id=chat_id, action="typing")

    # -- Polling --

    def poll(self) -> list[InboundMessage]:
        result = self._api("getUpdates", offset=self._offset, timeout=30,
                           allowed_updates=["message"])
        if not result or not isinstance(result, list):
            return self._flush_all()

        for update in result:
            uid = update.get("update_id", 0)

            # Advance offset so Telegram won't re-send these updates
            if uid >= self._offset:
                self._offset = uid + 1
                save_offset(self._offset_path, self._offset)

            # Simple dedup via set; clear at 5000 to bound memory
            if uid in self._seen:
                continue
            self._seen.add(uid)
            if len(self._seen) > 5000:
                self._seen.clear()

            msg = update.get("message")
            if not msg:
                continue

            # Media groups get buffered separately (multiple updates = one album)
            if msg.get("media_group_id"):
                mgid = msg["media_group_id"]
                if mgid not in self._media_buf:
                    self._media_buf[mgid] = {"ts": time.monotonic(), "entries": []}
                self._media_buf[mgid]["entries"].append((msg, update))
                continue

            inbound = self._parse(msg, update)
            if not inbound:
                continue
            if self.allowed_chats and inbound.peer_id not in self.allowed_chats:
                continue

            # Buffer text for coalescing (Telegram splits long pastes)
            key = (inbound.peer_id, inbound.sender_id)
            now = time.monotonic()
            if key in self._text_buf:
                self._text_buf[key]["text"] += "\n" + inbound.text
                self._text_buf[key]["ts"] = now
            else:
                self._text_buf[key] = {"text": inbound.text, "msg": inbound, "ts": now}

        return self._flush_all()

    # -- Flush buffered messages --

    def _flush_all(self) -> list[InboundMessage]:
        ready: list[InboundMessage] = []

        # Flush media groups after 500ms silence
        now = time.monotonic()
        expired_mg = [k for k, g in self._media_buf.items() if (now - g["ts"]) >= 0.5]
        for mgid in expired_mg:
            entries = self._media_buf.pop(mgid)["entries"]
            captions, media_items = [], []
            for m, _ in entries:
                if m.get("caption"):
                    captions.append(m["caption"])
                for mt in ("photo", "video", "document", "audio"):
                    if mt in m:
                        raw_m = m[mt]
                        if isinstance(raw_m, list) and raw_m:
                            fid = raw_m[-1]["file_id"]
                        elif isinstance(raw_m, dict):
                            fid = raw_m.get("file_id", "")
                        else:
                            fid = ""
                        media_items.append({"type": mt, "file_id": fid})
            inbound = self._parse(entries[0][0], entries[0][1])
            if inbound:
                inbound.text = "\n".join(captions) if captions else "[media group]"
                inbound.media = media_items
                if not self.allowed_chats or inbound.peer_id in self.allowed_chats:
                    ready.append(inbound)

        # Flush text buffer after 1s silence
        expired_txt = [k for k, b in self._text_buf.items() if (now - b["ts"]) >= 1.0]
        for key in expired_txt:
            buf = self._text_buf.pop(key)
            buf["msg"].text = buf["text"]
            ready.append(buf["msg"])

        return ready

    # -- Message parsing --

    def _parse(self, msg: dict, raw_update: dict) -> InboundMessage | None:
        chat = msg.get("chat", {})
        chat_type = chat.get("type", "")
        chat_id = str(chat.get("id", ""))
        user_id = str(msg.get("from", {}).get("id", ""))
        text = msg.get("text", "") or msg.get("caption", "")
        if not text:
            return None

        thread_id = msg.get("message_thread_id")
        is_forum = chat.get("is_forum", False)
        is_group = chat_type in ("group", "supergroup")

        if chat_type == "private":
            peer_id = user_id
        elif is_group and is_forum and thread_id is not None:
            peer_id = f"{chat_id}:topic:{thread_id}"
        else:
            peer_id = chat_id

        return InboundMessage(
            text=text, sender_id=user_id, channel="telegram",
            account_id=self.account_id, peer_id=peer_id,
            is_group=is_group, raw=raw_update,
        )

    def receive(self) -> InboundMessage | None:
        msgs = self.poll()
        return msgs[0] if msgs else None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        chat_id, thread_id = to, None
        if ":topic:" in to:
            parts = to.split(":topic:")
            chat_id, thread_id = parts[0], int(parts[1]) if len(parts) > 1 else None
        ok = True
        for chunk in self._chunk(text):
            if not self._api("sendMessage", chat_id=chat_id, text=chunk,
                             message_thread_id=thread_id):
                ok = False
        return ok

    def _chunk(self, text: str) -> list[str]:
        if len(text) <= self.MAX_MSG_LEN:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.MAX_MSG_LEN:
                chunks.append(text); break
            cut = text.rfind("\n", 0, self.MAX_MSG_LEN)
            if cut <= 0:
                cut = self.MAX_MSG_LEN
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def close(self) -> None:
        self._http.close()