"""
WhatsApp channel via Meta WhatsApp Cloud API.
Send: REST API. Receive: webhook (small HTTP server) + queue.
Config: token, phone_number_id (required); verify_token, webhook_port, allowed_chats (optional).
"""
from __future__ import annotations

from typing import Any
import json
import re
import threading
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from common.colors import RED, RESET
from channels.types_ import Channel, ChannelConfig, InboundMessage

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# WhatsApp Cloud API
API_VERSION = "v18.0"
GRAPH_BASE = f"https://graph.facebook.com/{API_VERSION}"
MAX_MSG_LEN = 4096  # WhatsApp supports long messages; chunk for consistency with Telegram


def _normalize_phone(to: str) -> str:
    """E.164 without + and spaces."""
    return re.sub(r"\D", "", to.strip())


class _WebhookHandler(BaseHTTPRequestHandler):
    """Handles GET (verification) and POST (incoming messages). Queue is set by WhatsAppChannel."""
    queue: queue.Queue = None
    verify_token: str = ""
    channel_ref: "WhatsAppChannel | None" = None

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress default log

    def do_GET(self) -> None:
        q = urlparse(self.path).query
        params = parse_qs(q)
        mode = (params.get("hub.mode") or [""])[0]
        token = (params.get("hub.verify_token") or [""])[0]
        challenge = (params.get("hub.challenge") or [""])[0]
        if mode == "subscribe" and token == self.verify_token and challenge:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(challenge.encode("utf-8"))
        else:
            self.send_response(403)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path.strip("/") != "webhook" and not self.path.endswith("webhook"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")
        if not body or not _WebhookHandler.queue or not _WebhookHandler.channel_ref:
            return
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            return
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") != "messages":
                    continue
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    inbound = _WebhookHandler.channel_ref._parse(msg, value)
                    if inbound:
                        if _WebhookHandler.channel_ref.allowed_chats and inbound.peer_id not in _WebhookHandler.channel_ref.allowed_chats:
                            continue
                        _WebhookHandler.queue.put(inbound)


class WhatsAppChannel(Channel):
    name = "whatsapp"
    MAX_MSG_LEN = MAX_MSG_LEN

    def __init__(self, account: ChannelConfig) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("WhatsAppChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self._token = account.token
        self._phone_number_id = (account.config.get("phone_number_id") or "").strip()
        if not self._phone_number_id:
            raise ValueError("WhatsAppChannel config must include phone_number_id")
        self._verify_token = (account.config.get("verify_token") or "owlbot-verify").strip()
        self._webhook_port = int(account.config.get("webhook_port", 8766))
        raw = account.config.get("allowed_chats", "")
        self.allowed_chats = {c.strip() for c in raw.split(",") if c.strip()} if raw else set()

        self._http = httpx.Client(timeout=35.0)
        self._inbox: queue.Queue = queue.Queue()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

        # Bind webhook handler to this channel and start server
        _WebhookHandler.queue = self._inbox
        _WebhookHandler.verify_token = self._verify_token
        _WebhookHandler.channel_ref = self
        try:
            self._server = HTTPServer(("", self._webhook_port), _WebhookHandler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        except OSError as e:
            print(f"  {RED}[whatsapp] Webhook server failed to bind port {self._webhook_port}: {e}{RESET}")

    def _api_send(self, to: str, text: str) -> bool:
        to = _normalize_phone(to)
        url = f"{GRAPH_BASE}/{self._phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"body": text},
        }
        try:
            resp = self._http.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            )
            data = resp.json()
            if "error" in data:
                print(f"  {RED}[whatsapp] send: {data['error'].get('message', data)}{RESET}")
                return False
            return True
        except Exception as exc:
            print(f"  {RED}[whatsapp] send: {exc}{RESET}")
            return False

    def _parse(self, msg: dict, value: dict) -> InboundMessage | None:
        from_id = str(msg.get("from", ""))
        msg_type = msg.get("type", "")
        if msg_type != "text":
            return InboundMessage(
                text="[non-text message]",
                sender_id=from_id,
                channel="whatsapp",
                account_id=self.account_id,
                peer_id=from_id,
                is_group=False,
                raw={"message": msg, "value": value},
            )
        body = (msg.get("text") or {}).get("body", "")
        if not body:
            return None
        return InboundMessage(
            text=body,
            sender_id=from_id,
            channel="whatsapp",
            account_id=self.account_id,
            peer_id=from_id,
            is_group=False,
            raw={"message": msg, "value": value},
        )

    def send_typing(self, chat_id: str) -> None:
        # Cloud API has no typing indicator
        pass

    def receive(self) -> InboundMessage | None:
        try:
            return self._inbox.get(timeout=1.0)
        except queue.Empty:
            return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        to = to.split(":topic:")[0] if ":topic:" in to else to
        to = _normalize_phone(to)
        ok = True
        for chunk in self._chunk(text):
            if not self._api_send(to, chunk):
                ok = False
        return ok

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

    def close(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        self._http.close()
        _WebhookHandler.channel_ref = None
        _WebhookHandler.queue = None
