import json
from neonize.client import NewClient
from neonize.events import ConnectedEv, MessageEv, ReceiptEv # 移除了不存在的 QRCodeEv
from neonize.utils import log
from neonize.proto.Neonize_pb2 import ChatPresence
from neonize.utils import ChatPresence, ChatPresenceMedia
from neonize.utils.jid import build_jid

def dump_message_structure(obj, max_depth=5, _depth=0):
    """将 message 等对象转成可打印的嵌套结构（dict/list/基本类型）。"""
    if _depth > max_depth:
        return "<max_depth>"
    try:
        if hasattr(obj, "DESCRIPTOR"):  # protobuf
            try:
                from google.protobuf.json_format import MessageToDict
                return MessageToDict(obj, preserving_proto_field_name=True)
            except ImportError:
                pass
        if hasattr(obj, "__dict__") and type(obj).__name__ not in ("str", "int", "float", "bool", "bytes"):
            out = {}
            for k in dir(obj):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(obj, k)
                    if callable(v):
                        continue
                    out[k] = dump_message_structure(v, max_depth, _depth + 1)
                except Exception:
                    out[k] = "<error>"
            return out
        if isinstance(obj, (list, tuple)):
            return [dump_message_structure(x, max_depth, _depth + 1) for x in obj[:20]]
        if isinstance(obj, dict):
            return {k: dump_message_structure(v, max_depth, _depth + 1) for k, v in list(obj.items())[:30]}
        return obj
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


# 初始化客户端
client = NewClient("db.sqlite3")

@client.event(ConnectedEv)
def on_connected(client: NewClient, _: ConnectedEv):
    log.info("✅ Successfully connected to WhatsApp！")

@client.event(MessageEv)
def on_message(client: NewClient, message: MessageEv):
    try:
        log.info("message 完整结构:\n" + json.dumps(dump_message_structure(message), indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        log.info("message 结构(fallback): " + repr(message))


    message_text = message.Message.conversation
    sender_jid = message.Info.MessageSource.Sender
    chat_jid = message.Info.MessageSource.Chat


    client.send_chat_presence(
        sender_jid,
        ChatPresence.CHAT_PRESENCE_COMPOSING,
        ChatPresenceMedia.CHAT_PRESENCE_MEDIA_TEXT,
    )

    if message_text.lower() == "ping" or message_text.lower() == "hi":
        client.send_message(chat_jid, "pong!")


client.connect()
