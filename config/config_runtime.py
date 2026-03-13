from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple, List

from common.paths import WORKSPACE_DIR
from agent.agent_ import Agent, AgentManager
from message.route_ import BindingTable, Binding
from channels.types_ import ChannelConfig

CONFIG_PATH = WORKSPACE_DIR / "runtime_config.json"

def setup_from_config() -> Tuple[AgentManager, BindingTable, List[ChannelConfig]] | None:
    """
    如果存在 runtime_config.json，则根据其中的 agents / bindings / channels
    构建 AgentManager、BindingTable 和 ChannelConfig 列表。
    """
    if not CONFIG_PATH.is_file():
        return None

    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not raw:
        return None

    mgr = AgentManager()
    bt = BindingTable()

    for a in raw.get("agents", []):
        mgr.register(Agent(
            id=a.get("id", "main"),
            name=a.get("name", "Main"),
            personality=a.get("personality", ""),
            model=a.get("model", ""),
        ))

    for b in raw.get("bindings", []):
        bt.add(Binding(
            agent_id=b.get("agent_id", "main"),
            channel=b.get("channel", "*"),
            account_id=b.get("account_id", "*"),
            peer_id=b.get("peer_id", "*"),
            priority=int(b.get("priority", 0)),
        ))

    channels_conf = raw.get("channels", [])
    auto_bridge = set(raw.get("auto_bridge", []))
    accounts: list[ChannelConfig] = []
    for ch in channels_conf:
        if not ch.get("enabled", True):
            continue
        ch_type = ch.get("type", "")
        if ch_type not in auto_bridge:
            # 只自动 bridge 在 auto_bridge 里的渠道
            continue
        acc = ChannelConfig(
            channel=ch_type,
            account_id=ch.get("account_id", ch_type + "-default"),
            token="",
            config=ch.get("config", {}) or {},
        )
        accounts.append(acc)

    return mgr, bt, accounts


def write_simple_default(path: Path | None = None) -> None:
    """
    生成一份最简单的默认配置（单 agent + whatsapp_web），供第一次使用时参考。
    不会覆盖已经存在的配置文件。
    """
    target = path or CONFIG_PATH
    if target.exists():
        return
    data: dict[str, Any] = {
        "agents": [
            {"id": "luna", "name": "Luna", "personality": "", "model": ""},
        ],
        "bindings": [
            # 全局默认：所有渠道 / 账号 / 对端 -> luna
            {"agent_id": "luna", "channel": "*", "account_id": "*", "peer_id": "*", "priority": 0},
            # WhatsApp 渠道默认 -> luna（示例，和上面等价，只是更具体）
            {"agent_id": "luna", "channel": "whatsapp_web", "account_id": "*", "peer_id": "*", "priority": 0},
        ],
        "channels": [
            {
                "type": "whatsapp_web",
                "account_id": "wa-default",
                "enabled": True,
                "config": {
                    "session_path": "",
                    "allowed_chats": "",
                    "allowed_groups": "",
                    "log_messages": True,
                },
            },
        ],
        "auto_bridge": ["whatsapp_web"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

