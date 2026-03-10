"""简单 CLI 向导：生成 runtime_config.json。

运行：
  python wizard_cli.py
"""

import json

from common.paths import WORKSPACE_DIR, STATE_DIR

from dispatch.config_runtime import CONFIG_PATH


def ask(prompt: str, default: str = "") -> str:
  s = input(f"{prompt} [{default}]: ").strip()
  return s or default


def main() -> None:
  print("=== Owlbot 配置向导 (CLI) ===")

  # Agent
  agent_id = ask("Agent ID", "luna")
  agent_name = ask("Agent 名称", "Luna")
  personality = ask("Agent 人设/性格描述", "")
  dm_scope = ask("会话隔离级别 dm_scope (main/per-peer/per-channel-peer/per-account-channel-peer)", "per-peer")

  # WhatsApp Web channel
  enable_wa = ask("是否启用 WhatsApp Web 渠道? (y/n)", "y").lower().startswith("y")
  channels = []
  auto_bridge = []
  if enable_wa:
    account_id = ask("WhatsApp Web account_id", "wa-default")
    default_session = str(STATE_DIR / "whatsapp_web" / f"session-{account_id}")
    session_path = ask("会话目录 session_path", default_session)
    allowed_chats = ask("允许的私聊列表(逗号分隔, 留空表示全部)", "")
    allowed_groups = ask("允许的群ID列表(逗号分隔, 留空表示不处理任何群)", "")
    log_messages = ask("是否在终端打印收到的消息? (y/n)", "y").lower().startswith("y")
    channels.append({
      "type": "whatsapp_web",
      "account_id": account_id,
      "enabled": True,
      "config": {
        "session_path": session_path,
        "allowed_chats": allowed_chats,
        "allowed_groups": allowed_groups,
        "log_messages": log_messages,
      },
    })
    auto_bridge.append("whatsapp_web")

  data = {
    "agents": [
      {
        "id": agent_id,
        "name": agent_name,
        "personality": personality,
        "model": "",
        "dm_scope": dm_scope,
      },
    ],
    "bindings": [
      {"agent_id": agent_id, "tier": 5, "match_key": "default", "match_value": "*", "priority": 0},
      # 所有 whatsapp_web 消息默认为这个 agent
      {"agent_id": agent_id, "tier": 4, "match_key": "channel", "match_value": "whatsapp_web", "priority": 0},
    ],
    "channels": channels,
    "auto_bridge": auto_bridge,
  }

  CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
  CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

  print(f"\n配置已写入: {CONFIG_PATH}")
  print("下次直接运行 main.py 即会按此配置启动 (自动 bridge WhatsApp Web，如已启用)。")


if __name__ == "__main__":
  main()

