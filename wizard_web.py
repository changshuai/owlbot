"""简单 Web 向导：通过浏览器填写配置表单，生成 runtime_config.json。

运行：
  python wizard_web.py
然后打开浏览器访问 http://127.0.0.1:8766
"""

from __future__ import annotations

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

from common.paths import STATE_DIR
from dispatch.config_runtime import CONFIG_PATH


# 用普通字符串存 HTML，发送时再按 UTF-8 编码，避免 bytes 字面量中出现中文导致语法错误。
HTML_FORM = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Owlbot 配置向导</title>
</head>
<body>
  <h1>Owlbot 配置向导 (Web)</h1>
  <form method="post" action="/save">
    <h2>Agent</h2>
    <label>Agent ID: <input name="agent_id" value="luna"></label><br>
    <label>Agent 名称: <input name="agent_name" value="Luna"></label><br>
    <label>Personality: <input name="personality" style="width:400px;"></label><br>
    <label>dm_scope:
      <select name="dm_scope">
        <option value="per-peer" selected>per-peer</option>
        <option value="main">main</option>
        <option value="per-channel-peer">per-channel-peer</option>
        <option value="per-account-channel-peer">per-account-channel-peer</option>
      </select>
    </label>

    <h2>WhatsApp Web 渠道</h2>
    <label><input type="checkbox" name="enable_wa" checked> 启用 WhatsApp Web</label><br>
    <label>account_id: <input name="wa_account_id" value="wa-default"></label><br>
    <label>session_path (留空则用默认): <input name="wa_session_path" style="width:400px;"></label><br>
    <label>allowed_chats (逗号分隔): <input name="wa_allowed_chats" style="width:400px;"></label><br>
    <label>allowed_groups (逗号分隔): <input name="wa_allowed_groups" style="width:400px;"></label><br>
    <label><input type="checkbox" name="wa_log_messages" checked> 终端打印收到的消息</label><br>

    <p><button type="submit">保存配置</button></p>

    {{BINDINGS}}
  </form>
</body>
</html>
"""


class WizardHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path != "/":
      self.send_response(404)
      self.end_headers()
      return
    self.send_response(200)
    self.send_header("Content-Type", "text/html; charset=utf-8")

    # 如果已有配置文件，展示当前绑定列表（只读，尽量用人话描述）
    bindings_html = ""
    try:
      if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        bds = raw.get("bindings", [])
        if bds:
          rows: list[str] = []
          for b in bds:
            agent = b.get("agent_id", "?")
            tier = int(b.get("tier", 5))
            key = b.get("match_key", "default")
            val = b.get("match_value", "*")

            # 用更人性化的语句描述绑定含义
            if key == "default" and tier == 5:
              text = f"所有没有更具体规则匹配到的消息，都会交给 Agent \"{agent}\" 处理。"
            elif key == "channel" and tier == 4:
              text = f"来自渠道 \"{val}\" 的所有消息，默认交给 Agent \"{agent}\" 处理。"
            elif key == "peer_id" and tier == 1:
              text = f"来自对话/用户 \"{val}\" 的消息，优先交给 Agent \"{agent}\" 处理。"
            else:
              text = (
                f"匹配规则 [tier {tier}] {key}={val} → Agent \"{agent}\" "
                "(这是较底层的高级用法，一般情况可以忽略这一行)。"
              )
            rows.append(f"<li>{text}</li>")

          bindings_html = (
            "<h2>当前绑定 (只读)</h2>"
            "<p>下面是当前的路由规则，用来决定「哪一条消息交给哪一个 Agent」。</p>"
            "<ul>" + "".join(rows) + "</ul>"
          )
    except Exception:
      pass

    body = HTML_FORM.replace("{{BINDINGS}}", bindings_html).encode("utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_POST(self):
    if self.path != "/save":
      self.send_response(404)
      self.end_headers()
      return
    length = int(self.headers.get("Content-Length", "0") or "0")
    body = self.rfile.read(length).decode("utf-8")
    form = parse_qs(body)

    def gv(name: str, default: str = "") -> str:
      return (form.get(name, [default])[0] or "").strip()

    def gb(name: str) -> bool:
      return name in form

    agent_id = gv("agent_id", "luna")
    agent_name = gv("agent_name", "Luna")
    personality = gv("personality", "")
    dm_scope = gv("dm_scope", "per-peer")

    channels = []
    auto_bridge = []
    if gb("enable_wa"):
      account_id = gv("wa_account_id", "wa-default")
      default_session = str(STATE_DIR / "whatsapp_web" / f"session-{account_id}")
      session_path = gv("wa_session_path", default_session)
      allowed_chats = gv("wa_allowed_chats", "")
      allowed_groups = gv("wa_allowed_groups", "")
      log_messages = gb("wa_log_messages")
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
        {"agent_id": agent_id, "tier": 4, "match_key": "channel", "match_value": "whatsapp_web", "priority": 0},
      ],
      "channels": channels,
      "auto_bridge": auto_bridge,
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    resp = f"配置已保存到 {CONFIG_PATH}. 下次运行 main.py 会按此配置启动。".encode("utf-8")
    self.send_response(200)
    self.send_header("Content-Type", "text/plain; charset=utf-8")
    self.send_header("Content-Length", str(len(resp)))
    self.end_headers()
    self.wfile.write(resp)


def main() -> None:
  addr = ("127.0.0.1", 8766)
  srv = HTTPServer(addr, WizardHandler)
  print(f"Web 向导已启动: http://{addr[0]}:{addr[1]}")
  print("填写完表单并保存后，可 Ctrl+C 退出本服务。")
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    print("\n停止 Web 向导。")


if __name__ == "__main__":
  main()

