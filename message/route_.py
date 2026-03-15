from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any
from common.colors import CYAN, GREEN, YELLOW, DIM, RESET, BOLD, MAGENTA, RED, BLUE
from common.paths import PROJECT_ROOT, WORKSPACE_DIR, AGENTS_DIR
from dotenv import load_dotenv
import sys
from agent.agent_ import Agent, normalize_agent_id, AgentManager, DEFAULT_AGENT_ID

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

@dataclass
class Binding:
    """
    不同的peer_id 可以绑定到同一个agent_id, 也可以绑定到不同的agent_id。
    同一个agent中的 context 会根据 session_key 来区分。 session_key 的格式为 {agent_id}-{channel}-{account_id}-{peer_id}。
    保证每个 session_key 对应到 peer_id 或者 channel，或者 channel + account_id。

    绑定规则（三层：channel / account / peer）：
    - channel:    渠道名，如 "whatsapp_web" / "telegram"，"*" 表示任意
    - account_id: 账号 ID，如 "wa-default"，"*" 表示任意
    - peer_id:    对端 ID（人/群的 jid / chat_id），"*" 表示任意

    优先级：
    - 精确到 peer      -> 最具体
    - 精确到 account   -> 次之
    - 只指定 channel   -> 再次
    - 全 "*"           -> 最弱（兜底）
    同一具体程度下，priority 越大越优先。
    """
    agent_id: str
    channel: str = "*"
    account_id: str = "*"
    peer_id: str = "*"
    priority: int = 0

    def display(self) -> str:
        return (
            f"[binding] ch={self.channel} acc={self.account_id} peer={self.peer_id} "
            f"-> agent:{self.agent_id} (pri={self.priority})"
        )

class BindingTable:
    def __init__(self) -> None:
        self._bindings: list[Binding] = []

    def add(self, binding: Binding) -> None:
        self._bindings.append(binding)

    def remove(
        self,
        agent_id: str,
        channel: str = "*",
        account_id: str = "*",
        peer_id: str = "*",
    ) -> bool:
        before = len(self._bindings)
        self._bindings = [
            b for b in self._bindings
            if not (
                b.agent_id == agent_id
                and b.channel == channel
                and b.account_id == account_id
                and b.peer_id == peer_id
            )
        ]
        return len(self._bindings) < before

    def list_all(self) -> list[Binding]:
        return list(self._bindings)

    def resolve(
        self,
        channel: str = "",
        account_id: str = "",
        guild_id: str = "",
        peer_id: str = "",
    ) -> tuple[str | None, Binding | None]:
        """
        根据 (channel, account_id, peer_id) 选出“最具体”的一条绑定：
        - 精确匹配字段数量更多的优先（3 > 2 > 1 > 0）
        - 同一具体程度下，priority 大的优先
        """
        ch = (channel or "").strip().lower()
        acc = (account_id or "").strip().lower()
        pid = (peer_id or "").strip().lower()

        best_key: tuple[int, int, int] | None = None  # (specificity, priority, -index)
        best_binding: Binding | None = None

        for idx, b in enumerate(self._bindings):
            if b.channel not in ("*", ch):
                continue
            if b.account_id not in ("*", acc):
                continue
            if b.peer_id not in ("*", pid):
                continue

            spec = 0
            if b.channel != "*":
                spec += 1
            if b.account_id != "*":
                spec += 1
            if b.peer_id != "*":
                spec += 1

            key = (spec, b.priority, -idx)
            if best_key is None or key > best_key:
                best_key = key
                best_binding = b

        if not best_binding:
            return None, None
        return best_binding.agent_id, best_binding

def build_session_key(
    agent_id: str,
    channel: str = "",
    account_id: str = "",
    peer_id: str = "",
) -> str:
    """
    Session key is decouple with agent,
    one agent can have multiple session keys.
    Like: WhatsApp agent hanle all the message form whatsapp_web account. maybe from Master peer also be from other friends and groups.
    
    {aid}-{channel}-{account}-{peer}.


    """
    aid = normalize_agent_id(agent_id)
    ch = (channel or "").strip().lower()
    acc = (account_id or "").strip().lower()
    pid = (peer_id or "").strip().lower()
    if not ch or not acc:
        raise ValueError(
            f"build_session_key: channel, account_id and peer_id are required, channel='{ch}' account_id='{acc}' peer_id='{pid}'"
        )
    if pid:
        return f"{aid}-{ch}-{acc}-{pid}"
    else:
        return f"{aid}-{ch}-{acc}-main" # main mean all the messages from different peers in this same channel and account stored in one session.


def resolve_route(bindings: BindingTable, mgr: AgentManager,
                  channel: str, peer_id: str,
                  account_id: str = "", guild_id: str = "") -> tuple[str, str]:
    agent_id, matched = bindings.resolve(
        channel=channel, account_id=account_id,
        guild_id=guild_id, peer_id=peer_id,
    )
    if not agent_id:
        agent_id = DEFAULT_AGENT_ID
        print(f"  {DIM}[route] No binding matched, default: {agent_id}{RESET}")
    elif matched:
        print(f"  {DIM}[route] Matched: {matched.display()}{RESET}")
    # agent = mgr.get_agent(agent_id)
    
    sk = build_session_key(agent_id, channel=channel, account_id=account_id, peer_id=peer_id)
    return agent_id, sk


def setup_demo() -> tuple[AgentManager, BindingTable]:
    mgr = AgentManager()
    mgr.register(Agent(
        id="luna", name="Luna"
    ))

    mgr.register(Agent(
        id="sage", name="Sage"
    ))
    bt = BindingTable()
    # 全局默认：所有渠道 / 账号 / 对端 -> luna
    bt.add(Binding(agent_id="luna", channel="*", account_id="*", peer_id="*", priority=0))
    # Telegram 渠道默认 -> sage
    bt.add(Binding(agent_id="sage", channel="telegram", account_id="*", peer_id="*", priority=0))
    # WhatsApp 渠道默认 -> luna
    bt.add(Binding(agent_id="luna", channel="whatsapp_web", account_id="*", peer_id="*", priority=0))
    return mgr, bt
