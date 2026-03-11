import asyncio, sys, threading
from message.route_ import BindingTable, AgentManager, DEFAULT_AGENT_ID, build_session_key, normalize_agent_id
from common.colors import DIM, RESET, BOLD, CYAN, GREEN, YELLOW, MAGENTA, BLUE, RED
from message.route_ import setup_demo
from message.agent_loop import run_agent
from LLMs import get_env_api_key
from message.route_ import resolve_route
from message.agent_loop import MODEL_PROVIDER, MODEL_ID
from message.gateway import GatewayServer
from message.config_runtime import setup_from_config as setup_from_runtime_config, write_simple_default
from channels.channel_manager import ChannelManager
from channels.types_ import ChannelAccount, InboundMessage, CLIChannel
from message.message_center import MessageCenter

_event_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

## Get Event Loop
def get_event_loop() -> asyncio.AbstractEventLoop:
    global _event_loop, _loop_thread
    if _event_loop is not None and _event_loop.is_running():
        return _event_loop
    _event_loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_forever()
    _loop_thread = threading.Thread(target=_run, daemon=True)
    _loop_thread.start()
    return _event_loop

## Run Async
def run_async(coro):
    loop = get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def cmd_bindings(bt: BindingTable) -> None:
    all_b = bt.list_all()
    if not all_b:
        print(f"  {DIM}(no bindings){RESET}"); return
    print(f"\n{BOLD}Route Bindings ({len(all_b)}):{RESET}")
    for b in all_b:
        c = [MAGENTA, BLUE, CYAN, GREEN, DIM][min(b.tier - 1, 4)]
        print(f"  {c}{b.display()}{RESET}")
    print()

def cmd_route(bt: BindingTable, mgr: AgentManager, args: str) -> None:
    parts = args.strip().split()
    if len(parts) < 2:
        print(f"  {YELLOW}Usage: /route <channel> <peer_id> [account_id] [guild_id]{RESET}"); return
    ch, pid = parts[0], parts[1]
    acc = parts[2] if len(parts) > 2 else ""
    gid = parts[3] if len(parts) > 3 else ""
    aid, sk = resolve_route(bt, mgr, channel=ch, peer_id=pid, account_id=acc, guild_id=gid)
    a = mgr.get_agent(aid)
    print(f"\n{BOLD}Route Resolution:{RESET}")
    print(f"  {DIM}Input:   ch={ch} peer={pid} acc={acc or '-'} guild={gid or '-'}{RESET}")
    print(f"  {CYAN}Agent:   {aid} ({a.name if a else '?'}){RESET}")
    print(f"  {GREEN}Session: {sk}{RESET}\n")

def cmd_agents(mgr: AgentManager) -> None:
    agents = mgr.list_agents()
    if not agents:
        print(f"  {DIM}(no agents){RESET}"); return
    print(f"\n{BOLD}Agents ({len(agents)}):{RESET}")
    for a in agents:
        print(f"  {CYAN}{a.id}{RESET} ({a.name})  model={a.effective_model}  dm_scope={a.dm_scope}")
        if a.personality:
            print(f"    {DIM}{a.personality[:70]}{'...' if len(a.personality) > 70 else ''}{RESET}")
    print()

def cmd_sessions(mgr: AgentManager) -> None:
    s = mgr.list_sessions()
    if not s:
        print(f"  {DIM}(no sessions){RESET}"); return
    print(f"\n{BOLD}Sessions ({len(s)}):{RESET}")
    for k, n in sorted(s.items()):
        print(f"  {GREEN}{k}{RESET} ({n} msgs)")
    print()



def repl() -> None:
    # 如果有 runtime_config.json，就优先按配置启动；否则用 demo 配置并写一份简单模板
    cfg = setup_from_runtime_config()
    if cfg:
        mgr, bindings, auto_accounts = cfg
    else:
        mgr, bindings = setup_demo()
        auto_accounts = []
        write_simple_default()
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  /bindings  /route <ch> <peer>  /agents  /sessions /gateway{RESET}")
    print()

    gw_started = False

    # 根据配置自动启动 bridge（如 whatsapp_web / cli）
    if auto_accounts:
        ch_mgr = ChannelManager()
        channels_to_bridge = ch_mgr.build_from_accounts(auto_accounts)
        if channels_to_bridge:
            bridge = MessageCenter(mgr, bindings, channels_to_bridge, run_async_fn=run_async)
            bridge.start()
            print(f"{GREEN}Auto bridge started for: {[a.channel for a in auto_accounts]}{RESET}")

    # CLI 也通过 MessageDispatcher 走同一套逻辑，只是输入循环在这里控制
    cli_channel = CLIChannel()
    cli_dispatcher = MessageCenter(mgr, bindings, [cli_channel], run_async_fn=run_async)

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}"); break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}"); break

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            args = user_input[len(cmd):].strip()
            if cmd == "/bindings":
                cmd_bindings(bindings)
            elif cmd == "/route":
                cmd_route(bindings, mgr, args)
            elif cmd == "/agents":
                cmd_agents(mgr)
            elif cmd == "/sessions":
                cmd_sessions(mgr)
            elif cmd == "/gateway":
                if gw_started:
                    print(f"  {DIM}Already running.{RESET}")
                else:
                    gw = GatewayServer(mgr, bindings)
                    asyncio.run_coroutine_threadsafe(gw.start(), get_event_loop())
                    print(f"{GREEN}Gateway running in background on ws://localhost:8765{RESET}\n")
                    gw_started = True
            else:
                print(f"  {YELLOW}Unknown: {cmd}{RESET}")
            continue

        # 普通文本输入：构造 InboundMessage，复用 MessageDispatcher 的 dispatch 逻辑
        msg = InboundMessage(
            text=user_input,
            sender_id="cli-user",
            channel="cli",
            account_id=cli_channel.account_id,
            peer_id="cli-user",
        )
        try:
            cli_dispatcher.handle_message(msg, cli_channel)
        except Exception as exc:
            print(f"\n{RED}Error: {exc}{RESET}\n")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = get_env_api_key(MODEL_PROVIDER)
    if not api_key:
        print(f"{YELLOW}Error: No API key for provider '{MODEL_PROVIDER}'.{RESET}")
        print(f"{DIM}Set e.g. OPENROUTER_API_KEY or ANTHROPIC_API_KEY in .env{RESET}")
        sys.exit(1)
    repl()

if __name__ == "__main__":
    main()
