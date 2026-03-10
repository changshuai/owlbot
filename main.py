import asyncio, sys, threading
from dispatch.agent_manager import BindingTable, AgentManager, DEFAULT_AGENT_ID, build_session_key, normalize_agent_id
from common.colors import DIM, RESET, BOLD, CYAN, GREEN, YELLOW, MAGENTA, BLUE, RED
from dispatch.agent_manager import AgentManager, setup_demo
from dispatch.agent_loop import run_agent
from LLMs import get_env_api_key
from dispatch.agent_manager import resolve_route
from dispatch.agent_loop import MODEL_PROVIDER, MODEL_ID
from dispatch.gateway import GatewayServer
from dispatch.config_runtime import setup_from_config, write_simple_default

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
    cfg = setup_from_config()
    if cfg:
        mgr, bindings, auto_accounts = cfg
    else:
        mgr, bindings = setup_demo()
        auto_accounts = []
        write_simple_default()
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  claw0  |  Section 05: Gateway & Routing{RESET}")
    print(f"{DIM}  Model: {MODEL_PROVIDER}/{MODEL_ID}{RESET}")
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  /bindings  /route <ch> <peer>  /agents  /sessions  /switch <id>  /gateway  /bridge [whatsapp_web]{RESET}")
    print()

    ch, pid = "cli", "repl-user"
    force_agent = ""
    gw_started = False
    bridge_started = False

    # 根据配置自动启动 bridge（如 whatsapp_web）
    if auto_accounts:
        from dispatch.channel_bridge import ChannelBridge
        channels_to_bridge = []
        for acc in auto_accounts:
            if acc.channel == "whatsapp_web":
                try:
                    from channels.whatsapp_web import WhatsAppWebChannel
                    wa = WhatsAppWebChannel(acc)
                    channels_to_bridge.append(wa)
                except Exception as e:
                    print(f"  {RED}Auto bridge whatsapp_web failed: {e}{RESET}")
        if channels_to_bridge:
            bridge = ChannelBridge(mgr, bindings, channels_to_bridge, run_async_fn=run_async)
            bridge.start()
            bridge_started = True
            print(f"{GREEN}Auto bridge started for: {[a.channel for a in auto_accounts]}{RESET}")

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
            elif cmd == "/switch":
                if not args:
                    print(f"  {DIM}force={force_agent or '(off)'}{RESET}")
                elif args.lower() == "off":
                    force_agent = ""
                    print(f"  {DIM}Routing mode restored.{RESET}")
                else:
                    aid = normalize_agent_id(args)
                    if mgr.get_agent(aid):
                        force_agent = aid
                        print(f"  {GREEN}Forcing: {aid}{RESET}")
                    else:
                        print(f"  {YELLOW}Not found: {aid}{RESET}")
            elif cmd == "/gateway":
                if gw_started:
                    print(f"  {DIM}Already running.{RESET}")
                else:
                    gw = GatewayServer(mgr, bindings)
                    asyncio.run_coroutine_threadsafe(gw.start(), get_event_loop())
                    print(f"{GREEN}Gateway running in background on ws://localhost:8765{RESET}\n")
                    gw_started = True
            elif cmd == "/bridge":
                if bridge_started:
                    print(f"  {DIM}Bridge already running.{RESET}")
                else:
                    channels_to_bridge = []
                    if args.strip().lower() == "whatsapp_web":
                        try:
                            from channels.whatsapp_web import WhatsAppWebChannel
                            from channels.types_ import ChannelAccount
                            acc = ChannelAccount(channel="whatsapp_web", account_id="wa-default", token="", config={})
                            wa = WhatsAppWebChannel(acc)
                            channels_to_bridge.append(wa)
                        except Exception as e:
                            import sys
                            print(f"  {RED}WhatsApp Web channel failed: {e}{RESET}")
                            print(f"  {DIM}Using Python: {sys.executable}{RESET}")
                            print(f"  {DIM}Use the env with whatsapp-web-py (e.g. conda activate owlbot), then run this script with that Python.{RESET}")
                            print(f"  {DIM}Or use WhatsApp Cloud API: channels/whatsapp.py (no QR).{RESET}")
                    if channels_to_bridge:
                        from dispatch.channel_bridge import ChannelBridge
                        bridge = ChannelBridge(mgr, bindings, channels_to_bridge, run_async_fn=run_async)
                        bridge.start()
                        bridge_started = True
                        print(f"{GREEN}Channel bridge started (poll → agent → reply). Scan QR if WhatsApp Web shows it.{RESET}\n")
                    else:
                        print(f"  {YELLOW}Usage: /bridge whatsapp_web{RESET}")
            else:
                print(f"  {YELLOW}Unknown: {cmd}{RESET}")
            continue

        if force_agent:
            agent_id = force_agent
            a = mgr.get_agent(agent_id)
            session_key = build_session_key(agent_id, channel=ch, peer_id=pid,
                                            dm_scope=a.dm_scope if a else "per-peer")
        else:
            agent_id, session_key = resolve_route(bindings, mgr, channel=ch, peer_id=pid)

        agent = mgr.get_agent(agent_id)
        name = agent.name if agent else agent_id
        print(f"  {DIM}-> {name} ({agent_id}) | {session_key}{RESET}")

        try:
            reply = run_async(run_agent(mgr, agent_id, session_key, user_input))
        except Exception as exc:
            print(f"\n{RED}Error: {exc}{RESET}\n"); continue
        print(f"\n{GREEN}{BOLD}{name}:{RESET} {reply}\n")

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
