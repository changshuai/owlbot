from __future__ import annotations

import asyncio, sys, threading
from message.route_ import BindingTable, AgentManager
from common.colors import DIM, RESET, BOLD, CYAN, GREEN, YELLOW, MAGENTA, BLUE, RED
from message.route_ import setup_demo
from agent.agent_loop import run_agent
from LLMs import get_env_api_key
from message.route_ import resolve_route
from agent.agent_loop import MODEL_PROVIDER, MODEL_ID
from message.gateway import GatewayServer
from config.config_runtime import setup_from_config as setup_from_runtime_config, write_simple_default
from channels.channel_manager import ChannelManager
from channels.types_ import ChannelConfig
from channels.cli_ import CLIChannel
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
        # 颜色按“具体程度”区分：peer > account > channel > global
        spec = 0
        if b.channel != "*":
            spec += 1
        if b.account_id != "*":
            spec += 1
        if b.peer_id != "*":
            spec += 1
        palette = [DIM, GREEN, CYAN, MAGENTA]
        c = palette[min(spec, len(palette) - 1)]
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
        print(f"  {CYAN}{a.id}{RESET} ({a.name})  model={a.effective_model}")
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
    cfg = setup_from_runtime_config()
    if cfg:
        mgr, bindings, channels = cfg
    else:
        mgr, bindings = setup_demo()
        channels = []
        write_simple_default()
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  /bindings  /route <ch> <peer>  /agents  /sessions /gateway(coming soon){RESET}")
    print()

    # add cli channel
    cli_channel = CLIChannel(ChannelConfig(channel="cli", account_id="cli-local"))
    channels.append(cli_channel)
    # start message center
    message_center = MessageCenter(mgr, bindings, channels, run_async_fn=run_async)
    message_center.start()
    print(f"{GREEN}Message center started{RESET}")

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
            # elif cmd == "/gateway":
            #     if gw_started:
            #         print(f"  {DIM}Already running.{RESET}")
            #     else:
            #         gw = GatewayServer(mgr, bindings)
            #         asyncio.run_coroutine_threadsafe(gw.start(), get_event_loop())
            #         print(f"{GREEN}Gateway running in background on ws://localhost:8765{RESET}\n")
            #         gw_started = True
            else:
                print(f"  {YELLOW}Unknown: {cmd}{RESET}")
            continue

        cli_channel.handle_message(user_input)
        
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
