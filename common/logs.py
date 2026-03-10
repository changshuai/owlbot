from dataclasses import dataclass, field
from typing import Any
from abc import ABC, abstractmethod
from common.colors import CYAN, GREEN, YELLOW, DIM, RESET, BOLD, RED, BLUE


def print_assistant(text: str, ch: str = "cli") -> None:
    prefix = f"[{ch}] " if ch != "cli" else ""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {prefix}{text}\n")

def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")

def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")

def print_channel(text: str) -> None:
    print(f"{BLUE}{text}{RESET}")