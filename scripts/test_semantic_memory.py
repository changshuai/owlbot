#!/usr/bin/env python3
"""
Smoke-test semantic memory (MEMORY_EMBEDDING_MODEL + sentence-transformers).

Usage (from repo root):
  python scripts/test_semantic_memory.py
  MEMORY_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5 python scripts/test_semantic_memory.py
  python scripts/test_semantic_memory.py --model sentence-transformers/all-MiniLM-L6-v2

Requires: pip install -r requirements-memory-semantic.txt
Loads:    <repo>/.env if present (python-dotenv).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_path = _repo_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Test MEMORY_EMBEDDING_MODEL hybrid_search.")
    parser.add_argument(
        "--model",
        metavar="ID_OR_PATH",
        help="Override MEMORY_EMBEDDING_MODEL for this run only",
    )
    args = parser.parse_args()

    _load_dotenv()
    if args.model:
        os.environ["MEMORY_EMBEDDING_MODEL"] = args.model.strip()

    model = os.environ.get("MEMORY_EMBEDDING_MODEL", "").strip()
    if not model:
        print(
            "MEMORY_EMBEDDING_MODEL is not set.\n"
            "  Add it to .env or export it, or pass --model <hf_id_or_local_path>",
            file=sys.stderr,
        )
        return 2

    print("MEMORY_EMBEDDING_MODEL:", model)

    try:
        from agent.memory_semantic import get_sentence_encoder, reset_encoder_state
        from agent.memory_store import MemoryStore
    except ImportError as e:
        print("Import error (run from repo root?):", e, file=sys.stderr)
        return 2

    reset_encoder_state()
    enc, mid = get_sentence_encoder()
    if enc is None:
        print(
            "Encoder failed to load.\n"
            "  - pip install -r requirements-memory-semantic.txt\n"
            "  - Check model id / local path and network (first download).",
            file=sys.stderr,
        )
        return 1
    print("Encoder loaded OK.")

    td = Path(tempfile.mkdtemp(prefix="owlbot_sem_test_"))
    ws = td / "w"
    ws.mkdir()
    store = MemoryStore(ws)
    sk = "smoke"
    store.write_memory("The user's favorite language is Python.", "fact", session_key=sk)
    store.write_memory("Quarterly revenue was flat.", "news", session_key=sk)

    query = "What programming language does the user like?"
    hits = store.hybrid_search(query, top_k=3, session_key=sk)

    print("Query:", query)
    print("Top hits:")
    for i, h in enumerate(hits, 1):
        snip = h.get("snippet", "")
        print(f"  {i}. score={h.get('score')} path={h.get('path')!r}")
        print(f"     {snip[:120]}{'...' if len(snip) > 120 else ''}")

    cache = ws / "memory" / store._safe_session_key(sk) / ".owlbot_semantic.npz"
    print("Embedding cache:", cache, "(exists:", cache.is_file(), ")")

    if not hits:
        print("FAIL: no hits", file=sys.stderr)
        return 3

    top = (hits[0].get("snippet") or "").lower()
    if "python" in top:
        print("PASS: top result mentions Python.")
        return 0

    print(
        "WARN: top snippet did not contain 'python' (keyword merge or model noise). "
        "Check scores above manually.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(_repo_root()))
    raise SystemExit(main())
