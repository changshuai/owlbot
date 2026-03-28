"""Tests for MemoryStore: diary vs context archive separation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.memory_store import MemoryStore


class TestLoadChunksExcludesArchives(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name) / "workspace-x"
        self.ws.mkdir()
        self.store = MemoryStore(self.ws)

    def test_load_all_chunks_skips_context_archive_files(self) -> None:
        sk = "session-a"
        self.store.write_memory("diary fact about apples", "general", session_key=sk)
        session_dir = self.ws / "memory" / self.store._safe_session_key(sk)
        arch = session_dir / "context_archive_20990101_120000_000000.jsonl"
        arch.write_text(
            json.dumps({"role": "user", "content": "secret from trimmed history"}) + "\n",
            encoding="utf-8",
        )

        chunks = self.store._load_all_chunks(session_key=sk)
        joined = " ".join(c["text"] for c in chunks if isinstance(c.get("text"), str))
        self.assertIn("apples", joined)
        self.assertNotIn("secret from trimmed", joined)

    def test_hybrid_search_does_not_match_archive_only_content(self) -> None:
        sk = "session-b"
        session_dir = self.ws / "memory" / self.store._safe_session_key(sk)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "context_archive_20990102_000000_000001.jsonl").write_text(
            json.dumps({"role": "user", "content": "unique_archive_token_xyz"}) + "\n",
            encoding="utf-8",
        )

        hits = self.store.hybrid_search("unique_archive_token_xyz", top_k=5, session_key=sk)
        self.assertEqual(len(hits), 0)

    def test_diary_jsonl_still_indexed(self) -> None:
        sk = "session-c"
        self.store.write_memory("visible memory token", "fact", session_key=sk)
        hits = self.store.hybrid_search("visible memory", top_k=5, session_key=sk)
        self.assertTrue(any("visible" in h["snippet"].lower() for h in hits))


if __name__ == "__main__":
    unittest.main()
