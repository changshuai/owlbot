"""Tests for chat session persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.session_store import chat_session_path, load_chat_session, save_chat_session


class TestSessionStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws_root = Path(self._tmp.name)

    def test_save_and_load_roundtrip(self) -> None:
        def fake_ws(aid: str) -> Path:
            return self.ws_root / f"workspace-{aid}"

        msgs = [
            {"role": "user", "content": "hi", "timestamp": 1},
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}], "timestamp": 2},
        ]
        with patch("agent.session_store.get_agent_workspace", fake_ws):
            save_chat_session("a1", "a1-cli-acc-peer", msgs)
            path = chat_session_path("a1", "a1-cli-acc-peer")
            self.assertTrue(path.is_file())
            loaded = load_chat_session("a1", "a1-cli-acc-peer")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["role"], "user")

    def test_persist_disabled_returns_empty(self) -> None:
        with patch.dict(os.environ, {"PERSIST_CHAT_SESSIONS": "0"}):
            with patch("agent.session_store.get_agent_workspace", lambda aid: self.ws_root / f"w-{aid}"):
                save_chat_session("x", "sk", [{"role": "user", "content": "a"}])
                loaded = load_chat_session("x", "sk")
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
