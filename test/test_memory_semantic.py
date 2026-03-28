"""Tests for optional semantic embeddings in MemoryStore.hybrid_search."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent.memory_semantic import (
    chunk_content_hashes,
    configured_embedding_model,
    load_cached_embeddings,
    reset_encoder_state,
    save_embeddings,
)
from agent.memory_store import MemoryStore


class _MockEncoder:
    """Deterministic vectors: query aligns with first chunk, not third."""

    _table: dict[str, np.ndarray] = {
        "user prefers python for scripting": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "unrelated stock market news": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "random gardening tips": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "does the user enjoy python": np.array([0.92, 0.08, 0.0], dtype=np.float32),
    }

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        rows = []
        for t in texts:
            v = self._table.get(t, np.ones(3, dtype=np.float32) / 3.0)
            v = np.asarray(v, dtype=np.float32)
            if normalize_embeddings:
                n = float(np.linalg.norm(v)) or 1.0
                v = v / n
            rows.append(v)
        return np.stack(rows)


class _EvergreenMockEncoder(_MockEncoder):
    """MEMORY.md chunk + diary; query aligned with evergreen."""

    _table = {
        **_MockEncoder._table,
        "rust systems programming preference": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "diary noise python toys": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "interest in rust tooling": np.array([0.95, 0.05, 0.0], dtype=np.float32),
    }


class _RaisingEncoder:
    def encode(self, *args, **kwargs):
        raise RuntimeError("simulated encode failure")


class TestMemorySemanticConfig(unittest.TestCase):
    def test_configured_model_none_when_unset(self) -> None:
        prev = os.environ.pop("MEMORY_EMBEDDING_MODEL", None)
        try:
            self.assertIsNone(configured_embedding_model())
        finally:
            if prev is not None:
                os.environ["MEMORY_EMBEDDING_MODEL"] = prev

    def test_configured_model_strip_empty(self) -> None:
        with patch.dict(os.environ, {"MEMORY_EMBEDDING_MODEL": "  \t  "}):
            self.assertIsNone(configured_embedding_model())

    def test_configured_model_returns_value(self) -> None:
        with patch.dict(os.environ, {"MEMORY_EMBEDDING_MODEL": "some-model-id"}):
            self.assertEqual(configured_embedding_model(), "some-model-id")


class TestSemanticHybridSearch(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name) / "ws"
        self.ws.mkdir()
        self.store = MemoryStore(self.ws)
        reset_encoder_state()

    def tearDown(self) -> None:
        reset_encoder_state()

    def test_paraphrase_ranks_target_first_with_mock_encoder(self) -> None:
        sk = "sem-1"
        self.store.write_memory(
            "user prefers python for scripting", "fact", session_key=sk
        )
        self.store.write_memory("unrelated stock market news", "news", session_key=sk)
        self.store.write_memory("random gardening tips", "note", session_key=sk)

        mock = _MockEncoder()
        with patch("agent.memory_store.memory_semantic.get_sentence_encoder", return_value=(mock, "mock")):
            hits = self.store.hybrid_search(
                "does the user enjoy python", top_k=3, session_key=sk
            )
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("python", hits[0]["snippet"].lower())

    def test_embedding_cache_skips_full_reencode(self) -> None:
        sk = "sem-cache"
        self.store.write_memory("user prefers python for scripting", "f", session_key=sk)
        self.store.write_memory("unrelated stock market news", "n", session_key=sk)
        self.store.write_memory("random gardening tips", "g", session_key=sk)
        mock = _MockEncoder()
        calls: list[int] = []

        def enc(texts, **kwargs):
            calls.append(len(texts))
            return _MockEncoder.encode(mock, texts, **kwargs)

        mock.encode = enc  # type: ignore[method-assign]

        with patch("agent.memory_store.memory_semantic.get_sentence_encoder", return_value=(mock, "mock")):
            self.store.hybrid_search("does the user enjoy python", top_k=2, session_key=sk)
            self.store.hybrid_search("does the user enjoy python", top_k=2, session_key=sk)

        session_dir = self.ws / "memory" / self.store._safe_session_key(sk)
        self.assertTrue((session_dir / ".owlbot_semantic.npz").is_file())
        # First search: encode 3 chunk texts + query; second: cache hit, query only
        self.assertEqual(calls, [3, 1, 1])

    def test_chunk_hashes_and_cache_roundtrip(self) -> None:
        chunks = [
            {"path": "a.jsonl", "text": "hello"},
            {"path": "a.jsonl", "text": "world"},
        ]
        h = chunk_content_hashes(chunks)
        self.assertEqual(len(h), 2)
        self.assertNotEqual(h[0], h[1])

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.npz"
            vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            save_embeddings(p, h, vecs)
            loaded = load_cached_embeddings(p, h)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_array_almost_equal(loaded, vecs)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.npz"
            save_embeddings(p, h, vecs)
            bad_hashes = [h[0], "wrong"]
            self.assertIsNone(load_cached_embeddings(p, bad_hashes))

    def test_load_cached_embeddings_missing_file(self) -> None:
        self.assertIsNone(
            load_cached_embeddings(Path("/nonexistent/owlbot_no_such.npz"), ["a"])
        )

    def test_chunk_content_hashes_stable(self) -> None:
        chunks = [{"path": "f.jsonl", "text": "stable body"}]
        a = chunk_content_hashes(chunks)
        b = chunk_content_hashes(chunks)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(a[0]), 64)

    def test_hybrid_search_falls_back_when_encoder_unavailable(self) -> None:
        sk = "fb-none"
        self.store.write_memory("unique_fallback_token_yyzz", "f", session_key=sk)
        with patch(
            "agent.memory_store.memory_semantic.get_sentence_encoder",
            return_value=(None, "unavailable"),
        ):
            hits = self.store.hybrid_search(
                "unique_fallback_token_yyzz", top_k=5, session_key=sk
            )
        self.assertTrue(hits)
        self.assertIn("yyzz", hits[0]["snippet"].lower())

    def test_hybrid_search_falls_back_when_embedding_raises(self) -> None:
        sk = "fb-raise"
        self.store.write_memory("unique_raise_token_qwop99", "f", session_key=sk)
        with patch(
            "agent.memory_store.memory_semantic.get_sentence_encoder",
            return_value=(_RaisingEncoder(), "bad"),
        ):
            hits = self.store.hybrid_search(
                "unique_raise_token_qwop99", top_k=5, session_key=sk
            )
        self.assertTrue(hits)
        self.assertIn("qwop99", hits[0]["snippet"].lower())

    def test_semantic_path_indexes_memory_md(self) -> None:
        (self.ws / "MEMORY.md").write_text(
            "rust systems programming preference\n", encoding="utf-8"
        )
        sk = "eg"
        self.store.write_memory("diary noise python toys", "n", session_key=sk)
        mock = _EvergreenMockEncoder()
        with patch(
            "agent.memory_store.memory_semantic.get_sentence_encoder",
            return_value=(mock, "mock"),
        ):
            hits = self.store.hybrid_search(
                "interest in rust tooling", top_k=2, session_key=sk
            )
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("rust", hits[0]["snippet"].lower())

    def test_embedding_cache_invalidates_when_new_memory_added(self) -> None:
        sk = "inv"
        self.store.write_memory("user prefers python for scripting", "f", session_key=sk)
        self.store.write_memory("unrelated stock market news", "n", session_key=sk)
        self.store.write_memory("random gardening tips", "g", session_key=sk)
        mock = _MockEncoder()
        calls: list[int] = []

        def enc(texts, **kwargs):
            calls.append(len(texts))
            return _MockEncoder.encode(mock, texts, **kwargs)

        mock.encode = enc  # type: ignore[method-assign]

        with patch(
            "agent.memory_store.memory_semantic.get_sentence_encoder",
            return_value=(mock, "mock"),
        ):
            self.store.hybrid_search("does the user enjoy python", top_k=2, session_key=sk)
            self.store.write_memory("fourth chunk about databases", "f", session_key=sk)
            self.store.hybrid_search("does the user enjoy python", top_k=2, session_key=sk)

        self.assertEqual(calls, [3, 1, 4, 1])

    def test_separate_semantic_cache_per_session(self) -> None:
        mock = _MockEncoder()
        with patch(
            "agent.memory_store.memory_semantic.get_sentence_encoder",
            return_value=(mock, "mock"),
        ):
            for sk in ["sess-a", "sess-b"]:
                self.store.write_memory(
                    "user prefers python for scripting", "f", session_key=sk
                )
                self.store.hybrid_search(
                    "does the user enjoy python", top_k=1, session_key=sk
                )
        p1 = self.ws / "memory" / self.store._safe_session_key("sess-a") / ".owlbot_semantic.npz"
        p2 = self.ws / "memory" / self.store._safe_session_key("sess-b") / ".owlbot_semantic.npz"
        self.assertTrue(p1.is_file())
        self.assertTrue(p2.is_file())


@unittest.skipUnless(
    os.environ.get("OWLBOT_TEST_SEMANTIC_MODEL", "").strip(),
    "Set OWLBOT_TEST_SEMANTIC_MODEL to a Sentence-Transformers id to run integration test.",
)
class TestSemanticIntegrationOptional(unittest.TestCase):
    def test_real_model_paraphrase(self) -> None:
        model = os.environ["OWLBOT_TEST_SEMANTIC_MODEL"].strip()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        ws = Path(self._tmp.name) / "w"
        ws.mkdir()
        store = MemoryStore(ws)
        reset_encoder_state()
        sk = "int"
        store.write_memory("The user's favorite language is Python.", "f", session_key=sk)
        store.write_memory("Quarterly revenue was flat.", "n", session_key=sk)

        with patch.dict(os.environ, {**os.environ, "MEMORY_EMBEDDING_MODEL": model}):
            reset_encoder_state()
            hits = store.hybrid_search(
                "What programming language does the user like?",
                top_k=2,
                session_key=sk,
            )
        reset_encoder_state()
        self.assertTrue(hits)
        self.assertIn("python", hits[0]["snippet"].lower())


if __name__ == "__main__":
    unittest.main()
