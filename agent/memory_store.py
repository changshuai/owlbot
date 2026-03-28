from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.paths import WORKSPACE_DIR, get_agent_workspace

from agent import memory_semantic

class MemoryStore:
    """
    Per-agent memory store.

    - Evergreen facts: MEMORY.md in the agent workspace root.
    - Session-scoped logs: memory/<session_key>/YYYY-MM-DD.jsonl
    - Context trim snapshots: memory/<session_key>/context_archive_<ts>.jsonl (excluded from hybrid search)
    - Hybrid search: keyword TF-IDF + vector leg (semantic if MEMORY_EMBEDDING_MODEL
      is set and sentence-transformers works; else hash-based pseudo-embeddings),
      temporal decay, MMR.
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.memory_root = workspace_dir / "memory"
        self.memory_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_session_key(session_key: str) -> str:
        key = (session_key or "").strip()
        if not key:
            key = "default-session"
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", key).strip("._-")
        return safe or "default-session"

    def _session_memory_dir(self, session_key: str) -> Path:
        session_dir = self.memory_root / self._safe_session_key(session_key)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def archive_messages(self, session_key: str, messages: list[dict[str, Any]]) -> str:
        """
        Persist removed session messages for context-window trimming.

        Returns a workspace-relative reference like memory/<safe_session>/context_archive_<ts>.jsonl
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        session_dir = self._session_memory_dir(session_key)
        path = session_dir / f"context_archive_{ts}.jsonl"
        try:
            with open(path, "w", encoding="utf-8") as f:
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
            safe_session = self._safe_session_key(session_key)
            return f"memory/{safe_session}/{path.name}"
        except Exception as exc:
            return f"Error writing archive: {exc}"

    def write_memory(
        self, content: str, category: str = "general", session_key: str = ""
    ) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        session_dir = self._session_memory_dir(session_key)
        path = session_dir / f"{today}.jsonl"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "content": content,
            "session_key": session_key,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            safe_session = self._safe_session_key(session_key)
            return f"Memory saved to memory/{safe_session}/{today}.jsonl ({category})"
        except Exception as exc:
            return f"Error writing memory: {exc}"

    def load_evergreen(self) -> str:
        path = self.workspace_dir / "MEMORY.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_all_chunks(self, session_key: str = "") -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        evergreen = self.load_evergreen()
        if evergreen:
            for para in evergreen.split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"path": "MEMORY.md", "text": para})
        session_dir = self._session_memory_dir(session_key)
        if session_dir.is_dir():
            for jf in sorted(session_dir.glob("*.jsonl")):
                if jf.name.startswith("context_archive_"):
                    continue
                try:
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("content", "")
                        if text:
                            cat = entry.get("category", "")
                            label = f"{jf.name} [{cat}]" if cat else jf.name
                            chunks.append({"path": label, "text": text})
                except Exception:
                    continue
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]

    def search_memory(
        self, query: str, top_k: int = 5, session_key: str = ""
    ) -> list[dict[str, Any]]:
        chunks = self._load_all_chunks(session_key=session_key)
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        chunk_tokens = [self._tokenize(c["text"]) for c in chunks]

        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        n = len(chunks)

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {
                t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1)
                for t, c in tf.items()
            }

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qvec = tfidf(query_tokens)
        scored: list[dict[str, Any]] = []
        for i, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(qvec, tfidf(tokens))
            if score > 0.0:
                snippet = chunks[i]["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                scored.append(
                    {
                        "path": chunks[i]["path"],
                        "score": round(score, 4),
                        "snippet": snippet,
                    }
                )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # --- Hybrid Memory Search Enhancement (vector + keyword + temporal + MMR) ---

    @staticmethod
    def _hash_vector(text: str, dim: int = 64) -> list[float]:
        tokens = MemoryStore._tokenize(text)
        vec = [0.0] * dim
        for token in tokens:
            h = hash(token)
            for i in range(dim):
                bit = (h >> (i % 62)) & 1
                vec[i] += 1.0 if bit else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def _vector_cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _jaccard_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return inter / union if union else 0.0

    def _vector_search(
        self, query: str, chunks: list[dict[str, str]], top_k: int = 10
    ) -> list[dict[str, Any]]:
        q_vec = self._hash_vector(query)
        scored = []
        for chunk in chunks:
            c_vec = self._hash_vector(chunk["text"])
            score = self._vector_cosine(q_vec, c_vec)
            if score > 0.0:
                scored.append({"chunk": chunk, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _keyword_search(
        self, query: str, chunks: list[dict[str, str]], top_k: int = 10
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        chunk_tokens = [self._tokenize(c["text"]) for c in chunks]
        n = len(chunks)
        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for t in set(tokens):
                df[t] = df.get(t) + 1 if t in df else 1

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {
                t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1)
                for t, c in tf.items()
            }

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qvec = tfidf(query_tokens)
        scored = []
        for i, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(qvec, tfidf(tokens))
            if score > 0.0:
                scored.append({"chunk": chunks[i], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _embedding_vector_search(
        self,
        encoder: Any,
        query: str,
        chunks: list[dict[str, str]],
        session_key: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Cosine similarity via normalized embeddings; cached per session under memory/."""
        try:
            import numpy as np
        except ImportError:
            return self._vector_search(query, chunks, top_k=top_k)

        session_dir = self._session_memory_dir(session_key)
        cache_path = session_dir / ".owlbot_semantic.npz"
        hashes = memory_semantic.chunk_content_hashes(chunks)
        vecs = memory_semantic.load_cached_embeddings(cache_path, hashes)
        texts = [c["text"] for c in chunks]
        if vecs is None and texts:
            raw = encoder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            vecs = np.asarray(raw, dtype=np.float32)
            try:
                memory_semantic.save_embeddings(cache_path, hashes, vecs)
            except Exception:
                pass
        elif vecs is None:
            return []

        q_raw = encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        q = np.asarray(q_raw[0], dtype=np.float32)
        sims = vecs @ q
        order = np.argsort(-sims)[:top_k]
        scored: list[dict[str, Any]] = []
        for i in order:
            idx = int(i)
            s = float(sims[idx])
            if s > 0.0:
                scored.append({"chunk": chunks[idx], "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    @staticmethod
    def _merge_hybrid_results(
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for r in vector_results:
            key = r["chunk"]["text"][:100]
            merged[key] = {"chunk": r["chunk"], "score": r["score"] * vector_weight}
        for r in keyword_results:
            key = r["chunk"]["text"][:100]
            if key in merged:
                merged[key]["score"] += r["score"] * text_weight
            else:
                merged[key] = {"chunk": r["chunk"], "score": r["score"] * text_weight}
        result = list(merged.values())
        result.sort(key=lambda x: x["score"], reverse=True)
        return result

    @staticmethod
    def _temporal_decay(
        results: list[dict[str, Any]], decay_rate: float = 0.01
    ) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        for r in results:
            path = r["chunk"].get("path", "")
            age_days = 0.0
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path)
            if date_match:
                try:
                    chunk_date = datetime.strptime(
                        date_match.group(1), "%Y-%m-%d"
                    ).replace(tzinfo=timezone.utc)
                    age_days = (now - chunk_date).total_seconds() / 86400.0
                except ValueError:
                    pass
            r["score"] *= math.exp(-decay_rate * age_days)
        return results

    @staticmethod
    def _mmr_rerank(
        results: list[dict[str, Any]],
        lambda_param: float = 0.7,
    ) -> list[dict[str, Any]]:
        if len(results) <= 1:
            return results
        tokenized = [MemoryStore._tokenize(r["chunk"]["text"]) for r in results]
        selected: list[int] = []
        remaining = list(range(len(results)))
        reranked: list[dict[str, Any]] = []
        while remaining:
            best_idx = -1
            best_mmr = float("-inf")
            for idx in remaining:
                relevance = results[idx]["score"]
                max_sim = 0.0
                for sel_idx in selected:
                    sim = MemoryStore._jaccard_similarity(
                        tokenized[idx], tokenized[sel_idx]
                    )
                    if sim > max_sim:
                        max_sim = sim
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
            reranked.append(results[best_idx])
        return reranked

    def hybrid_search(
        self, query: str, top_k: int = 5, session_key: str = ""
    ) -> list[dict[str, Any]]:
        chunks = self._load_all_chunks(session_key=session_key)
        if not chunks:
            return []
        keyword_results = self._keyword_search(query, chunks, top_k=10)
        encoder, _ = memory_semantic.get_sentence_encoder()
        if encoder is not None:
            try:
                vector_results = self._embedding_vector_search(
                    encoder, query, chunks, session_key, top_k=10
                )
            except Exception:
                vector_results = self._vector_search(query, chunks, top_k=10)
        else:
            vector_results = self._vector_search(query, chunks, top_k=10)
        merged = self._merge_hybrid_results(vector_results, keyword_results)
        decayed = self._temporal_decay(merged)
        reranked = self._mmr_rerank(decayed)
        result = []
        for r in reranked[:top_k]:
            snippet = r["chunk"]["text"]
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            result.append(
                {
                    "path": r["chunk"]["path"],
                    "score": round(r["score"], 4),
                    "snippet": snippet,
                }
            )
        return result

    def get_stats(self, session_key: str = "") -> dict[str, Any]:
        evergreen = self.load_evergreen()
        session_dir = self._session_memory_dir(session_key)
        daily_files = [
            f
            for f in (session_dir.glob("*.jsonl") if session_dir.is_dir() else [])
            if not f.name.startswith("context_archive_")
        ]
        total_entries = 0
        for f in daily_files:
            try:
                total_entries += sum(
                    1
                    for line in f.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            except Exception:
                pass
        return {
            "evergreen_chars": len(evergreen),
            "daily_files": len(daily_files),
            "daily_entries": total_entries,
        }


_MEMORY_STORES: dict[str, MemoryStore] = {}




def get_memory_store(agent_id: str) -> MemoryStore:
    """Get or create a MemoryStore bound to this agent's workspace."""
    if agent_id not in _MEMORY_STORES:
        ws = get_agent_workspace(agent_id)
        _MEMORY_STORES[agent_id] = MemoryStore(ws)
    return _MEMORY_STORES[agent_id]

