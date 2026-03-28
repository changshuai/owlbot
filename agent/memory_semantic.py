"""
Optional offline semantic embeddings for MemoryStore.hybrid_search.

Enable with env MEMORY_EMBEDDING_MODEL set to a Sentence-Transformers model id
or local path. Requires: pip install -r requirements-memory-semantic.txt

If the package or model is unavailable, hybrid_search keeps hash-based vectors.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

MEMORY_EMBEDDING_MODEL_ENV = "MEMORY_EMBEDDING_MODEL"

_encoder = None
_encoder_model_id: Optional[str] = None


def reset_encoder_state() -> None:
    """Clear lazy-loaded model (for tests or model switch)."""
    global _encoder, _encoder_model_id
    _encoder = None
    _encoder_model_id = None


def configured_embedding_model() -> Optional[str]:
    v = os.environ.get(MEMORY_EMBEDDING_MODEL_ENV, "").strip()
    return v or None


def get_sentence_encoder() -> tuple[Any, Optional[str]]:
    """
    Returns (encoder, model_id). encoder is None if disabled, import fails,
    or model load fails.
    """
    global _encoder, _encoder_model_id
    model_id = configured_embedding_model()
    if not model_id:
        return None, None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, model_id
    if _encoder is not None and _encoder_model_id == model_id:
        return _encoder, model_id
    try:
        _encoder = SentenceTransformer(model_id)
        _encoder_model_id = model_id
    except Exception:
        _encoder = None
        _encoder_model_id = None
        return None, model_id
    return _encoder, model_id


def chunk_content_hashes(chunks: list[dict[str, str]]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        raw = f"{c.get('path', '')}\0{c.get('text', '')}".encode("utf-8")
        out.append(hashlib.sha256(raw).hexdigest())
    return out


def load_cached_embeddings(
    cache_path: Path, expected_hashes: list[str]
) -> Optional[Any]:
    try:
        import numpy as np
    except ImportError:
        return None
    if not cache_path.is_file():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        stored = data["hashes"].tolist()
        if len(stored) != len(expected_hashes):
            return None
        if any(a != b for a, b in zip(stored, expected_hashes)):
            return None
        return data["vectors"]
    except Exception:
        return None


def save_embeddings(cache_path: Path, hashes: list[str], vectors: Any) -> None:
    import os
    import tempfile

    import numpy as np

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".npz", dir=cache_path.parent, prefix=".owlbot_embed_"
    )
    os.close(fd)
    try:
        np.savez_compressed(
            tmp_path,
            hashes=np.array(hashes, dtype=object),
            vectors=np.asarray(vectors, dtype=np.float32),
        )
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
