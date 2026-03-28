from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.paths import WORKSPACE_DIR, get_agent_workspace


def _safe_session_key(session_key: str) -> str:
    key = (session_key or "").strip()
    if not key:
        key = "default-session"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", key).strip("._-")
    return safe or "default-session"


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    rel_path: str  # relative to WORKSPACE_DIR
    bytes: int


def save_artifact_text(
    *,
    agent_id: str,
    session_key: str,
    kind: str,
    text: str,
    meta: dict[str, Any] | None = None,
) -> ArtifactRef:
    """
    Persist a large text blob for later retrieval.

    Stored under workspace-<agent_id>/artifacts/<session_key>/.
    """
    ws = get_agent_workspace(agent_id)
    root = ws / "artifacts" / _safe_session_key(session_key)
    root.mkdir(parents=True, exist_ok=True)

    artifact_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}"
    base = f"{kind}_{artifact_id}"
    txt_path = root / f"{base}.txt"
    json_path = root / f"{base}.json"

    payload = text if isinstance(text, str) else str(text)
    txt_path.write_text(payload, encoding="utf-8")

    meta_out = {
        "artifact_id": artifact_id,
        "kind": kind,
        "created_at_ms": int(time.time() * 1000),
        "chars": len(payload),
        "bytes": txt_path.stat().st_size,
        "text_path": str(txt_path),
        "agent_id": agent_id,
        "session_key": session_key,
        "meta": meta or {},
    }
    json_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")

    rel_txt = str(txt_path.resolve()).replace(str(WORKSPACE_DIR.resolve()) + "/", "")
    return ArtifactRef(artifact_id=artifact_id, rel_path=rel_txt, bytes=txt_path.stat().st_size)


def load_artifact_text(*, agent_id: str, session_key: str, artifact_id: str) -> str:
    ws = get_agent_workspace(agent_id)
    root = ws / "artifacts" / _safe_session_key(session_key)
    if not root.is_dir():
        raise FileNotFoundError("No artifacts for this session.")

    # Find matching artifact by scanning metadata (fast enough for per-session dirs).
    prefix = f"_"+artifact_id
    for jf in root.glob(f"*_*.json"):
        if prefix in jf.stem:
            meta = json.loads(jf.read_text(encoding="utf-8"))
            txt_path = Path(meta.get("text_path", ""))
            if txt_path.is_file():
                return txt_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Artifact not found: {artifact_id}")

