"""Small detached Hungarian-index cache; raw features remain in the frozen table."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .correspondence import hungarian_match, relation_summary
from .raw_anchor import raw_mean_cosine


PREFIXES = (1, 2, 4, 8, 16)


def pair_id(q: str, c: str, qprefix: int, cprefix: int = 16) -> str:
    return f"{q}|{cprefix}|{c}|{qprefix}"


def build_pair_cache(banks, table, path: Path, *, candidate_prefix: int = 16) -> dict[str, Any]:
    pairs = {(b.query_key, c) for b in banks for c in b.candidates}
    entries: dict[str, Any] = {}
    for q, c in sorted(pairs):
        cf = table.get_frame_sequence(c, candidate_prefix)
        for p in PREFIXES:
            qf = table.get_frame_sequence(q, p)
            match = hungarian_match(qf, cf)
            raw = raw_mean_cosine(qf, cf)
            entries[pair_id(q, c, p, candidate_prefix)] = {
                "query_key": q, "candidate_key": c, "query_prefix": p, "candidate_prefix": candidate_prefix,
                "q_indices": match["q_indices"], "c_indices": match["c_indices"],
                "similarities": match["similarities"], "matrix_shape": match["matrix_shape"],
                "summary": relation_summary(qf, cf, match, raw).tolist(), "raw_cosine": float(raw),
            }
    payload = {"phase": "Phase76A", "candidate_prefix": candidate_prefix, "prefixes": PREFIXES, "pair_count": len(entries), "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as h:
            json.dump(payload, h, sort_keys=True, separators=(",", ":")); h.write("\n"); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return payload


def load_pair_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def cache_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
