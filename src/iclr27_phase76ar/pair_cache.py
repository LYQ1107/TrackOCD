"""Atomic detached pair cache with per-match quality features."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.iclr27_phase76a.correspondence import hungarian_match, relation_summary
from src.iclr27_phase76a.raw_anchor import raw_mean_cosine

from .data import PREFIXES


def pair_id(query: str, candidate: str, query_prefix: int, candidate_prefix: int = 16) -> str:
    return f"{query}|{candidate_prefix}|{candidate}|{query_prefix}"


def _local_gaps(sim: np.ndarray, qi: np.ndarray, ci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return matched row/column best-vs-second gaps for each assignment."""
    if not len(qi):
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    row_gaps: list[float] = []; col_gaps: list[float] = []
    for q_idx, c_idx in zip(qi.tolist(), ci.tolist()):
        row = np.asarray(sim[q_idx], dtype=np.float32)
        row_sorted = np.sort(row)[::-1]
        row_best = float(row_sorted[0]) if len(row_sorted) else 0.0
        row_second = float(row_sorted[1]) if len(row_sorted) > 1 else row_best
        col = np.asarray(sim[:, c_idx], dtype=np.float32)
        col_sorted = np.sort(col)[::-1]
        col_best = float(col_sorted[0]) if len(col_sorted) else 0.0
        col_second = float(col_sorted[1]) if len(col_sorted) > 1 else col_best
        row_gaps.append(row_best - row_second)
        col_gaps.append(col_best - col_second)
    return np.asarray(row_gaps, dtype=np.float32), np.asarray(col_gaps, dtype=np.float32)


def _local_temporal(values: np.ndarray, index: int) -> float:
    if len(values) < 2:
        return 1.0 if len(values) else 0.0
    left = values[max(index - 1, 0)]
    right = values[min(index + 1, len(values) - 1)]
    return float(np.mean(left * right))


def match_quality_features(qf: np.ndarray, cf: np.ndarray, match: dict[str, Any]) -> np.ndarray:
    """Five causal per-match features required by the AR contract.

    Columns are: matched cosine, row-best gap, column-best gap, query local
    consistency and candidate local consistency.  All are computed only from
    the query causal prefix and the candidate's causal prefix.
    """
    q = np.asarray(qf, dtype=np.float32); c = np.asarray(cf, dtype=np.float32)
    if not len(q) or not len(c):
        return np.zeros((0, 5), dtype=np.float32)
    sim = q @ c.T
    qi = np.asarray(match.get("q_indices", []), dtype=np.int64)
    ci = np.asarray(match.get("c_indices", []), dtype=np.int64)
    if not len(qi):
        return np.zeros((0, 5), dtype=np.float32)
    row_gap, col_gap = _local_gaps(sim, qi, ci)
    rows: list[list[float]] = []
    for n, (q_idx, c_idx) in enumerate(zip(qi.tolist(), ci.tolist())):
        rows.append([
            float(sim[q_idx, c_idx]), float(row_gap[n]), float(col_gap[n]),
            _local_temporal(q, int(q_idx)), _local_temporal(c, int(c_idx)),
        ])
    return np.asarray(rows, dtype=np.float32)


def build_pair_cache(banks: Iterable[Any], table, path: Path, *, candidate_prefix: int = 16) -> dict[str, Any]:
    pairs = {(str(b.query_key), str(c)) for b in banks for c in b.candidates}
    entries: dict[str, Any] = {}
    for query, candidate in sorted(pairs):
        cf = table.get_frame_sequence(candidate, candidate_prefix)
        for prefix in PREFIXES:
            qf = table.get_frame_sequence(query, prefix)
            match = hungarian_match(qf, cf)
            raw = raw_mean_cosine(qf, cf)
            token_quality = match_quality_features(qf, cf, match)
            entries[pair_id(query, candidate, prefix, candidate_prefix)] = {
                "query_key": query, "candidate_key": candidate,
                "query_prefix": prefix, "candidate_prefix": candidate_prefix,
                "q_indices": match["q_indices"], "c_indices": match["c_indices"],
                "similarities": match["similarities"], "matrix_shape": match["matrix_shape"],
                "summary": relation_summary(qf, cf, match, raw).tolist(),
                "quality_features": token_quality.tolist(), "raw_cosine": float(raw),
            }
    payload = {
        "phase": "Phase76AR", "candidate_prefix": candidate_prefix,
        "prefixes": list(PREFIXES), "pair_count": len(entries), "entries": entries,
        "feature_contract": ["matched_cosine", "row_best_gap", "col_best_gap", "query_local_consistency", "candidate_local_consistency"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":")); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return payload


def load_pair_cache(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def cache_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
