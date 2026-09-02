"""Read-only TRAIN memory-mimic banks and causal raw-score materialisation.

The bank metadata is used only to construct TRAIN supervision.  The tensors
returned by :func:`materialize_bank` contain causal visual similarities and
derived temporal statistics; category and track identifiers never enter the
model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase75d.protocol import FrozenTrackTable, PREFIXES, load_frozen_tracks


ROOT = Path(__file__).resolve().parents[2]
STREAM_ROOT = ROOT / "outputs/iclr27_phase76ar/banks"


@dataclass(frozen=True)
class MemoryBank:
    fold: int
    split: str
    episode_id: str
    query_key: str
    candidates: tuple[str, ...]
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    category: int
    video: int
    negative_provenance: dict[str, tuple[int, ...]]


def _bank(row: dict[str, Any]) -> MemoryBank:
    return MemoryBank(
        fold=int(row["fold"]), split=str(row["split"]), episode_id=str(row["episode_id"]),
        query_key=str(row["query_key"]), candidates=tuple(str(x) for x in row["candidates"]),
        positives=tuple(str(x) for x in row["positives"]), negatives=tuple(str(x) for x in row["negatives"]),
        category=int(row.get("category", -1)), video=int(row.get("video", -1)),
        negative_provenance={str(k): tuple(int(x) for x in v) for k, v in row.get("negative_provenance", {}).items()},
    )


def load_memory_banks(fold: int, split: str, *, stream_root: Path = STREAM_ROOT) -> list[MemoryBank]:
    """Load only the frozen Phase76AR ``memory_mimic`` stream."""
    path = stream_root / f"streams_f{int(fold)}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload[split].get("memory_mimic", [])
    banks = [_bank(x) for x in rows]
    if not banks:
        raise RuntimeError(f"no memory-mimic banks for fold={fold} split={split}")
    return banks


def manifest_hash(fold: int, *, stream_root: Path = STREAM_ROOT) -> str:
    path = stream_root / f"streams_f{int(fold)}.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_bank(bank: MemoryBank, table: FrozenTrackTable, *, candidate_prefix: int = 16) -> np.ndarray:
    """Return causal raw cosine scores with shape ``[5, candidates]``.

    The query uses prefix ``p`` while each prior-video support candidate is
    fixed at its completed causal prefix.  This mirrors the memory-mimic
    contract and makes the state update genuinely sequential rather than five
    independent pair calls.
    """
    candidate_vectors = np.asarray([table.raw_vector(k, candidate_prefix) for k in bank.candidates], dtype=np.float32)
    rows: list[np.ndarray] = []
    for prefix in PREFIXES:
        query = table.raw_vector(bank.query_key, prefix).astype(np.float32, copy=False)
        rows.append(np.asarray(candidate_vectors @ query, dtype=np.float32))
    return np.stack(rows, axis=0)


def source_hashes(table: FrozenTrackTable, fold: int, *, stream_root: Path = STREAM_ROOT) -> dict[str, str]:
    return {"csv": table.csv_sha256, "features": table.feature_sha256, "stream": manifest_hash(fold, stream_root=stream_root)}


def frozen_table() -> FrozenTrackTable:
    return load_frozen_tracks()

