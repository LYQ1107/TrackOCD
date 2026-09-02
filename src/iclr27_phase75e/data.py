"""Read-only TRAIN episode and frozen feature access for Phase75E."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase75d.protocol import FrozenTrackTable, PREFIXES, load_frozen_tracks


ROOT = Path(__file__).resolve().parents[2]
EPISODE_ROOT = ROOT / "outputs/iclr27_phase30/manifests"


@dataclass(frozen=True)
class FitEpisode:
    fold: int
    episode_id: str
    query_key: str
    positive_keys: tuple[str, ...]
    negative_key: str


def manifest_hash(fold: int) -> str:
    path = EPISODE_ROOT / f"episode_manifest_f{fold}.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fit_episodes(fold: int, known_keys: set[str]) -> list[FitEpisode]:
    """Load only fit multi-positive rows and their explicit hard negative."""
    path = EPISODE_ROOT / f"episode_manifest_f{fold}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    out: list[FitEpisode] = []
    for row in manifest["records"]:
        if row.get("split") != "fit" or row.get("kind") != "multi_positive_cross_video":
            continue
        q = str(row["query_track_key"])
        pos = tuple(dict.fromkeys(str(x) for x in row.get("support_track_keys", [])))
        neg = str(row.get("hard_negative_track_key", ""))
        if not q or not pos or not neg or q not in known_keys or neg not in known_keys:
            continue
        if any(x not in known_keys for x in pos):
            continue
        if neg in pos or neg == q:
            continue
        out.append(FitEpisode(fold, str(row["episode_id"]), q, pos, neg))
    if not out:
        raise RuntimeError(f"no legal Phase30 fit episodes for fold {fold}")
    return out


def episode_feature_cache(table: FrozenTrackTable, episodes: list[FitEpisode]) -> dict[str, dict[int, np.ndarray]]:
    """Materialize only causal vectors needed by a fold's fit episodes."""
    keys: set[str] = set()
    for ep in episodes:
        keys.add(ep.query_key); keys.add(ep.negative_key); keys.update(ep.positive_keys)
    cache: dict[str, dict[int, np.ndarray]] = {}
    for key in sorted(keys):
        cache[key] = {p: table.get_frame_sequence(key, p).astype(np.float32, copy=True) for p in PREFIXES}
    return cache


def val_keys_for_fold(fold: int, table: FrozenTrackTable) -> list[str]:
    data = json.loads((EPISODE_ROOT / f"episode_manifest_f{fold}.json").read_text(encoding="utf-8"))
    return sorted({str(r["query_track_key"]) for r in data["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})


def frozen_table() -> FrozenTrackTable:
    return load_frozen_tracks()
