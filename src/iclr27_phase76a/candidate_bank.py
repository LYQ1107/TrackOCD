"""Deterministic, metadata-only candidate banks for Phase76A."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateBank:
    fold: int
    split: str
    episode_id: str
    query_key: str
    candidates: tuple[str, ...]
    positives: tuple[str, ...]
    negatives: tuple[str, ...]
    raw_scores: tuple[float, ...]
    category: int
    video: int


def _stable_order(items: list[str], seed: int) -> list[str]:
    return sorted(items, key=lambda x: (hashlib.sha256(f"{seed}:{x}".encode()).hexdigest(), x))


def _split_track_keys(manifest: dict[str, Any], split: str, table) -> list[str]:
    keys = {str(r.get("query_track_key")) for r in manifest.get("records", []) if r.get("split") == split}
    for r in manifest.get("records", []):
        if r.get("split") != split:
            continue
        keys.update(str(x) for x in r.get("support_track_keys", []))
        if r.get("hard_negative_track_key") is not None:
            keys.add(str(r["hard_negative_track_key"]))
    return sorted(k for k in keys if k in table.metadata)


def build_banks(manifest_path: Path, fold: int, split: str, table, *, seed: int = 7600) -> list[CandidateBank]:
    """Build <=3 positives + top12 raw hard negatives per query.

    Category/video values are used only to construct TRAIN metadata and labels;
    they are not passed to the relation network.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_keys = _split_track_keys(manifest, split, table)
    by_query: dict[str, dict[str, Any]] = {}
    for row in manifest.get("records", []):
        if row.get("split") != split or row.get("kind") != "multi_positive_cross_video":
            continue
        q = str(row.get("query_track_key"))
        if q not in table.metadata:
            continue
        bucket = by_query.setdefault(q, {"episodes": [], "pos": []})
        bucket["episodes"].append(str(row.get("episode_id", q)))
        bucket["pos"].extend(str(x) for x in row.get("support_track_keys", []))
    banks: list[CandidateBank] = []
    for q in sorted(by_query):
        qm = table.metadata[q]
        positives = []
        for p in by_query[q]["pos"]:
            if p not in table.metadata or p == q:
                continue
            pm = table.metadata[p]
            if pm["category"] == qm["category"] and pm["video"] != qm["video"] and p not in positives:
                positives.append(p)
        positives = _stable_order(positives, seed + fold)[:3]
        if not positives:
            continue
        neg_pool = []
        for c in all_keys:
            if c == q or c in positives:
                continue
            cm = table.metadata[c]
            if cm["category"] == qm["category"] or cm["video"] == qm["video"]:
                continue
            score = float(table.raw_vector(q, 16) @ table.raw_vector(c, 16))
            neg_pool.append((score, c))
        neg_pool.sort(key=lambda x: (-x[0], x[1]))
        negatives = [c for _, c in neg_pool[:12]]
        candidates = tuple(positives + negatives)
        if len(candidates) < 2 or not negatives:
            continue
        raw_scores = tuple(float(table.raw_vector(q, 16) @ table.raw_vector(c, 16)) for c in candidates)
        eid = by_query[q]["episodes"][0]
        banks.append(CandidateBank(fold, split, eid, q, candidates, tuple(positives), tuple(negatives), raw_scores, int(qm["category"]), int(qm["video"])))
    return banks


def banks_hash(banks: list[CandidateBank]) -> str:
    payload = [asdict(x) for x in banks]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_banks(path: Path, banks: list[CandidateBank], *, manifest_sha256: str, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "Phase76A", "split": banks[0].split if banks else None,
        "fold": banks[0].fold if banks else None, "seed": seed,
        "candidate_rule": "same-category/different-video positives <=3; different-category/different-video raw top12 hard negatives; max15",
        "banks_hash": banks_hash(banks), "manifest_sha256": manifest_sha256,
        "banks": [asdict(x) for x in banks],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_banks(path: Path) -> list[CandidateBank]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CandidateBank(**x) for x in payload["banks"]]

