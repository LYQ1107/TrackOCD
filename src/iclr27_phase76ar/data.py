"""TRAIN-only stream construction for Phase76AR.

Two concrete objects are exposed: ``MemoryMimicBank`` is rebuilt from the
fold-local track table and prefix-union hard negatives; ``LegalFitEpisode`` is
read directly from the frozen Phase30 fit manifest.  The two loaders are kept
separate so an odd/even training step cannot silently become a single-stream
run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PREFIXES = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class MemoryMimicBank:
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
    source: str = "memory_mimic"


@dataclass(frozen=True)
class LegalFitEpisode:
    fold: int
    split: str
    episode_id: str
    query_key: str
    positive_keys: tuple[str, ...]
    negative_keys: tuple[str, ...]
    category: int
    video: int
    source: str = "legal_fit"

    @property
    def candidates(self) -> tuple[str, ...]:
        return self.positive_keys + self.negative_keys


def _stable_hash(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def _stable_order(keys: Iterable[str], seed: int) -> list[str]:
    return sorted((str(x) for x in keys), key=lambda x: (_stable_hash(seed, x), x))


def _manifest_records(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in payload.get("records", []) if r.get("split") == split]


def _split_keys(records: list[dict[str, Any]], table) -> list[str]:
    keys: set[str] = set()
    for row in records:
        q = row.get("query_track_key")
        if q is not None and str(q) in table.metadata:
            keys.add(str(q))
        for key in row.get("support_track_keys", []):
            if str(key) in table.metadata:
                keys.add(str(key))
        key = row.get("hard_negative_track_key")
        if key is not None and str(key) in table.metadata:
            keys.add(str(key))
    return sorted(keys)


def _positive_map(records: list[dict[str, Any]], table) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for row in records:
        if row.get("kind") != "multi_positive_cross_video":
            continue
        q = str(row.get("query_track_key"))
        if q not in table.metadata:
            continue
        bucket = out.setdefault(q, [])
        for key in row.get("support_track_keys", []):
            key = str(key)
            if key not in table.metadata or key == q:
                continue
            qm, cm = table.metadata[q], table.metadata[key]
            if qm["category"] == cm["category"] and qm["video"] != cm["video"] and key not in bucket:
                bucket.append(key)
    return {q: tuple(v) for q, v in out.items()}


def _prefix_union_negatives(query: str, positives: set[str], keys: list[str], table, fold: int) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    qm = table.metadata[query]
    seen: dict[str, list[int]] = {}
    best: dict[str, float] = {}
    for prefix in PREFIXES:
        qv = table.raw_vector(query, prefix)
        scored: list[tuple[float, str]] = []
        for key in keys:
            if key == query or key in positives:
                continue
            cm = table.metadata[key]
            if cm["category"] == qm["category"] or cm["video"] == qm["video"]:
                continue
            score = float(qv @ table.raw_vector(key, prefix))
            scored.append((score, key))
            old = best.get(key, -1e9)
            if score > old:
                best[key] = score
        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, key in scored[:4]:
            seen.setdefault(key, []).append(prefix)
    ordered = sorted(seen, key=lambda key: (-len(seen[key]), -best.get(key, -1e9), key))[:12]
    prov = {key: tuple(sorted(seen[key])) for key in ordered}
    return tuple(ordered), prov


def build_memory_mimic(manifest_path: Path, fold: int, split: str, table, *, seed: int = 7600) -> list[MemoryMimicBank]:
    records = _manifest_records(manifest_path, split)
    keys = _split_keys(records, table)
    positives = _positive_map(records, table)
    banks: list[MemoryMimicBank] = []
    for query in sorted(positives):
        p = tuple(_stable_order(positives[query], seed + fold)[:3])
        if not p:
            continue
        neg, provenance = _prefix_union_negatives(query, set(p), keys, table, fold)
        if not neg:
            continue
        meta = table.metadata[query]
        banks.append(MemoryMimicBank(
            fold=fold, split=split, episode_id=f"memory-{fold}-{split}-{query}",
            query_key=query, candidates=p + neg, positives=p, negatives=neg,
            category=int(meta["category"]), video=int(meta["video"]), negative_provenance=provenance,
        ))
    return banks


def build_legal_fit(manifest_path: Path, fold: int, split: str, table) -> list[LegalFitEpisode]:
    records = _manifest_records(manifest_path, split)
    out: list[LegalFitEpisode] = []
    for row in records:
        if row.get("kind") != "multi_positive_cross_video":
            continue
        q = str(row.get("query_track_key"))
        if q not in table.metadata:
            continue
        pos: list[str] = []
        for key in row.get("support_track_keys", []):
            key = str(key)
            if key not in table.metadata or key == q or key in pos:
                continue
            qm, cm = table.metadata[q], table.metadata[key]
            if qm["category"] == cm["category"] and qm["video"] != cm["video"]:
                pos.append(key)
        neg = []
        key = row.get("hard_negative_track_key")
        if key is not None and str(key) in table.metadata and str(key) not in pos:
            neg.append(str(key))
        if pos and neg:
            meta = table.metadata[q]
            out.append(LegalFitEpisode(
                fold=fold, split=split, episode_id=str(row.get("episode_id", q)),
                query_key=q, positive_keys=tuple(pos), negative_keys=tuple(neg),
                category=int(meta["category"]), video=int(meta["video"]),
            ))
    return out


def bank_payload(banks: Iterable[MemoryMimicBank], legal: Iterable[LegalFitEpisode], *, manifest_sha256: str, fold: int, split: str) -> dict[str, Any]:
    memory = [asdict(x) for x in banks]
    legal_rows = [asdict(x) for x in legal]
    payload = {
        "phase": "Phase76AR", "fold": fold, "split": split,
        "manifest_sha256": manifest_sha256,
        "prefixes": list(PREFIXES), "memory_mimic": memory, "legal_fit": legal_rows,
        "stream_hashes": {
            "memory_mimic": hashlib.sha256(json.dumps(memory, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "legal_fit": hashlib.sha256(json.dumps(legal_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        },
        "stream_counts": {"memory_mimic": len(memory), "legal_fit": len(legal_rows)},
        "forbidden_inference_inputs": ["category", "semantic_id", "physical_id", "text", "future", "held/DEV+/Q1/public-new/sealed labels"],
    }
    return payload


def _from_memory(row: dict[str, Any]) -> MemoryMimicBank:
    row = dict(row); row["candidates"] = tuple(row["candidates"]); row["positives"] = tuple(row["positives"]); row["negatives"] = tuple(row["negatives"])
    row["negative_provenance"] = {str(k): tuple(v) for k, v in row.get("negative_provenance", {}).items()}
    return MemoryMimicBank(**row)


def _from_legal(row: dict[str, Any]) -> LegalFitEpisode:
    row = dict(row); row["positive_keys"] = tuple(row["positive_keys"]); row["negative_keys"] = tuple(row["negative_keys"])
    return LegalFitEpisode(**row)


def load_stream_payload(path: Path, split: str = "fit") -> tuple[list[MemoryMimicBank], list[LegalFitEpisode]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # A fold file stores independent fit and validation stream objects.  The
    # optional split argument makes the boundary explicit and prevents a
    # caller from accidentally mixing validation records into training.
    section = payload.get(split, payload)
    return ([_from_memory(x) for x in section.get("memory_mimic", [])], [_from_legal(x) for x in section.get("legal_fit", [])])


def stream_hashes(memory: Iterable[MemoryMimicBank], legal: Iterable[LegalFitEpisode]) -> dict[str, str]:
    a = [asdict(x) for x in memory]; b = [asdict(x) for x in legal]
    return {
        "memory_mimic": hashlib.sha256(json.dumps(a, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest(),
        "legal_fit": hashlib.sha256(json.dumps(b, sort_keys=True, default=list, separators=(",", ":")).encode()).hexdigest(),
    }
