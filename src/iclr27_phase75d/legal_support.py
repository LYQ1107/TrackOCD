"""Legal support episodes projected from the frozen Phase30 manifests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LegalSupportEpisode:
    fold: int
    episode_id: str
    query_key: str
    positive_support_keys: tuple[str, ...]
    negative_support_keys: tuple[str, ...]
    kind: str = "multi_positive_cross_video"


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_legal_episodes(root: Path, fold: int, known_keys: set[str] | None = None) -> tuple[list[LegalSupportEpisode], list[dict[str, Any]], dict[str, Any]]:
    path = root / f"episode_manifest_f{fold}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["records"]:
        if row.get("split") != "val":
            continue
        by_query.setdefault(str(row["query_track_key"]), []).append(row)
    episodes: list[LegalSupportEpisode] = []
    unevaluable: list[dict[str, Any]] = []
    for row in manifest["records"]:
        if row.get("split") != "val" or row.get("kind") != "multi_positive_cross_video":
            continue
        q = str(row["query_track_key"])
        pos = tuple(dict.fromkeys(str(x) for x in row.get("support_track_keys", [])))
        negatives: list[str] = []
        hard = row.get("hard_negative_track_key")
        if hard is not None:
            negatives.append(str(hard))
        # The paired null record is explicit in the frozen manifest.  Do not
        # search by category or cosine and do not invent any candidate.
        for mate in by_query.get(q, []):
            if mate.get("kind") == "null_no_match_hard_negative":
                negatives.extend(str(x) for x in mate.get("support_track_keys", []))
                if mate.get("hard_negative_track_key") is not None:
                    negatives.append(str(mate["hard_negative_track_key"]))
        neg = tuple(x for x in dict.fromkeys(negatives) if x not in pos and x != q)
        missing = []
        if known_keys is not None:
            for x in (q, *pos, *neg):
                if x not in known_keys:
                    missing.append(x)
        if not pos or not neg or missing:
            unevaluable.append({"fold": fold, "episode_id": row.get("episode_id"), "query_key": q, "positive_count": len(pos), "negative_count": len(neg), "missing_keys": missing, "reason": "NOT_EVALUABLE_MISSING_LEGAL_CANDIDATE"})
            continue
        episodes.append(LegalSupportEpisode(fold=fold, episode_id=str(row["episode_id"]), query_key=q, positive_support_keys=pos, negative_support_keys=neg))
    summary = {
        "fold": fold, "manifest": str(path.resolve()), "manifest_sha256": manifest_sha256(path),
        "validation_records": sum(r.get("split") == "val" for r in manifest["records"]),
        "evaluable_episodes": len(episodes), "unevaluable_episodes": len(unevaluable),
        "support_visibility": "source support prefix16; query causal prefix p",
        "candidate_construction": "manifest-explicit support_track_keys and hard_negative_track_key only",
    }
    return episodes, unevaluable, summary


def episode_bank_hash(episodes: list[LegalSupportEpisode]) -> str:
    payload = [asdict(x) for x in episodes]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
