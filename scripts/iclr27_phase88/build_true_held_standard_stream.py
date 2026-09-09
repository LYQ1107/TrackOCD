#!/usr/bin/env python3
"""Build a causal OLD/NEW held stream manifest for reporting only.

Categories are evaluator-side metadata.  Track tensors are loaded later from
the same causal feature store and never receive these values.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase88"
MEMMAP = "/data2/usr_for_deadline/trackocd_phase88/shared_features"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
sys.path.insert(0, str(ROOT))
from src.iclr27_phase88.data_memmap import Phase88FoldData  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main() -> None:
    pos = load_jsonl(POS); neg = load_jsonl(NEG)
    if len(pos) != 76 or len(neg) != 76:
        raise RuntimeError(f"held denominator changed: {len(pos)} {len(neg)}")
    target_category = {str(r["target_tracklet_key"]): int(r["category_gt_denominator_only"]) for r in pos}
    folds = []
    conflicts = []
    for fold in range(4):
        data = Phase88FoldData(fold, MEMMAP)
        tracks: dict[str, dict] = {}
        known_keys = {str(k) for k, _ in data.known_eval_keys}
        for k, cat in sorted(data.known_eval_keys, key=lambda x: str(x[0])):
            key = str(k)
            if key in target_category:
                conflicts.append({"fold": fold, "track_key": key, "reason": "known_anchor_overlaps_held_target"})
                continue
            tracks[key] = {"track_key": key, "video_id": int(data.video(key)), "category_eval": int(cat),
                           "role": "old", "order": -1, "category_inference_input": False}
        for event_index, event in enumerate(pos + neg):
            if int(event["fold"]) != fold:
                continue
            is_pos = event.get("kind") == "positive_existing"
            source_category = int(event["category_gt_denominator_only"] if is_pos else event["distractor_category_gt_denominator_only"])
            for source_index, key_raw in enumerate(event["source_tracklet_keys"]):
                key = str(key_raw)
                item = {"track_key": key, "video_id": int(event["source_video"]), "category_eval": source_category,
                        "role": "new_source", "order": event_index * 2, "category_inference_input": False}
                if key in tracks and (tracks[key].get("category_eval"), tracks[key].get("video_id")) != (item["category_eval"], item["video_id"]):
                    conflicts.append({"fold": fold, "track_key": key, "reason": "source_metadata_conflict", "old": tracks[key], "new": item})
                else:
                    tracks.setdefault(key, item)
                    tracks[key]["order"] = min(int(tracks[key]["order"]), int(item["order"]))
            target_key = str(event["target_tracklet_key"])
            if target_key not in target_category:
                raise RuntimeError(f"missing positive target category for {target_key}")
            item = {"track_key": target_key, "video_id": int(event["target_video"]), "category_eval": int(target_category[target_key]),
                    "role": "new_target", "order": event_index * 2 + 1, "category_inference_input": False}
            if target_key in tracks and tracks[target_key].get("role") == "old":
                conflicts.append({"fold": fold, "track_key": target_key, "reason": "target_old_overlap", "old": tracks[target_key], "new": item})
                continue
            if target_key in tracks and (tracks[target_key].get("category_eval"), tracks[target_key].get("video_id")) != (item["category_eval"], item["video_id"]):
                conflicts.append({"fold": fold, "track_key": target_key, "reason": "target_metadata_conflict", "old": tracks[target_key], "new": item})
            else:
                tracks.setdefault(target_key, item)
                tracks[target_key]["order"] = min(int(tracks[target_key]["order"]), int(item["order"]))
        rows = []
        for idx, item in enumerate(sorted(tracks.values(), key=lambda x: (0 if x["role"] == "old" else 1, int(x["order"]), str(x["track_key"])))):
            row = dict(item)
            row["stream_index"] = idx
            row["causal_order_key"] = [int(row["order"]), str(row["track_key"])]
            rows.append(row)
        folds.append({"fold": fold, "track_count": len(rows), "old_track_count": sum(r["role"] == "old" for r in rows),
                      "new_track_count": sum(r["role"] != "old" for r in rows), "tracks": rows,
                      "state_memory_reset_between_tracks": False,
                      "ordering": "old validation anchors, then held source-before-target partial order, opaque stable key tie-break"})
    payload = {
        "schema_version": "trackocd.phase88.held_standard_openworld_stream.v1", "phase": 88,
        "name": "held_standard_openworld_stream_v1", "stream_kind": "TRUE_HELD_OLD_NEW",
        "deterministic": True, "causal": True, "support_mode": False,
        "old_definition": "TRAIN-disjoint validation known anchors only",
        "new_definition": "held positive/negative source and target physical tracks, each deduplicated per fold",
        "category_eval_only": True, "category_text_used_for_order": False,
        "future_rows_or_tracks": False, "state_memory_reset_between_tracks": False,
        "held_manifest_hashes": {"positive": sha(POS), "negative": sha(NEG)},
        "source_manifest_paths": [str(POS.resolve()), str(NEG.resolve())],
        "conflicts": conflicts, "folds": folds,
        "public_dev_q1_sealed_accessed": False, "used_for_selection": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    out = OUT / "manifests/held_standard_openworld_stream_v1.json"
    atomic_json(out, payload)
    print(json.dumps({"path": str(out.resolve()), "fold_track_counts": [x["track_count"] for x in folds], "conflicts": len(conflicts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
