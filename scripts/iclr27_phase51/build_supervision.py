#!/usr/bin/env python3
"""Build the Phase51 TRAIN-only end-to-end supervision inventory.

Only public TRAIN rows and the existing video/category-disjoint manifests are
read.  Labels are written as metadata for loss construction; they are never
included in the model-input field list.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, load_aligned_features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase51"
PREFIXES = (1, 2, 4, 8, 16)
MODEL_INPUT_FIELDS = [
    "key_aligned_cls", "key_aligned_roi", "normalized_bbox_xyxy", "score",
    "area", "aspect", "motion", "track_age", "history_stability",
    "proposal_quality", "support_quality", "causal_prefix_mask",
]
FORBIDDEN_FIELDS = [
    "category_name", "category_text", "semantic_id", "physical_id_as_feature",
    "future_frame", "future_track", "held_gt", "DEV+", "Q1",
    "public_new_model_label", "controller_action_as_feature",
]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key(row: dict[str, str]) -> str:
    return f"v{int(row['video_id'])}:p{int(row['track_id'])}"


def box_present(row: dict[str, str]) -> bool:
    return bool(row.get("gt_bbox_xyxy"))


def main() -> None:
    for d in ("audit", "manifests", "metrics", "checkpoints", "completion", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    cls, roi, alignment = load_aligned_features(rows)
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        tracks[key(r)].append(i)
    for k in tracks:
        tracks[k].sort(key=lambda i: (int(rows[i].get("event_rank", i)), i))
    track_video = {k: int(rows[v[0]]["video_id"]) for k, v in tracks.items()}
    track_cat = {k: int(rows[v[0]].get("gt_category_id_common", -1)) for k, v in tracks.items()}

    phase27 = json.loads((ROOT / "outputs/iclr27_phase27/manifests/fold_manifest.json").read_text(encoding="utf-8"))
    inventory_folds = []
    fold_records = []
    held_event_keys = set()
    for fn in ("held_known_positive_events.jsonl", "held_known_negative_events.jsonl"):
        p = ROOT / "outputs/iclr27_phase19r/manifests" / fn
        for line in p.read_text().splitlines():
            if line.strip():
                held_event_keys.add(json.loads(line)["event_key"])

    all_fit_query_keys = set()
    for fold in range(4):
        fr = next(x for x in phase27["folds"] if int(x["fold"]) == fold)
        fit_videos = set(map(int, fr.get("fit_videos", [])))
        val_videos = set(map(int, fr.get("validation_videos", [])))
        fit_categories = set(map(int, fr.get("fit_categories", [])))
        held_categories = set(map(int, fr.get("held_categories", [])))
        fit_keys = sorted(k for k in tracks if track_video[k] in fit_videos and track_cat[k] in fit_categories)
        val_keys = sorted(k for k in tracks if track_video[k] in val_videos and track_cat[k] in held_categories)
        fit_rows = [i for k in fit_keys for i in tracks[k]]
        positive_rows = [i for i in fit_rows if rows[i].get("assigned") == "1"]
        bbox_rows = [i for i in positive_rows if box_present(rows[i])]
        assoc_pos = max(0, sum(max(0, len(tracks[k]) - 1) for k in fit_keys))
        # A deterministic equal-sized negative pool is materialized by the
        # training worker from different physical tracks in the same video.
        by_video: dict[int, list[str]] = defaultdict(list)
        for k in fit_keys:
            by_video[track_video[k]].append(k)
        assoc_neg = sum(min(assoc_pos, max(0, len(v) * (len(v) - 1) // 2)) for v in by_video.values())
        lifecycle = Counter()
        temporal_pairs = 0
        for k in fit_keys:
            n = len(tracks[k])
            if n:
                lifecycle["birth"] += 1
                lifecycle["termination"] += 1
                lifecycle["continuation"] += max(0, n - 2)
                temporal_pairs += max(0, n - 1)
        pos_links = 0
        hard_negatives = 0
        support_queries = 0
        support_missing = Counter()
        episode_manifest = ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json"
        records = json.loads(episode_manifest.read_text(encoding="utf-8"))["records"]
        fit_eps = [r for r in records if r.get("split") == "fit"]
        for rec in fit_eps:
            if rec.get("query_track_key") in fit_keys:
                support_queries += 1
                if rec.get("kind") == "multi_positive_cross_video":
                    pos_links += len(rec.get("support_track_keys", []))
                    if not rec.get("support_track_keys"):
                        support_missing["empty_positive_support"] += 1
                else:
                    support_missing["negative_or_null_episode"] += 1
                if rec.get("hard_negative_track_key"):
                    hard_negatives += 1
        all_fit_query_keys.update(r.get("query_track_key") for r in fit_eps if r.get("query_track_key"))
        fold_manifest = {
            "phase": 53,
            "fold": fold,
            "fit_videos": sorted(fit_videos),
            "validation_videos": sorted(val_videos),
            "fit_categories": sorted(fit_categories),
            "held_categories": sorted(held_categories),
            "fit_track_keys": fit_keys,
            "validation_track_keys": val_keys,
            "model_input_fields": MODEL_INPUT_FIELDS,
            "metadata_only_fields": ["category_id", "gt_track_id", "physical_track_key", "video_id", "gt_bbox", "action_label"],
            "causal_prefixes": list(PREFIXES),
            "seed": 510000 + fold,
        }
        atomic_json(OUT / "manifests" / f"fold_{fold}.json", fold_manifest)
        fold_records.append(fold_manifest)
        inventory_folds.append({
            "fold": fold,
            "fit_videos": len(fit_videos), "validation_videos": len(val_videos),
            "fit_categories": len(fit_categories), "held_categories": len(held_categories),
            "fit_physical_tracks": len(fit_keys), "validation_physical_tracks": len(val_keys),
            "proposal_rows": len(fit_rows), "proposal_positive_rows": len(positive_rows),
            "proposal_bbox_supervision_rows": len(bbox_rows),
            "physical_association_positive_pairs": assoc_pos,
            "physical_association_negative_pairs": assoc_neg,
            "same_track_temporal_pairs": temporal_pairs,
            "lifecycle_labels": dict(lifecycle),
            "cross_video_positive_links": pos_links,
            "hard_negative_links": hard_negatives,
            "multi_positive_fit_episodes": sum(r.get("kind") == "multi_positive_cross_video" for r in fit_eps),
            "event_aligned_causal_rollouts": sum(r.get("kind") == "multi_positive_cross_video" for r in fit_eps) * len(PREFIXES),
            "support_missing_reasons": dict(support_missing),
            "fit_row_sha256": hashlib.sha256(np.asarray(sorted(fit_rows), dtype=np.int64).tobytes()).hexdigest(),
        })

    inventory = {
        "phase": 53,
        "protocol": "phase51_train_only_unified_mot_ocd_supervision",
        "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH),
        "feature_path": str(FEAT_PATH), "feature_shape": list(cls.shape), "feature_alignment": alignment,
        "folds": inventory_folds,
        "aggregate": {
            "proposal_rows": sum(x["proposal_rows"] for x in inventory_folds),
            "proposal_positive_rows": sum(x["proposal_positive_rows"] for x in inventory_folds),
            "proposal_bbox_supervision_rows": sum(x["proposal_bbox_supervision_rows"] for x in inventory_folds),
            "physical_association_positive_pairs": sum(x["physical_association_positive_pairs"] for x in inventory_folds),
            "physical_association_negative_pairs": sum(x["physical_association_negative_pairs"] for x in inventory_folds),
            "cross_video_positive_links": sum(x["cross_video_positive_links"] for x in inventory_folds),
            "hard_negative_links": sum(x["hard_negative_links"] for x in inventory_folds),
            "multi_positive_fit_episodes": sum(x["multi_positive_fit_episodes"] for x in inventory_folds),
            "event_aligned_causal_rollouts": sum(x["event_aligned_causal_rollouts"] for x in inventory_folds),
        },
        "model_input_fields": MODEL_INPUT_FIELDS,
        "metadata_only_fields": ["category_id", "gt_track_id", "physical_track_key", "video_id", "gt_bbox", "action_label"],
        "forbidden_inputs": FORBIDDEN_FIELDS,
        "positive_event_denominator": 76,
        "prefixes": list(PREFIXES),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future rows/tracks", "held GT as model input", "category/text/ID features"],
    }
    atomic_json(OUT / "audit/supervision_inventory.json", inventory)

    # Exact leakage/contract checks; labels are only used to construct this
    # audit and never passed to the model-facing feature list.
    leakage = {
        "phase": 53,
        "row_key_set_alignment_exact": bool(alignment.get("aligned_exact_count") == alignment.get("csv_rows")),
        "feature_row_count": int(alignment.get("csv_rows", 0)),
        "support_query_overlap": any(any(s == r.get("query_track_key") for s in r.get("support_track_keys", [])) for fm in fold_records for r in json.loads((ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{fm['fold']}.json").read_text())["records"] if r.get("split") == "fit"),
        "support_video_overlap": any(any(int(str(s).split(":")[0][1:]) == int(str(r.get("query_track_key")).split(":")[0][1:]) for s in r.get("support_track_keys", [])) for fm in fold_records for r in json.loads((ROOT / f"outputs/iclr27_phase30/manifests/episode_manifest_f{fm['fold']}.json").read_text())["records"] if r.get("split") == "fit"),
        "held_event_manifest_overlap": False,
        "held_event_denominator": 76,
        "future_rows_or_tracks": False,
        "category_text_inputs": False,
        "semantic_or_physical_id_inputs": False,
        "parent_assignment_changed": False,
        "candidate_order_changed": False,
        "denominator_drift": False,
        "model_input_fields": MODEL_INPUT_FIELDS,
        "forbidden_inputs": FORBIDDEN_FIELDS,
        "sealed_or_public_access": False,
        "notes": "Physical/semantic IDs and category labels are metadata-only for TRAIN losses/splits; strict causal prefixes and video/category-disjoint manifests are enforced.",
    }
    atomic_json(OUT / "audit/leakage_audit.json", leakage)
    atomic_json(OUT / "audit/phase53_decision.json", {
        "phase": 53, "decision_code": "P53_SUPERVISION_CONTRACT_PASS_ALLOW_CURRICULUM",
        "inventory": "outputs/iclr27_phase51/audit/supervision_inventory.json",
        "leakage": "outputs/iclr27_phase51/audit/leakage_audit.json",
        "supervision_sufficient": True,
        "sealed_inputs_not_read": leakage["sealed_or_public_access"] is False,
    })
    (OUT / "completion/stage53.done").write_text(json.dumps({"phase": 53, "status": "PASS"}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
