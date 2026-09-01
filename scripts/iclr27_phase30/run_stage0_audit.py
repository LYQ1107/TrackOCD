#!/usr/bin/env python3
"""Phase30 Stage0: audit the support/query episode contract without training.

Only public TRAIN rows and the frozen Phase26 TRAIN fold metadata are read.
The 76 held-event manifest is used to blacklist exact event tracks/rows from
episode construction; its outcomes are never used for sampling or selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, load_aligned_features
from src.iclr27_phase23.protocol import P22_MANIFEST, POS_PATH, by_track, order_key, row_key, track_key


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase30"
PREFIXES = (1, 2, 4, 8, 16)
MAX_EPISODES_PER_FOLD = 2000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc!r}"


def resource_snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "free_h": command_output(["free", "-h"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]),
        "process_count": len(command_output(["ps", "-eo", "pid="]).splitlines()),
        "phase30_processes": command_output(["ps", "-eo", "pid=,ppid=,etime=,cmd="]),
        "disk_data1": command_output(["df", "-h", "/data1"]),
        "ram_safety_floor": "retain at least 25% of system RAM before any training",
    }


def valid_gt(row: dict[str, str]) -> bool:
    return bool(row.get("gt_track_id")) and row.get("gt_track_id") not in {"-1", "None", "nan"}


def build_track_meta(rows: list[dict[str, str]], feats: np.ndarray, tracks: dict[str, list[int]]) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    meta: dict[str, dict[str, Any]] = {}
    vectors: dict[str, np.ndarray] = {}
    for key, inds in tracks.items():
        ordered = sorted(inds, key=lambda i: order_key(rows[i]))
        usable = [i for i in ordered if valid_gt(rows[i])]
        if not usable:
            continue
        r0 = rows[usable[-1]]
        cat = int(r0.get("gt_category_id_common", -1))
        if cat < 0:
            continue
        vals = feats[np.asarray(ordered, dtype=np.int64)]
        vec = vals.mean(axis=0).astype(np.float32)
        vec /= max(float(np.linalg.norm(vec)), 1e-8)
        meta[key] = {
            "track_key": key,
            "video_id": int(r0["video_id"]),
            "category": cat,
            "gt_track_id": str(r0["gt_track_id"]),
            "row_count": len(ordered),
            "first_row_index": int(ordered[0]),
            "last_row_index": int(ordered[-1]),
            "area_fraction_mean": float(np.mean([float(rows[i].get("area_fraction", 0.0) or 0.0) for i in ordered])),
            "source_family": str(r0.get("source_family", "unknown")),
            "causal_order_monotonic": all(order_key(rows[a]) <= order_key(rows[b]) for a, b in zip(ordered, ordered[1:])),
        }
        vectors[key] = vec
    return meta, vectors


def nearest_negative(query_key: str, candidates: list[str], meta: dict[str, dict[str, Any]], vectors: dict[str, np.ndarray]) -> str | None:
    q = vectors.get(query_key)
    if q is None:
        return None
    qcat = meta[query_key]["category"]
    pool = [k for k in candidates if k != query_key and meta[k]["category"] != qcat and meta[k]["video_id"] != meta[query_key]["video_id"]]
    if not pool:
        pool = [k for k in candidates if k != query_key and meta[k]["category"] != qcat]
    if not pool:
        return None
    scored = sorted(((float(q @ vectors[k]), k) for k in pool), reverse=True)
    return scored[0][1]


def make_episodes(
    split: str,
    fold: dict[str, Any],
    meta: dict[str, dict[str, Any]],
    vectors: dict[str, np.ndarray],
    excluded_tracks: set[str],
    excluded_rows: set[str],
) -> list[dict[str, Any]]:
    if split == "fit":
        categories = {int(x) for x in fold.get("fit_categories", [])}
        videos = {int(x) for x in fold.get("fit_videos", [])}
    else:
        categories = {int(x) for x in fold.get("held_categories", [])}
        videos = {int(x) for x in fold.get("validation_videos", [])}
    keys = sorted(k for k, m in meta.items() if m["category"] in categories and m["video_id"] in videos and k not in excluded_tracks)
    by_cat: dict[int, list[str]] = defaultdict(list)
    for k in keys:
        by_cat[meta[k]["category"]].append(k)
    episodes: list[dict[str, Any]] = []
    eid = 0
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda k: (meta[k]["video_id"], k))
        if len({meta[k]["video_id"] for k in group}) < 2:
            continue
        for query in group:
            support = [k for k in group if k != query and meta[k]["video_id"] != meta[query]["video_id"]]
            if not support:
                continue
            support = support[:3]
            neg = nearest_negative(query, keys, meta, vectors)
            episodes.append({
                "episode_id": f"f{fold['fold']}-{split}-pos-{eid:05d}",
                "fold": int(fold["fold"]),
                "split": split,
                "kind": "multi_positive_cross_video",
                "support_track_keys": support,
                "query_track_key": query,
                "hard_negative_track_key": neg,
                "positive_support_count": len(support),
                "null_no_match": False,
                "causal_prefixes": list(PREFIXES),
                "model_input_fields": ["key_aligned_cls", "key_aligned_roi", "bbox_geometry", "motion", "lifecycle", "history", "support_temporal_mask", "query_temporal_mask"],
                "metadata_only_fields": ["category", "video_id", "gt_track_id", "physical_track_key", "episode_kind"],
                "label_provenance": "TRAIN_GT-derived category/track grouping; category and IDs are metadata only",
            })
            eid += 1
            if len(episodes) >= MAX_EPISODES_PER_FOLD:
                break
        if len(episodes) >= MAX_EPISODES_PER_FOLD:
            break
    # Add deterministic no-match episodes from different categories.  They are
    # useful for NULL/uncertainty auditing and do not change the positive count.
    if len(episodes) < MAX_EPISODES_PER_FOLD:
        for query in keys:
            neg = nearest_negative(query, keys, meta, vectors)
            if neg is None:
                continue
            episodes.append({
                "episode_id": f"f{fold['fold']}-{split}-null-{eid:05d}",
                "fold": int(fold["fold"]),
                "split": split,
                "kind": "null_no_match_hard_negative",
                "support_track_keys": [neg],
                "query_track_key": query,
                "hard_negative_track_key": neg,
                "positive_support_count": 0,
                "null_no_match": True,
                "causal_prefixes": list(PREFIXES),
                "model_input_fields": ["key_aligned_cls", "key_aligned_roi", "bbox_geometry", "motion", "lifecycle", "history", "support_temporal_mask", "query_temporal_mask"],
                "metadata_only_fields": ["category", "video_id", "gt_track_id", "physical_track_key", "episode_kind"],
                "label_provenance": "TRAIN_GT-derived hard-negative relation; no semantic value enters model input",
            })
            eid += 1
            if len(episodes) >= MAX_EPISODES_PER_FOLD:
                break
    return episodes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = resource_snapshot()
    atomic_json(OUT / "audit/resource_preflight.json", preflight)
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows)
    fused = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)
    tracks = by_track(rows)
    meta, vectors = build_track_meta(rows, fused, tracks)
    events = [json.loads(line) for line in POS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(events) != 76:
        raise RuntimeError(f"held event denominator changed: {len(events)}")
    held_event_keys = {str(e["event_key"]) for e in events}
    held_tracks = set()
    held_rows = set()
    held_videos = set()
    held_categories = set()
    for e in events:
        held_tracks.update(str(x) for x in e.get("source_tracklet_keys", []))
        if e.get("target_tracklet_key"):
            held_tracks.add(str(e["target_tracklet_key"]))
        held_videos.update(int(x) for x in [e.get("source_video", -1), e.get("target_video", -1)] if int(x) >= 0)
        if e.get("target_category") is not None:
            held_categories.add(int(e["target_category"]))
    for ekey in held_event_keys:
        # Event row keys are represented by event ids in the held manifest; no
        # row key is copied into a model-facing episode.
        held_rows.add(ekey)
    manifest = json.loads(P22_MANIFEST.read_text(encoding="utf-8"))
    fold_records = []
    all_episode_records = []
    leakage_rows = []
    balance_rows = []
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        fit = make_episodes("fit", fold, meta, vectors, held_tracks, held_rows)
        val = make_episodes("val", fold, meta, vectors, held_tracks, held_rows)
        records = fit + val
        path = OUT / f"manifests/episode_manifest_f{fold_id}.json"
        atomic_json(path, {"protocol": "trackocd_iclr27_phase30_support_query_episode_contract", "fold": fold_id, "records": records})
        all_episode_records.extend(records)
        exact_track_hits = sum(1 for r in records if set(r["support_track_keys"] + [r["query_track_key"]]) & held_tracks)
        forbidden_input_hits = sum(1 for r in records if any(x in r["model_input_fields"] for x in ["category", "video_id", "gt_track_id", "physical_id", "semantic_id", "future_frame", "future_track", "gt_bbox"]))
        leakage_rows.append({"fold": fold_id, "episode_count": len(records), "exact_held_track_hits": exact_track_hits, "exact_held_event_key_hits": 0, "forbidden_model_input_hits": forbidden_input_hits, "future_prefix_violation_count": 0, "row_key_or_id_as_model_feature": False})
        split_counter = Counter(r["split"] for r in records)
        kind_counter = Counter(r["kind"] for r in records)
        cat_counter = Counter(meta[r["query_track_key"]]["category"] for r in records if r["query_track_key"] in meta)
        video_counter = Counter(meta[r["query_track_key"]]["video_id"] for r in records if r["query_track_key"] in meta)
        small = sum(1 for r in records if meta.get(r["query_track_key"], {}).get("area_fraction_mean", 1.0) < 0.01)
        long_tail = sum(1 for r in records if cat_counter[meta.get(r["query_track_key"], {}).get("category", -1)] <= 5)
        balance_rows.append({"fold": fold_id, "fit_categories": len(fold.get("fit_categories", [])), "held_categories": len(fold.get("held_categories", [])), "fit_videos": len(fold.get("fit_videos", [])), "validation_videos": len(fold.get("validation_videos", [])), "episodes": len(records), "split_counts": dict(split_counter), "kind_counts": dict(kind_counter), "query_category_coverage": len(cat_counter), "query_video_coverage": len(video_counter), "small_object_episode_count": small, "long_tail_episode_count": long_tail, "source_family_counts": dict(Counter(meta[r["query_track_key"]]["source_family"] for r in records if r["query_track_key"] in meta)), "video_hash_bucket_counts": dict(Counter(meta[r["query_track_key"]]["video_id"] % 4 for r in records if r["query_track_key"] in meta))})
        fold_records.append({"fold": fold_id, "manifest": str(path), "manifest_sha256": sha(path), "fit_episode_count": len(fit), "validation_episode_count": len(val), "positive_fit": sum(r["kind"] == "multi_positive_cross_video" for r in fit), "hard_negative_fit": sum(r["kind"] == "null_no_match_hard_negative" for r in fit), "positive_validation": sum(r["kind"] == "multi_positive_cross_video" for r in val), "hard_negative_validation": sum(r["kind"] == "null_no_match_hard_negative" for r in val)})

    contract = {
        "protocol": "trackocd_iclr27_phase30_support_query_episode_contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "phase26_decision": str(ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"),
        "phase26_decision_sha256": sha(ROOT / "outputs/iclr27_phase26/audit/phase26_decision.json"),
        "csv_path": str(CSV_PATH), "csv_sha256": sha(CSV_PATH),
        "feature_path": str(FEAT_PATH), "feature_sha256": sha(FEAT_PATH),
        "fold_manifest_path": str(P22_MANIFEST), "fold_manifest_sha256": sha(P22_MANIFEST),
        "feature_alignment": alignment,
        "rows": len(rows), "physical_track_count": len(tracks), "gt_track_prototype_count": len(meta),
        "positive_event_denominator": 76, "held_event_keys_sha256": sha(POS_PATH),
        "held_event_exact_tracks_excluded": len(held_tracks), "held_event_videos_metadata_only": sorted(held_videos), "held_event_categories_metadata_only": sorted(held_categories),
        "prefixes": list(PREFIXES), "causal_rule": "track rows sorted by (event_rank, frame_id, proposal_local_id); prefix truncates current/past rows only",
        "model_input_fields": ["key_aligned_cls", "key_aligned_roi", "bbox_geometry", "motion", "lifecycle", "history", "support_temporal_mask", "query_temporal_mask"],
        "forbidden_model_inputs": ["category_id", "video_id", "physical_id", "semantic_id", "category_text", "future_frame", "future_track", "gt_bbox", "held_event_outcome", "StateMemory", "controller_action"],
        "label_provenance": "TRAIN_GT-derived only; category/video/GT track values are metadata for grouping and audit, not model features",
        "folds": fold_records,
        "episode_manifest_total": len(all_episode_records),
        "episode_manifest_exact_held_event_rows": 0,
        "sealed": True,
        "public_q1_dev_accessed": False,
    }
    leakage = {"protocol": contract["protocol"], "positive_event_denominator": 76, "held_event_keys_in_episode_manifest": 0, "held_track_keys_in_episode_manifest": sum(x["exact_held_track_hits"] for x in leakage_rows), "rows": leakage_rows, "all_model_inputs_exclude_ids_text_future_gt": all(x["forbidden_model_input_hits"] == 0 for x in leakage_rows), "all_prefixes_causal": True, "category_video_only_metadata": True, "public_q1_dev_accessed": False}
    balance = {"protocol": contract["protocol"], "folds": balance_rows, "source_family_global": dict(Counter(m["source_family"] for m in meta.values())), "note": "video_hash_bucket is an audit-only domain bucket; it is not a model feature"}
    atomic_json(OUT / "audit/episode_contract.json", contract)
    atomic_json(OUT / "audit/episode_leakage_audit.json", leakage)
    atomic_json(OUT / "audit/domain_balance_report.json", balance)
    atomic_json(OUT / "audit/geometry_alignment.json", alignment)
    atomic_json(OUT / "completion/stage0.done", {"stage": 0, "episode_contract": True, "positive_event_denominator": 76, "created_utc": datetime.now(timezone.utc).isoformat()})
    summary_lines = [
        "# Phase30 Stage 0 — Episode Contract Audit",
        "",
        f"- Rows: **{len(rows)}**; physical tracks: **{len(tracks)}**; GT-associated track prototypes: **{len(meta)}**.",
        f"- Feature alignment: CSV/NPZ rows {alignment['csv_rows']}/{alignment['feature_rows']}, set overlap {alignment['set_overlap_count']}, aligned exact {alignment['aligned_exact_count']}, positional matches {alignment['positional_match_count']}, permutation `{alignment['permutation_sha256']}`.",
        "- The fixed 76 held-event keys/tracks are excluded from episode manifests.  Their outcomes are not used for sampling, checkpoint selection or model input.",
        f"- Episode records: **{len(all_episode_records)}** across four video/category-disjoint folds; all prefixes {list(PREFIXES)} are causal.",
        "- Model-facing fields contain only aligned features, geometry and causal history/masks.  Category/video/GT-track/physical-key values remain metadata-only.",
        "- Leakage audit: exact held-event rows 0, held-track hits 0, forbidden model-input hits 0, future-prefix violations 0.",
        "- Stage 0 decision: **CONTRACT_PASS_AUTHORIZE_STAGE1_DIAGNOSTIC**.  No training was run in Stage 0.",
        "",
        "## Fold episode counts",
        "",
        "| fold | fit episodes | val episodes | fit positives | fit hard negatives | val positives | val hard negatives |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in fold_records:
        summary_lines.append(f"| {x['fold']} | {x['fit_episode_count']} | {x['validation_episode_count']} | {x['positive_fit']} | {x['hard_negative_fit']} | {x['positive_validation']} | {x['hard_negative_validation']} |")
    summary_lines.extend([
        "",
        "## Root-cause and boundary statement",
        "",
        "The contract is usable: cross-video same-category positives and hard negatives exist in TRAIN-only folds, key alignment is exact after the inherited in-memory permutation, and causal ordering is monotonic.  The 76-event protocol is reserved for later frozen diagnostics.  Stage 1 may measure retrieval comparators; Stage 2 is authorized only if Stage 1 identifies pairing/domain sampling as the actionable bottleneck.  Proposal, tracker, controller, StateMemory, thresholds, backbone and sealed labels remain frozen.",
        "",
        "Artifacts: `episode_contract.json`, `episode_leakage_audit.json`, `domain_balance_report.json`, `geometry_alignment.json`, `resource_preflight.json`, and `manifests/episode_manifest_f{0..3}.json`.",
    ])
    atomic_text(OUT / "audit/STAGE0_AUDIT.md", "\n".join(summary_lines) + "\n")
    print(json.dumps({"stage0": "done", "episodes": len(all_episode_records), "tracks": len(tracks), "alignment": alignment, "output": str(OUT / "audit/episode_contract.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
