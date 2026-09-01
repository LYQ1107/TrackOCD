#!/usr/bin/env python
"""Phase20 Stage 0: exact observability audit on frozen DSCT proposals.

This script only reads the public TRAIN-derived Phase19R CSV/feature cache and
the TRAIN-derived pseudo-held event manifest.  It never opens DEV+, Q1, or a
public new-model label file.  Every event is retained at every causal prefix;
the ceiling is therefore an exact denominator rather than a filtered score.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OLD_MANIFEST = ROOT / "outputs/iclr27_phase19r/manifests/fold_manifest.json"
OLD_POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
OLD_NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
OUT = ROOT / "outputs/iclr27_phase20"
PREFIXES = (1, 2, 4, 8, 16)
IOU_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def parse_key(key: str) -> tuple[int, int]:
    a, b = key.split(":p")
    return int(a[1:]), int(b)


def load_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bool_assigned(row: dict[str, str]) -> bool:
    return str(row.get("assigned", "0")) == "1"


def iou(row: dict[str, str]) -> float:
    try:
        return float(row.get("row_iou", 0.0))
    except (TypeError, ValueError):
        return 0.0


def reliable(row: dict[str, str]) -> bool:
    return bool_assigned(row) and iou(row) >= IOU_THRESHOLD


def ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0)), int(r.get("proposal_local_id", 0))))


def failure_reason(source_rows: list[dict[str, str]], target_rows: list[dict[str, str]], p: int) -> str:
    if not source_rows:
        return "source_track_missing"
    if not any(bool_assigned(r) for r in source_rows):
        return "source_no_assigned_proposal"
    if not any(reliable(r) for r in source_rows):
        return "source_no_reliable_observation"
    if not target_rows:
        return "target_track_missing"
    prefix = target_rows[: min(int(p), len(target_rows))]
    if not prefix:
        return "target_no_proposal_in_prefix"
    if not any(bool_assigned(r) for r in prefix):
        return "target_no_assigned_proposal_in_prefix"
    if not any(reliable(r) for r in prefix):
        return "target_no_reliable_observation_in_prefix"
    return "observable"


def main() -> None:
    out_audit = OUT / "audit"
    out_manifest = OUT / "manifests"
    out_audit.mkdir(parents=True, exist_ok=True)
    out_manifest.mkdir(parents=True, exist_ok=True)

    with SRC.open(newline="") as f:
        rows = list(csv.DictReader(f))
    by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_track[f"v{int(row['video_id'])}:p{int(row['track_id'])}"].append(row)
    for key in list(by_track):
        by_track[key] = ordered_rows(by_track[key])

    # Copy only the lightweight TRAIN-derived fold description into the new
    # namespace.  The source CSV is referenced, not copied.
    fold_manifest = json.loads(OLD_MANIFEST.read_text())
    fold_manifest = dict(fold_manifest)
    fold_manifest["phase20_namespace"] = True
    fold_manifest["source_rows_path"] = str(SRC)
    fold_manifest["source_rows_sha256"] = sha256(SRC)
    fold_manifest["labels_used"] = "public TRAIN category/video metadata only"
    fold_manifest["sealed_inputs"] = ["DEV+", "Q1", "public new-model labels"]
    atomic_json(out_manifest / "fold_manifest.json", fold_manifest)

    events = load_events(OLD_POS) + load_events(OLD_NEG)
    events.sort(key=lambda e: str(e.get("event_key", "")))
    positives = [e for e in events if e.get("kind") == "positive_existing"]
    negatives = [e for e in events if e.get("kind") == "negative_new"]
    assert len(positives) == 76 and len(negatives) == 76, (len(positives), len(negatives))

    event_records: list[dict[str, Any]] = []
    by_fold_prefix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    all_prefix: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        sk = str(event["source_tracklet_keys"][0])
        tk = str(event["target_tracklet_key"])
        source_rows = by_track.get(sk, [])
        target_rows = by_track.get(tk, [])
        source_video = int(event.get("source_video", parse_key(sk)[0]))
        target_video = int(event.get("target_video", parse_key(tk)[0]))
        assert source_video != target_video, event["event_key"]
        for p in PREFIXES:
            target_prefix = target_rows[: min(int(p), len(target_rows))]
            source_assigned = [r for r in source_rows if bool_assigned(r)]
            target_assigned = [r for r in target_prefix if bool_assigned(r)]
            source_rel = [r for r in source_rows if reliable(r)]
            target_rel = [r for r in target_prefix if reliable(r)]
            target_ious = [iou(r) for r in target_prefix]
            source_ious = [iou(r) for r in source_rows]
            try:
                target_areas = [float(r.get("area_fraction", 0.0)) for r in target_prefix]
                source_areas = [float(r.get("area_fraction", 0.0)) for r in source_rows]
            except (TypeError, ValueError):
                target_areas, source_areas = [0.0 for _ in target_prefix], [0.0 for _ in source_rows]
            rec: dict[str, Any] = {
                "event_key": str(event["event_key"]),
                "fold": int(event["fold"]),
                "kind": str(event["kind"]),
                "category": int(event.get("target_category_gt_denominator_only", event.get("category_gt_denominator_only", -1))),
                "source_category": int(event.get("distractor_category_gt_denominator_only", event.get("category_gt_denominator_only", -1))),
                "source_tracklet_key": sk,
                "target_tracklet_key": tk,
                "source_video": source_video,
                "target_video": target_video,
                "causal_prefix_requested": int(p),
                "source_track_length": len(source_rows),
                "target_track_length": len(target_rows),
                "source_has_proposal": bool(source_rows),
                "target_has_proposal_in_prefix": bool(target_prefix),
                "source_assigned_rows": len(source_assigned),
                "target_assigned_rows_in_prefix": len(target_assigned),
                "source_reliable_rows": len(source_rel),
                "target_reliable_rows_in_prefix": len(target_rel),
                "source_materialized": bool(source_rows),
                "source_reliable_materialized": bool(source_rel),
                "target_visible": bool(target_prefix),
                "target_reliably_visible": bool(target_rel),
                "source_mean_iou": float(statistics.mean(source_ious)) if source_ious else 0.0,
                "target_prefix_mean_iou": float(statistics.mean(target_ious)) if target_ious else 0.0,
                "target_prefix_median_iou": float(statistics.median(target_ious)) if target_ious else 0.0,
                "target_prefix_max_iou": float(max(target_ious)) if target_ious else 0.0,
                "source_mean_area_fraction": float(statistics.mean(source_areas)) if source_areas else 0.0,
                "target_prefix_mean_area_fraction": float(statistics.mean(target_areas)) if target_areas else 0.0,
                "target_first_reliable_index_calculated": next((i for i, r in enumerate(target_rows) if reliable(r)), None),
                "target_first_reliable_index_manifest": event.get("target_first_reliable_prefix_index_gt_only"),
                "failure_reason": failure_reason(source_rows, target_rows, p),
            }
            rec["perfect_correspondence_ct_ceiling"] = bool(
                rec["kind"] == "positive_existing"
                and rec["source_reliable_materialized"]
                and rec["target_reliably_visible"]
            )
            event_records.append(rec)
            all_prefix[p].append(rec)

    # Exact aggregate with positive CT denominator held fixed at 76 for every
    # prefix; negative events remain in observability counts but never enter CT.
    prefix_summary: list[dict[str, Any]] = []
    for p in PREFIXES:
        rs = all_prefix[p]
        pos = [r for r in rs if r["kind"] == "positive_existing"]
        neg = [r for r in rs if r["kind"] == "negative_new"]
        good = [r for r in pos if r["perfect_correspondence_ct_ceiling"]]
        cats = sorted({int(r["category"]) for r in good})
        vids = sorted({int(r["target_video"]) for r in good})
        reasons = defaultdict(int)
        for r in pos:
            reasons[str(r["failure_reason"])] += 1
        prefix_summary.append({
            "prefix": int(p),
            "positive_denominator": len(pos),
            "negative_denominator": len(neg),
            "source_reliable_materialized": int(sum(r["source_reliable_materialized"] for r in pos)),
            "target_reliably_visible": int(sum(r["target_reliably_visible"] for r in pos)),
            "perfect_correspondence_ct_ceiling_correct": len(good),
            "perfect_correspondence_ct_ceiling_recall": len(good) / max(len(pos), 1),
            "ceiling_category_coverage": len(cats),
            "ceiling_video_coverage": len(vids),
            "ceiling_categories": cats,
            "ceiling_videos": vids,
            "mean_target_prefix_iou": float(statistics.mean(r["target_prefix_mean_iou"] for r in pos)) if pos else 0.0,
            "median_target_prefix_iou": float(statistics.median(r["target_prefix_median_iou"] for r in pos)) if pos else 0.0,
            "mean_source_track_length": float(statistics.mean(r["source_track_length"] for r in pos)) if pos else 0.0,
            "mean_target_track_length": float(statistics.mean(r["target_track_length"] for r in pos)) if pos else 0.0,
            "mean_source_area_fraction": float(statistics.mean(r["source_mean_area_fraction"] for r in pos)) if pos else 0.0,
            "mean_target_prefix_area_fraction": float(statistics.mean(r["target_prefix_mean_area_fraction"] for r in pos)) if pos else 0.0,
            "failure_reasons": dict(sorted(reasons.items())),
        })

    per_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in PREFIXES:
        for fold in range(4):
            rs = [r for r in all_prefix[p] if int(r["fold"]) == fold]
            pos = [r for r in rs if r["kind"] == "positive_existing"]
            good = [r for r in pos if r["perfect_correspondence_ct_ceiling"]]
            per_fold[str(fold)].append({
                "prefix": int(p), "positive_denominator": len(pos),
                "ceiling_correct": len(good),
                "ceiling_recall": len(good) / max(len(pos), 1),
                "source_reliable": int(sum(r["source_reliable_materialized"] for r in pos)),
                "target_reliable": int(sum(r["target_reliably_visible"] for r in pos)),
                "category_coverage": len({int(r["category"]) for r in good}),
                "video_coverage": len({int(r["target_video"]) for r in good}),
            })

    # CSV is intentionally event x prefix: no failure is discarded.
    csv_path = out_audit / "observability_events.csv"
    fields = list(event_records[0].keys())
    fd, tmp = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=str(csv_path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(event_records)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, csv_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    atomic_json(out_audit / "observability_events.json", event_records)

    max_ceiling = max(x["perfect_correspondence_ct_ceiling_recall"] for x in prefix_summary)
    ceiling = {
        "protocol": "trackocd_iclr27_phase20_stage0_observability",
        "source_rows": len(rows),
        "source_rows_path": str(SRC),
        "source_rows_sha256": sha256(SRC),
        "events_total": len(events), "positive_events": len(positives), "negative_events": len(negatives),
        "fixed_positive_ct_denominator": len(positives),
        "prefixes": list(PREFIXES),
        "reliable_rule": "assigned == 1 and row_iou >= 0.5",
        "proposal_source": "frozen DSCT rows; GT-tight boxes are not in main path",
        "prefix_summary": prefix_summary,
        "max_positive_ceiling_recall": max_ceiling,
        "gate_o_rule": "pass only when the largest causal-prefix perfect-correspondence ceiling is >= 0.50 (majority of positive events)",
        "gate_o_pass": bool(max_ceiling >= 0.50),
        "decision": "O_PASS_ENTER_STAGE1" if max_ceiling >= 0.50 else "O_FAIL_STOP_CORRESPONDENCE_LONG_TRAINING",
        "labels_used": "public TRAIN category/video metadata only",
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
    }
    atomic_json(out_audit / "observability_ceiling.json", ceiling)
    atomic_json(out_audit / "observability_by_prefix.json", {
        "protocol": "trackocd_iclr27_phase20_stage0_observability_by_prefix",
        "prefixes": list(PREFIXES), "aggregate": prefix_summary, "by_fold": dict(per_fold),
        "event_artifact": str(out_audit / "observability_events.json"),
    })
    atomic_json(OUT / "completion/stage0.done", {
        "stage": "stage0", "gate_o_pass": ceiling["gate_o_pass"],
        "max_positive_ceiling_recall": max_ceiling,
        "event_csv": str(csv_path), "event_json": str(out_audit / "observability_events.json"),
    })
    print(json.dumps({"events": len(events), "prefix_summary": prefix_summary,
                      "gate_o_pass": ceiling["gate_o_pass"], "max_ceiling": max_ceiling}, indent=2))


if __name__ == "__main__":
    main()
