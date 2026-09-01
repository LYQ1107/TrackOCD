#!/usr/bin/env python3
"""Phase22 Stage 0: evidence-backed taxonomy of the 76 failed events.

Only the public-TRAIN-derived Phase21 CSV/event audit is read.  The script
never opens DEV+, Q1, or a public new-model label file and never changes the
frozen evaluator.  Every positive event is retained in the output; the
prefix16 failed subset is a view of the same 76-event denominator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
POS_PATH = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
P21_AUDIT = ROOT / "outputs/iclr27_phase21/audit/observability_event_audit.json"
P21_GEOM = ROOT / "outputs/iclr27_phase21/audit/geometry_audit.json"
P21_SUMMARY = ROOT / "outputs/iclr27_phase21/audit/full_76_event_summary.csv"
OUT = ROOT / "outputs/iclr27_phase22"
PREFIX = 16
IOU_THR = 0.5
SMALL_AREA_THR = 0.01
LOW_STABILITY_THR = 0.25


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def parse_box(value: str | None) -> list[float] | None:
    if value is None or not str(value).strip():
        return None
    try:
        box = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        out = [float(x) for x in box]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


def box_area_fraction(row: dict[str, str], gt: bool = False) -> float:
    if gt:
        box = parse_box(row.get("gt_bbox_xyxy"))
        w, h = fval(row, "image_width"), fval(row, "image_height")
        if box is None or w <= 0 or h <= 0:
            return 0.0
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (w * h)
    return fval(row, "area_fraction")


def track_key(row: dict[str, str]) -> str:
    return f"v{int(row['video_id'])}:p{int(row['track_id'])}"


def ordered(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0)), int(r.get("proposal_local_id", 0))))


def is_assigned(row: dict[str, str]) -> bool:
    return str(row.get("assigned", "0")) == "1"


def reliable(row: dict[str, str]) -> bool:
    return is_assigned(row) and fval(row, "row_iou") >= IOU_THR


def row_evidence(row: dict[str, str]) -> dict[str, Any]:
    return {
        "row_key": str(row.get("row_key", "")),
        "video_id": int(row.get("video_id", -1)),
        "frame_id": int(row.get("frame_id", -1)),
        "event_rank": int(row.get("event_rank", -1)),
        "proposal_local_id": int(row.get("proposal_local_id", -1)),
        "track_id": int(row.get("track_id", -1)),
        "assigned": int(is_assigned(row)),
        "score": fval(row, "score"),
        "row_iou": fval(row, "row_iou"),
        "area_fraction": box_area_fraction(row),
        "gt_area_fraction": box_area_fraction(row, gt=True),
        "box_width_norm": fval(row, "box_width_norm"),
        "box_height_norm": fval(row, "box_height_norm"),
        "box_aspect": math.exp(fval(row, "box_aspect_log")),
        "track_temporal_iou": fval(row, "track_temporal_iou"),
        "causal_box_stability_iou": fval(row, "causal_box_stability_iou"),
        "image_width": int(fval(row, "image_width")),
        "image_height": int(fval(row, "image_height")),
    }


def compact_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    ev = [row_evidence(r) for r in rows]
    def vals(key: str) -> list[float]:
        return [float(x[key]) for x in ev]
    def stat(key: str) -> dict[str, float | int]:
        xs = vals(key)
        if not xs:
            return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {"count": len(xs), "min": min(xs), "mean": statistics.mean(xs), "median": statistics.median(xs), "max": max(xs)}
    return {
        "candidate_box_count": len(rows),
        "assigned_count": sum(is_assigned(r) for r in rows),
        "reliable_count": sum(reliable(r) for r in rows),
        "max_iou": max((fval(r, "row_iou") for r in rows), default=0.0),
        "max_score": max((fval(r, "score") for r in rows), default=0.0),
        "iou": stat("row_iou"),
        "score": stat("score"),
        "area_fraction": stat("area_fraction"),
        "gt_area_fraction": stat("gt_area_fraction"),
        "aspect": stat("box_aspect"),
        "track_temporal_iou": stat("track_temporal_iou"),
        "causal_stability_iou": stat("causal_box_stability_iou"),
        "frames": [int(x["frame_id"]) for x in ev],
        "event_ranks": [int(x["event_rank"]) for x in ev],
    }


def classify(event: dict[str, Any], source: list[dict[str, str]], target_prefix: list[dict[str, str]], geom: dict[str, Any]) -> dict[str, Any]:
    source_ev = [row_evidence(r) for r in source]
    target_ev = [row_evidence(r) for r in target_prefix]
    source_rel = any(reliable(r) for r in source)
    target_rel = any(reliable(r) for r in target_prefix)
    source_assigned = any(is_assigned(r) for r in source)
    target_assigned = any(is_assigned(r) for r in target_prefix)
    expected_keys = [str(x) for x in event.get("target_row_keys", [])]
    present_keys = {str(r.get("row_key", "")) for r in target_prefix}
    missing_keys = [x for x in expected_keys if x not in present_keys]
    failures: list[str] = []
    if not source:
        failures.append("source_proposal_missing")
    elif not source_assigned:
        failures.append("source_assignment_or_wrong_proposal")
    if not target_prefix:
        failures.append("target_proposal_missing_in_prefix")
    elif not target_assigned:
        failures.append("target_assignment_or_wrong_proposal")
    if missing_keys:
        failures.append("proposal_wrong_frame_or_rank")
    if geom.get("invalid_bbox_rows", 0) or geom.get("normalized_coordinate_mismatch_rows", 0) or geom.get("stored_iou_mismatch_rows", 0):
        failures.append("bbox_coordinate_or_scale_problem")
    if source and target_prefix and source_assigned and target_assigned and not source_rel and not target_rel:
        primary = "both_source_target_box_iou_below_0.5"
    elif source and source_assigned and not source_rel:
        primary = "source_box_iou_below_0.5"
    elif target_prefix and target_assigned and not target_rel:
        primary = "target_box_iou_below_0.5"
    elif source_rel and target_rel:
        primary = "both_observable_assignment_or_matching_failure"
    elif failures:
        primary = failures[0]
    else:
        primary = "other_unresolved"
    # These are evidence flags, not causal claims.  Occlusion labels are not
    # present in the frozen CSV, so they are explicitly marked unavailable.
    combined = source_ev + target_ev
    gt_areas = [float(x["gt_area_fraction"]) for x in combined if float(x["gt_area_fraction"]) > 0]
    aspects = [float(x["box_aspect"]) for x in combined if float(x["box_aspect"]) > 0]
    stabilities = [float(x["track_temporal_iou"]) for x in combined]
    secondary: list[str] = []
    if gt_areas and statistics.median(gt_areas) < SMALL_AREA_THR:
        secondary.append("small_object_evidence_median_gt_area_lt_1pct")
    if aspects and (statistics.median(aspects) < 1 / 3 or statistics.median(aspects) > 3):
        secondary.append("extreme_aspect_evidence")
    if stabilities and statistics.median(stabilities) < LOW_STABILITY_THR:
        secondary.append("low_temporal_stability_proxy")
    if int(event.get("source_video", -1)) != int(event.get("target_video", -2)):
        secondary.append("cross_video_pair_domain_difference_hypothesis_not_proven")
    secondary.append("occlusion_label_unavailable_in_source")
    return {
        "event_key": str(event["event_key"]),
        "fold": int(event["fold"]),
        "category": int(event.get("category_gt_denominator_only", -1)),
        "source_video": int(event.get("source_video", -1)),
        "target_video": int(event.get("target_video", -1)),
        "source_tracklet_key": str(event["source_tracklet_keys"][0]),
        "target_tracklet_key": str(event["target_tracklet_key"]),
        "prefix": PREFIX,
        "is_failed_event": not (source_rel and target_rel),
        "primary_failure_class": primary if not (source_rel and target_rel) else "none_observable",
        "failure_flags": sorted(set(failures)),
        "secondary_evidence_flags": sorted(set(secondary)),
        "evidence_status": {
            "geometry_errors_in_phase21": bool(geom.get("invalid_bbox_rows", 0) or geom.get("normalized_coordinate_mismatch_rows", 0) or geom.get("stored_iou_mismatch_rows", 0)),
            "cross_video_or_category_is_causal": False,
            "occlusion_is_observed": False,
            "notes": "Cross-video/category and size/stability flags are correlational diagnostics; no hidden label or future frame is used.",
        },
        "source": {
            "track_length": len(source),
            "prefix_visible": bool(source),
            "summary": compact_stats(source),
            "candidate_boxes": source_ev,
        },
        "target": {
            "track_length": len(by_track_global.get(str(event["target_tracklet_key"]), [])),
            "prefix_visible": bool(target_prefix),
            "prefix_candidate_box_count": len(target_prefix),
            "summary": compact_stats(target_prefix),
            "candidate_boxes": target_ev,
        },
        "assignment_and_chronology": {
            "source_assigned": source_assigned,
            "target_assigned_in_prefix": target_assigned,
            "source_reliable": source_rel,
            "target_reliable_in_prefix": target_rel,
            "expected_target_row_keys": expected_keys,
            "expected_target_row_keys_missing_in_prefix": missing_keys,
            "target_first_reliable_prefix_index_manifest": event.get("target_first_reliable_prefix_index_gt_only"),
            "target_frames_in_prefix": [int(r.get("frame_id", -1)) for r in target_prefix],
            "target_event_ranks_in_prefix": [int(r.get("event_rank", -1)) for r in target_prefix],
        },
        "perfect_correspondence_ceiling": bool(source_rel and target_rel),
    }


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    global by_track_global
    by_track_global = defaultdict(list)
    for row in rows:
        by_track_global[track_key(row)].append(row)
    for key in list(by_track_global):
        by_track_global[key] = ordered(by_track_global[key])
    events = [json.loads(x) for x in POS_PATH.read_text().splitlines() if x.strip()]
    assert len(events) == 76, len(events)
    geom = json.loads(P21_GEOM.read_text())
    p21_records = json.loads(P21_AUDIT.read_text())["records"]
    p21_16 = {str(x["event_key"]): x for x in p21_records if x["kind"] == "positive_existing" and int(x["prefix"]) == PREFIX}
    assert len(p21_16) == 76, len(p21_16)
    records = []
    for event in sorted(events, key=lambda e: str(e["event_key"])):
        sk = str(event["source_tracklet_keys"][0]); tk = str(event["target_tracklet_key"])
        source = by_track_global.get(sk, []); target = by_track_global.get(tk, [])[:PREFIX]
        rec = classify(event, source, target, geom)
        rec["phase21_audit_reference"] = p21_16[rec["event_key"]]
        records.append(rec)
    failed = [x for x in records if x["is_failed_event"]]
    primary = Counter(x["primary_failure_class"] for x in failed)
    secondary = Counter(flag for x in failed for flag in x["secondary_evidence_flags"])
    by_fold: dict[str, Any] = {}
    for fold in sorted({int(x["fold"]) for x in records}):
        fs = [x for x in failed if int(x["fold"]) == fold]
        by_fold[str(fold)] = {"failed": len(fs), "denominator": len([x for x in records if int(x["fold"]) == fold]), "primary": dict(Counter(x["primary_failure_class"] for x in fs))}
    by_category: dict[str, Any] = {}
    for cat in sorted({int(x["category"]) for x in records}):
        cs = [x for x in failed if int(x["category"]) == cat]
        by_category[str(cat)] = {"failed": len(cs), "denominator": len([x for x in records if int(x["category"]) == cat]), "primary": dict(Counter(x["primary_failure_class"] for x in cs))}
    taxonomy = {
        "protocol": "trackocd_iclr27_phase22_stage0_failure_taxonomy",
        "source_csv": str(CSV_PATH),
        "source_csv_sha256": sha256(CSV_PATH),
        "phase21_inputs": {"event_audit": str(P21_AUDIT), "geometry_audit": str(P21_GEOM), "full_event_summary": str(P21_SUMMARY)},
        "positive_event_denominator": 76,
        "prefix": PREFIX,
        "reliable_rule": "assigned == 1 and row_iou >= 0.5",
        "records": records,
        "failed_event_count": len(failed),
        "phase21_prefix16_ceiling": sum(x["perfect_correspondence_ceiling"] for x in records),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
    }
    summary = {
        "protocol": "trackocd_iclr27_phase22_stage0_failure_taxonomy_summary",
        "positive_event_denominator": 76,
        "prefix16_failed": len(failed),
        "prefix16_ceiling": 76 - len(failed),
        "dominant_primary_failure": primary.most_common(1)[0][0] if primary else "none",
        "primary_failure_counts": dict(primary),
        "secondary_evidence_counts": dict(secondary),
        "by_fold": by_fold,
        "by_category": by_category,
        "geometry_error_counts_from_phase21": {k: geom.get(k, 0) for k in ["invalid_bbox_rows", "normalized_coordinate_mismatch_rows", "stored_iou_mismatch_rows", "chronology_bad_track_count", "event_rank_duplicate_track_count"]},
        "interpretation": "The primary taxonomy is proposal/box evidence. Cross-video/category, size, and stability flags are not causal claims; occlusion labels are unavailable.",
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
    }
    atomic_json(OUT / "audit/failure_taxonomy_76.json", taxonomy)
    atomic_json(OUT / "audit/failure_taxonomy_summary.json", summary)
    atomic_json(OUT / "completion/stage0.done", {"stage": "stage0", "positive_denominator": 76, "failed_prefix16": len(failed), "ceiling_prefix16": 76 - len(failed), "decision": "proposal_failure_taxonomy_complete"})
    lines = [
        "# Phase22 Stage 0 — prefix16 proposal failure taxonomy", "",
        "Every one of the 76 positive events is retained.  This audit reads only public-TRAIN-derived DSCT rows and the existing Phase21 event audit; DEV+, Q1, and public new-model labels are sealed.", "",
        f"The frozen reliable rule is `{taxonomy['reliable_rule']}`.  Prefix16 reproduces **{76-len(failed)}/76** observable events and **{len(failed)}/76** failures.", "",
        "## Primary failure classes", "", "| primary class | events | interpretation |", "|---|---:|---|",
    ]
    interpretations = {
        "source_box_iou_below_0.5": "source track has assigned rows but no reliable box",
        "target_box_iou_below_0.5": "target prefix has assigned rows but no reliable box",
        "both_source_target_box_iou_below_0.5": "both sides have assigned rows but no reliable box",
        "source_proposal_missing": "no source track rows",
        "target_proposal_missing_in_prefix": "no target rows in causal prefix",
        "source_assignment_or_wrong_proposal": "source rows exist but none assigned",
        "target_assignment_or_wrong_proposal": "target rows exist but none assigned",
        "proposal_wrong_frame_or_rank": "manifest row key absent from causal prefix",
        "bbox_coordinate_or_scale_problem": "Phase21 geometry audit flagged a transform error",
        "both_observable_assignment_or_matching_failure": "both sides reliable; investigate evaluator matching",
    }
    for k, n in primary.most_common(): lines.append(f"| `{k}` | {n} | {interpretations.get(k, 'unresolved') } |")
    lines += ["", "## Evidence and limits", "", "For each event the JSON stores candidate-box counts, row keys, frames/ranks, assigned/reliable counts, max and distributional IoU/score/area/aspect/stability, actual image dimensions, expected target keys, and the Phase21 reference record.  Size/aspect/stability flags are descriptive; the frozen CSV has no occlusion label, and cross-video/category differences are hypotheses rather than established causes.", "", "Fold and category aggregates are in `failure_taxonomy_summary.json`; no event or denominator was removed."]
    (ROOT / "docs/iclr27_phase22/STAGE0_FAILURE_TAXONOMY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"events": 76, "failed_prefix16": len(failed), "ceiling_prefix16": 76 - len(failed), "primary": dict(primary)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
