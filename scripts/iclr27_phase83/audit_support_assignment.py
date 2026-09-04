#!/usr/bin/env python3
"""Read-only Phase83 audit of support assignment provenance and O ceiling.

The script consumes Phase75B event diagnostics but never changes the frozen
Phase75B evaluator.  GT-derived fields are copied only into audit evidence;
they are explicitly excluded from any router input.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
JOIN = ROOT / "outputs/iclr27_phase74s/manifests/evaluator_join_v2.jsonl"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
IOU_T = 0.5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def side_evidence(record: dict[str, Any], side: str) -> dict[str, Any]:
    x = record[side]
    details = record.get(f"{side}_row_details", [])
    frame_ids = sorted({int(d.get("frame_id", -1)) for d in details})
    image_ids = sorted({int(d.get("image_id", -1)) for d in details})
    event_iou = [float(d.get("event_transformed_iou") or 0.0) for d in details]
    temporal_iou = [float(d.get("event_track_temporal_iou") or 0.0) for d in details]
    q0_iou = [float(d.get("q0_max_iou") or 0.0) for d in details]
    # Deployment-safe candidate descriptors; no GT/category/ID field is fed to
    # a model.  They are retained here only to make the audit reproducible.
    return {
        "candidate_count": int(x.get("candidate_count", 0)),
        "candidate_rows": int(x.get("candidate_rows", 0)),
        "rows_used": int(x.get("rows_used", 0)),
        "no_detection_rows": int(x.get("no_detection_rows", 0)),
        "asset_missing_rows": int(x.get("asset_missing_rows", 0)),
        "ambiguity_rows": int(x.get("ambiguity_rows", 0)),
        "max_iou": float(x.get("max_iou", 0.0)),
        "mean_max_iou": float(x.get("mean_max_iou", 0.0)),
        "median_max_iou": float(x.get("median_max_iou", 0.0)),
        "q0_reliable_rows": int(x.get("q0_reliable_rows", 0)),
        "event_reliable_rows": int(x.get("event_reliable_rows", 0)),
        "joint_reliable_rows": int(x.get("joint_reliable_rows", 0)),
        "frame_ids": frame_ids,
        "image_ids": image_ids,
        "event_iou_max": max(event_iou, default=0.0),
        "event_temporal_iou_max": max(temporal_iou, default=0.0),
        "q0_iou_max_from_rows": max(q0_iou, default=0.0),
        "row_details": details,
    }


def classify(record: dict[str, Any], side: str) -> tuple[str, str]:
    x = record[side]
    details = record.get(f"{side}_row_details", [])
    if int(x.get("candidate_count", 0)) <= 0 or int(x.get("q0_reliable_rows", 0)) == 0 and not details:
        return "A_NO_PROPOSAL", f"{side}: no causal candidate rows"
    if float(x.get("max_iou", 0.0)) < IOU_T:
        return "B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05", f"{side}: pool max IoU < 0.5"
    # A good native pool but no frozen event-row assignment is a selection or
    # parent-assignment discrepancy, not a geometry failure.
    if int(x.get("event_reliable_rows", 0)) == 0:
        assigned = any(bool(d.get("event_assigned")) for d in details)
        event_iou = max((float(d.get("event_transformed_iou") or 0.0) for d in details), default=0.0)
        if assigned and event_iou < IOU_T:
            return "E_SUPPORT_SELECTION_WRONG", f"{side}: native pool good, frozen assigned row IoU below threshold"
        return "C_POOL_GOOD_BUT_ASSIGNED_0", f"{side}: native pool good, no reliable assigned row"
    # Assigned rows exist but transformed IoU is below threshold for the
    # aggregate only when a stricter temporal/event condition fails.
    if int(x.get("q0_reliable_rows", 0)) > 0 and int(x.get("event_reliable_rows", 0)) < int(x.get("q0_reliable_rows", 0)):
        return "D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05", f"{side}: assigned/native rows diverge under transformed IoU"
    return "G_OTHER", f"{side}: no dominant failure"


def callgraph() -> dict[str, Any]:
    paths = [
        ROOT / "data/iclr27_phase17/sources/public_role_rows_phase17.csv",
        ROOT / "src/iclr27_phase17r/data/build_corrected_public.py",
        CSV_PATH,
        ROOT / "scripts/iclr27_phase19r/build_pseudo_events.py",
        POS, NEG, JOIN,
        ROOT / "scripts/iclr27_phase75b/run_observability.py",
        OBS,
        ROOT / "scripts/iclr27_phase82p/evaluate_strict_o_residual.py",
    ]
    file_info = []
    for p in paths:
        if p.is_file():
            file_info.append({"path": str(p.resolve()), "sha256": sha(p), "bytes": p.stat().st_size})
        else:
            file_info.append({"path": str(p), "missing": True})
    return {
        "schema_version": "trackocd.phase83.support_assignment_callgraph.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "assigned_origin": "upstream data/iclr27_phase17/sources/public_role_rows_phase17.csv; corrected-public builder preserves assigned and row_iou",
        "row_iou_origin": "upstream role rows; corrected-public builder preserves row_iou and adds geometry/causal fields",
        "track_temporal_iou_origin": "upstream role rows; corrected-public builder preserves track_temporal_iou",
        "event_join_origin": "Phase74S evaluator_join_v2 metadata join; no GT is supplied to inference",
        "q0_pool_origin": "Phase75B native lineage max IoU diagnostic; post-hoc only",
        "flow": [
            "public_role_rows_phase17.csv -> build_corrected_public.py -> public_rows_corrected.csv",
            "build_pseudo_events.py -> held_known_positive_events.jsonl / held_known_negative_events.jsonl",
            "evaluator_join_v2.jsonl -> run_observability.py -> event_observability.jsonl",
            "evaluate_strict_o_residual.py -> post-hoc replay mapping (not used as Phase83 input)",
        ],
        "fields_never_used_as_router_input": ["assigned", "row_iou", "track_temporal_iou", "gt_bbox_xyxy", "gt_category_id_common", "semantic/physical IDs", "event identity"],
        "files": file_info,
    }


def main() -> None:
    obs = load_jsonl(OBS)
    join = {str(x["event_key"]): x for x in load_jsonl(JOIN)}
    p16 = [x for x in obs if x.get("polarity") == "positive" and int(x.get("prefix", -1)) == 16]
    if len(p16) != 76:
        raise RuntimeError(f"expected 76 positive prefix16 records, found {len(p16)}")
    event_rows: list[dict[str, Any]] = []
    categories = Counter(); fold_summary: dict[int, Counter] = defaultdict(Counter)
    for r in p16:
        meta = join.get(str(r["event_key"]), {})
        roles = {}
        reasons = []
        for side in ("source", "target"):
            cat, reason = classify(r, side); roles[side] = cat; reasons.append(reason)
        # Event-level root cause is the first explicit causal issue in a fixed
        # order; both-side failures are retained as a composite label.
        if roles["source"] == "A_NO_PROPOSAL" or roles["target"] == "A_NO_PROPOSAL":
            overall = "A_NO_PROPOSAL"
        elif roles["source"].startswith("B_") or roles["target"].startswith("B_"):
            overall = "B_PROPOSAL_EXISTS_BUT_MAX_IOU_LT_05"
        elif roles["source"] == "E_SUPPORT_SELECTION_WRONG" or roles["target"] == "E_SUPPORT_SELECTION_WRONG":
            overall = "E_SUPPORT_SELECTION_WRONG"
        elif roles["source"] == "C_POOL_GOOD_BUT_ASSIGNED_0" or roles["target"] == "C_POOL_GOOD_BUT_ASSIGNED_0":
            overall = "C_POOL_GOOD_BUT_ASSIGNED_0"
        elif roles["source"].startswith("D_") or roles["target"].startswith("D_"):
            overall = "D_ASSIGNED_BUT_TRANSFORMED_IOU_LT_05"
        else:
            overall = "G_OTHER"
        # Event is already a frozen diagnostic record; copy evidence without
        # reducing denominator or joining held GT into any model input.
        src, tgt = side_evidence(r, "source"), side_evidence(r, "target")
        row = {
            "event_key": str(r["event_key"]), "model_event_uid": str(r["model_event_uid"]), "fold": int(r["fold"]),
            "category": meta.get("category"), "source_video": r.get("source_video"), "target_video": r.get("target_video"),
            "source_tracklet_key": r.get("source_tracklet_key"), "target_tracklet_key": r.get("target_tracklet_key"),
            "overall_failure": overall, "source_failure": roles["source"], "target_failure": roles["target"],
            "source": src, "target": tgt, "reason_evidence": reasons,
            "frozen_event_both_reliable": bool(r.get("both_reliable")), "frozen_event_source_reliable": bool(r.get("source_reliable")), "frozen_event_target_reliable": bool(r.get("target_reliable")),
            "pool_source_reliable": bool(src["max_iou"] >= IOU_T), "pool_target_reliable": bool(tgt["max_iou"] >= IOU_T),
            "pool_both_reliable": bool(src["max_iou"] >= IOU_T and tgt["max_iou"] >= IOU_T),
        }
        event_rows.append(row); categories[overall] += 1; fold_summary[int(r["fold"])][overall] += 1
    taxonomy_summary = {
        "schema_version": "trackocd.phase83.failure_taxonomy_summary.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "denominator": 76, "prefix": 16, "event_failure_counts": dict(sorted(categories.items())),
        "fold_failure_counts": {str(f): dict(sorted(c.items())) for f, c in sorted(fold_summary.items())},
        "pool_upper_bound": {
            "source_max_iou_ge_0.5": sum(x["pool_source_reliable"] for x in event_rows),
            "target_max_iou_ge_0.5": sum(x["pool_target_reliable"] for x in event_rows),
            "both_max_iou_ge_0.5": sum(x["pool_both_reliable"] for x in event_rows),
            "frozen_event_source_reliable": sum(x["frozen_event_source_reliable"] for x in event_rows),
            "frozen_event_target_reliable": sum(x["frozen_event_target_reliable"] for x in event_rows),
            "frozen_event_both_reliable": sum(x["frozen_event_both_reliable"] for x in event_rows),
        },
        "interpretation": "pool counts are post-hoc proposal upper-bound diagnostics; frozen event reliability remains the registered O comparator",
        "negative_denominator": 76, "public_dev_q1_sealed_accessed": False,
    }
    # Prefix/fold ceiling table uses the exact Phase75B records and denominator.
    by_prefix = []
    for p in (1, 2, 4, 8, 16):
        rs = [x for x in obs if x.get("polarity") == "positive" and int(x.get("prefix", -1)) == p]
        by_prefix.append({"prefix": p, "events": len(rs), "source_reliable": sum(bool(x.get("source_reliable")) for x in rs), "target_reliable": sum(bool(x.get("target_reliable")) for x in rs), "both_reliable": sum(bool(x.get("both_reliable")) for x in rs), "source_pool_iou_ge_0.5": sum(float(x["source"].get("max_iou", 0.0)) >= IOU_T for x in rs), "target_pool_iou_ge_0.5": sum(float(x["target"].get("max_iou", 0.0)) >= IOU_T for x in rs), "both_pool_iou_ge_0.5": sum(float(x["source"].get("max_iou", 0.0)) >= IOU_T and float(x["target"].get("max_iou", 0.0)) >= IOU_T for x in rs)})
    flat = []
    for x in event_rows:
        flat.append({"event_key": x["event_key"], "fold": x["fold"], "category": x["category"], "source_video": x["source_video"], "target_video": x["target_video"], "overall_failure": x["overall_failure"], "source_failure": x["source_failure"], "target_failure": x["target_failure"], "source_candidate_count": x["source"]["candidate_count"], "target_candidate_count": x["target"]["candidate_count"], "source_max_iou": x["source"]["max_iou"], "target_max_iou": x["target"]["max_iou"], "source_max_event_iou": x["source"]["event_iou_max"], "target_max_event_iou": x["target"]["event_iou_max"], "source_event_reliable": x["frozen_event_source_reliable"], "target_event_reliable": x["frozen_event_target_reliable"], "pool_source_reliable": x["pool_source_reliable"], "pool_target_reliable": x["pool_target_reliable"], "pool_both_reliable": x["pool_both_reliable"]})
    atomic_json(OUT / "audit/support_assignment_callgraph.json", callgraph())
    atomic_json(OUT / "audit/failure_taxonomy_76.json", {"schema_version": "trackocd.phase83.failure_taxonomy.v1", "events": event_rows, "public_dev_q1_sealed_accessed": False, "gt_used_only_posthoc_audit": True})
    atomic_json(OUT / "audit/failure_taxonomy_summary.json", taxonomy_summary)
    atomic_json(OUT / "audit/observability_by_prefix.json", {"prefix_rows": by_prefix, "denominator": 76, "source": str(OBS.resolve()), "source_sha256": sha(OBS)})
    atomic_json(OUT / "audit/pool_ceiling.json", taxonomy_summary["pool_upper_bound"])
    atomic_csv(OUT / "audit/failure_taxonomy_76.csv", flat, list(flat[0]))
    atomic_json(OUT / "status.json", {"phase": "Phase83", "status": "STAGE0_O_SUPPORT_AUDIT_COMPLETE", "next_action": "run Physical-to-R temporal mean raw R diagnostic and then TRAIN-only O router", "pool_ceiling": taxonomy_summary["pool_upper_bound"], "public_dev_q1_sealed_accessed": False, "resource_event": "read_only_cpu"})
    marker = {"status": "STAGE0_O_SUPPORT_AUDIT_COMPLETE", "files": [str(OUT / "audit/support_assignment_callgraph.json"), str(OUT / "audit/failure_taxonomy_76.json"), str(OUT / "audit/failure_taxonomy_summary.json")], "sha256": {str(OUT / "audit/failure_taxonomy_76.json"): sha(OUT / "audit/failure_taxonomy_76.json")}}
    atomic_json(OUT / "completion/stage0_audit.done", marker)
    print(json.dumps({"status": marker["status"], "pool_ceiling": taxonomy_summary["pool_upper_bound"], "failure_counts": taxonomy_summary["event_failure_counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
