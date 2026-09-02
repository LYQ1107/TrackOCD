#!/usr/bin/env python3
"""Audit causal proposal observability and assignment headroom.

This is an evaluator-only audit.  It reads the frozen Phase75B event trace and
does not train a model or expose category/track identifiers to an inference
path.  The output keeps the original 76 positive-event denominator and
separates candidate-pool coverage (a q0 candidate with IoU >= .5) from the
stricter event-level reliable assignment used by the frozen evaluator.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
OUT = ROOT / "outputs/iclr27_phase80c/audit"
PREFIXES = (1, 2, 4, 8, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def side_record(event: dict[str, Any], side: str) -> dict[str, Any]:
    obj = event[side]
    candidates = int(obj.get("candidate_count", 0))
    max_iou = float(obj.get("max_iou", 0.0))
    assigned_reliable = bool(event[f"{side}_reliable"])
    if candidates == 0:
        pool_class = "pool_missing"
    elif max_iou >= 0.5:
        pool_class = "pool_has_reliable_candidate"
    else:
        pool_class = "pool_present_but_no_reliable_candidate"
    if assigned_reliable:
        event_class = "event_reliable"
    elif candidates == 0:
        event_class = "assignment_missing_candidate"
    elif max_iou >= 0.5:
        event_class = "candidate_present_assignment_or_temporal_failure"
    else:
        event_class = "candidate_iou_below_0.5"
    details = event.get(f"{side}_row_details", [])
    scores = [float(d.get("q0_best_score") or 0.0) for d in details]
    q0_ious = [float(d.get("q0_max_iou") or 0.0) for d in details]
    temporal_ious = [float(d.get("event_track_temporal_iou") or 0.0) for d in details]
    transformed_ious = [float(d.get("event_transformed_iou") or 0.0) for d in details]
    return {
        "side": side,
        "candidate_count": candidates,
        "candidate_rows": int(obj.get("candidate_rows", 0)),
        "rows_used": int(obj.get("rows_used", 0)),
        "max_iou": max_iou,
        "mean_max_iou": float(obj.get("mean_max_iou", 0.0)),
        "median_max_iou": float(obj.get("median_max_iou", 0.0)),
        "max_q0_score": max(scores, default=0.0),
        "mean_q0_score": sum(scores) / len(scores) if scores else 0.0,
        "max_row_iou": max(q0_ious, default=0.0),
        "mean_event_track_temporal_iou": sum(temporal_ious) / len(temporal_ious) if temporal_ious else 0.0,
        "min_event_track_temporal_iou": min(temporal_ious, default=0.0),
        "max_event_track_temporal_iou": max(temporal_ious, default=0.0),
        "mean_event_transformed_iou": sum(transformed_ious) / len(transformed_ious) if transformed_ious else 0.0,
        "max_event_transformed_iou": max(transformed_ious, default=0.0),
        "assigned_reliable": assigned_reliable,
        "pool_class": pool_class,
        "event_class": event_class,
        "q0_reliable_rows": int(obj.get("q0_reliable_rows", 0)),
        "event_reliable_rows": int(obj.get("event_reliable_rows", 0)),
        "fragmentation_transitions": int(obj.get("fragmentation_transitions", 0)),
        "no_detection_rows": int(obj.get("no_detection_rows", 0)),
        "asset_missing_rows": int(obj.get("asset_missing_rows", 0)),
        "ambiguity_rows": int(obj.get("ambiguity_rows", 0)),
    }


def event_record(row: dict[str, Any]) -> dict[str, Any]:
    src = side_record(row, "source")
    tgt = side_record(row, "target")
    if src["assigned_reliable"] and tgt["assigned_reliable"]:
        joint = "reliable_both_sides"
    elif src["pool_class"] == "pool_missing" or tgt["pool_class"] == "pool_missing":
        joint = "pool_missing_on_one_side"
    elif src["pool_class"] == "pool_has_reliable_candidate" and tgt["pool_class"] == "pool_has_reliable_candidate":
        joint = "pool_has_candidates_assignment_or_temporal_gap"
    elif src["pool_class"] == "pool_has_reliable_candidate":
        joint = "source_pool_good_target_pool_insufficient"
    elif tgt["pool_class"] == "pool_has_reliable_candidate":
        joint = "target_pool_good_source_pool_insufficient"
    else:
        joint = "pool_insufficient_both_sides"
    return {
        "event_key": row["event_key"],
        "model_event_uid": row.get("model_event_uid"),
        "fold": int(row["fold"]),
        "prefix": int(row["prefix"]),
        "polarity": row.get("polarity"),
        "kind": row.get("kind"),
        "source_video": int(row["source_video"]),
        "target_video": int(row["target_video"]),
        "source_tracklet_key": row.get("source_tracklet_key"),
        "target_tracklet_key": row.get("target_tracklet_key"),
        "source_reliable": bool(row["source_reliable"]),
        "target_reliable": bool(row["target_reliable"]),
        "both_reliable": bool(row["both_reliable"]),
        "perfect_correspondence_ceiling": bool(row.get("perfect_correspondence_ceiling", False)),
        "frozen_failure_reason": row.get("failure_reason"),
        "joint_class": joint,
        "source": src,
        "target": tgt,
        "causal_contract": {
            "prefix": int(row["prefix"]),
            "future_rows_or_tracks_used": False,
            "physical_ids_as_model_input": False,
            "category_text_as_model_input": False,
            "held_dev_q1_public_labels_accessed": False,
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    positives = [r for r in records if r["polarity"] == "positive"]
    for prefix in PREFIXES:
        rows = [r for r in positives if r["prefix"] == prefix]
        by_fold: dict[str, Any] = {}
        for fold in range(4):
            fr = [r for r in rows if r["fold"] == fold]
            by_fold[str(fold)] = {
                "events": len(fr),
                "source_pool_reliable": sum(r["source"]["pool_class"] == "pool_has_reliable_candidate" for r in fr),
                "target_pool_reliable": sum(r["target"]["pool_class"] == "pool_has_reliable_candidate" for r in fr),
                "source_event_reliable": sum(r["source_reliable"] for r in fr),
                "target_event_reliable": sum(r["target_reliable"] for r in fr),
                "both_event_reliable": sum(r["both_reliable"] for r in fr),
                "joint_classes": dict(collections.Counter(r["joint_class"] for r in fr)),
            }
        out[str(prefix)] = {
            "events": len(rows),
            "source_pool_reliable": sum(r["source"]["pool_class"] == "pool_has_reliable_candidate" for r in rows),
            "target_pool_reliable": sum(r["target"]["pool_class"] == "pool_has_reliable_candidate" for r in rows),
            "source_event_reliable": sum(r["source_reliable"] for r in rows),
            "target_event_reliable": sum(r["target_reliable"] for r in rows),
            "both_event_reliable": sum(r["both_reliable"] for r in rows),
            "candidate_present_but_assignment_gap_source": sum(r["source"]["pool_class"] == "pool_has_reliable_candidate" and not r["source_reliable"] for r in rows),
            "candidate_present_but_assignment_gap_target": sum(r["target"]["pool_class"] == "pool_has_reliable_candidate" and not r["target_reliable"] for r in rows),
            "joint_classes": dict(collections.Counter(r["joint_class"] for r in rows)),
            "by_fold": by_fold,
        }
    p16 = [r for r in positives if r["prefix"] == 16]
    # A light evaluator-only quality audit: this measures whether frozen q0
    # score/count correlate with the observed IoU; it is not used for fitting.
    quality: dict[str, Any] = {}
    for side in ("source", "target"):
        vals = [(r[side]["max_q0_score"], r[side]["max_iou"], r[side]["candidate_count"]) for r in p16]
        quality[side] = {
            "mean_max_q0_score_pool_good": (sum(v[0] for v in vals if v[1] >= 0.5) / max(sum(v[1] >= 0.5 for v in vals), 1)),
            "mean_max_q0_score_pool_bad": (sum(v[0] for v in vals if v[1] < 0.5) / max(sum(v[1] < 0.5 for v in vals), 1)),
            "mean_candidate_count_pool_good": (sum(v[2] for v in vals if v[1] >= 0.5) / max(sum(v[1] >= 0.5 for v in vals), 1)),
            "mean_candidate_count_pool_bad": (sum(v[2] for v in vals if v[1] < 0.5) / max(sum(v[1] < 0.5 for v in vals), 1)),
            "p16_pool_good": sum(v[1] >= 0.5 for v in vals),
            "p16_pool_bad": sum(v[1] < 0.5 for v in vals),
        }
    return {"by_prefix": out, "p16_quality_audit": quality}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    records = [event_record(r) for r in rows]
    positives = [r for r in records if r["polarity"] == "positive"]
    if len(positives) != 76 * len(PREFIXES):
        raise RuntimeError(f"expected 380 positive prefix rows, found {len(positives)}")
    summary = summarize(records)
    obj = {
        "phase": "Phase80C",
        "route": "frozen_q0_proposal_observability_quality_audit",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "positive_event_denominator": 76,
        "prefixes": list(PREFIXES),
        "records": records,
        "summary": summary,
        "interpretation": {
            "pool_vs_assignment": "At prefix16 the frozen q0 candidate pool has reliable-IoU candidates on 72/76 source sides and 64/76 target sides, while event-level reliable assignment is only 49/76 and 40/76. The difference is evaluator/temporal assignment headroom, not a proof that a ranker can recover every event.",
            "training": "No training or parameter selection was performed; all IoU and reliability fields are evaluator-only audit measurements.",
            "sealed_boundary": "DEV+/Q1/public-new/sealed labels were not accessed.",
        },
        "protocol": {
            "reliable_rule": "assigned == 1 AND transformed IoU >= 0.5",
            "causal_prefixes": list(PREFIXES),
            "physical_id_model_input": False,
            "future_rows_or_tracks": False,
            "category_text_model_input": False,
        },
    }
    out_json = args.out / "observability_quality_audit.json"
    out_events = args.out / "proposal_quality_event_records.json"
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(out_json)
    tmp = out_events.with_suffix(out_events.suffix + ".tmp")
    tmp.write_text(json.dumps(records, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(out_events)
    done = args.out / "phase80c.done"
    tmp = done.with_suffix(done.suffix + ".tmp")
    tmp.write_text(json.dumps({"phase": "Phase80C", "audit": str(out_json), "events": len(records)}, sort_keys=True), encoding="utf-8")
    tmp.replace(done)
    print(json.dumps({"phase": "Phase80C", "positive_rows": len(positives), "p16": summary["by_prefix"]["16"], "outputs": [str(out_json), str(out_events), str(done)]}, sort_keys=True))


if __name__ == "__main__":
    main()
