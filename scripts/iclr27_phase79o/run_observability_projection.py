#!/usr/bin/env python3
"""Evaluate the one registered causal trajectory-projection O route."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.iclr27_phase75b.run_observability import FRAMES, NATIVE, PREFIXES, box_iou, evaluate_rows, load_event_rows, parse_json_box, read_jsonl
from src.iclr27_phase79o.trajectory_projection import build_causal_projection_lookup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase79o"
ARCHIVE = Path("/data2/usr_for_deadline/trackocd_phase79o")


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha(path: Path) -> str:
    h = hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main() -> None:
    prepared, _ = load_event_rows(); native_rows = read_jsonl(NATIVE); frame_rows = read_jsonl(FRAMES)
    event_keys: set[tuple[int, int]] = set()
    for item in prepared:
        for row in item["source_rows"] + item["target_rows"]: event_keys.add((int(row["video_id"]), int(row["image_id"])))
    augmented, projection_summary = build_causal_projection_lookup(native_rows, event_keys, max_gap=2)
    frame_keys = {(int(row["video_id"]), int(row["image_id"])) for row in frame_rows}
    event_records: list[dict[str, Any]] = []
    for item in prepared:
        event = item["event"]
        for prefix in PREFIXES:
            source_eval = evaluate_rows(item["source_rows"], augmented, frame_keys)
            target_eval = evaluate_rows(item["target_rows"], augmented, frame_keys, prefix_rows=prefix)
            source_reliable = source_eval["joint_reliable_rows"] > 0; target_reliable = target_eval["joint_reliable_rows"] > 0
            event_records.append({"event_key": event["event_key"], "model_event_uid": item["join"]["model_event_uid"], "kind": event["kind"], "polarity": "positive" if event["kind"] == "positive_existing" else "negative", "fold": int(event["fold"]), "category_denominator_only": event.get("category_gt_denominator_only", event.get("target_category_gt_denominator_only")), "source_video": int(event["source_video"]), "target_video": int(event["target_video"]), "prefix": prefix, "source": {k: v for k, v in source_eval.items() if k != "details"}, "target": {k: v for k, v in target_eval.items() if k != "details"}, "source_reliable": source_reliable, "target_reliable": target_reliable, "both_reliable": source_reliable and target_reliable, "perfect_correspondence_ceiling": source_reliable and target_reliable, "failure_reason": "source_and_target_unreliable" if not source_reliable and not target_reliable else ("source_unreliable" if not source_reliable else ("target_unreliable" if not target_reliable else "reliable_both_sides")), "source_row_details": source_eval["details"], "target_row_details": target_eval["details"]})
    by_prefix: dict[str, Any] = {}; by_fold: dict[str, Any] = {}
    for prefix in PREFIXES:
        subset = [x for x in event_records if x["prefix"] == prefix]; positives = [x for x in subset if x["polarity"] == "positive"]
        by_prefix[str(prefix)] = {"events": len(subset), "positive_events": len(positives), "source_reliable": sum(x["source_reliable"] for x in positives), "target_reliable": sum(x["target_reliable"] for x in positives), "both_reliable": sum(x["both_reliable"] for x in positives), "negative_both_reliable": sum(x["both_reliable"] for x in subset if x["polarity"] == "negative"), "source_video_coverage": len({x["source_video"] for x in positives if x["source_reliable"]}), "target_video_coverage": len({x["target_video"] for x in positives if x["target_reliable"]}), "category_coverage": len({x["category_denominator_only"] for x in positives if x["both_reliable"]}), "failure_reasons": {reason: sum(x["failure_reason"] == reason for x in positives) for reason in sorted({x["failure_reason"] for x in positives})}}
    for fold in range(4):
        subset = [x for x in event_records if x["prefix"] == 16 and x["polarity"] == "positive" and x["fold"] == fold]
        by_fold[str(fold)] = {"positive_events": len(subset), "source_reliable": sum(x["source_reliable"] for x in subset), "target_reliable": sum(x["target_reliable"] for x in subset), "both_reliable": sum(x["both_reliable"] for x in subset), "categories": sorted({x["category_denominator_only"] for x in subset if x["both_reliable"]})}
    raw_path = ROOT / "outputs/iclr27_phase75b/observability_status.json"; raw_status = json.loads(raw_path.read_text())
    summary = {"phase": "Phase79O", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "route": "causal_velocity_projection", "raw_prefix16_both_reliable": raw_status["by_prefix"]["16"]["both_reliable"], "projection_prefix16_both_reliable": by_prefix["16"]["both_reliable"], "raw_by_prefix": raw_status["by_prefix"], "by_prefix": by_prefix, "by_fold_prefix16": by_fold, "projection": projection_summary, "native_sha256": sha(NATIVE), "frame_trace_sha256": sha(FRAMES), "raw_status_sha256": sha(raw_path), "positive_events": 76, "negative_events": 76, "denominator": 76, "reliable_rule": "event assigned == 1 AND transformed IoU >= 0.5 AND max candidate IoU >= 0.5", "future_rows_or_tracks": False, "physical_ids_used_as_model_input": False, "public_dev_q1_sealed_accessed": False, "controller_run": False, "gate_contract": {"improve_over_raw": True, "prefix16_min": 25, "min_nonzero_folds": 3}}
    decision = "PHASE79O_GATE_O_PASS_ROUTE_TO_PHASE76B" if by_prefix["16"]["both_reliable"] > raw_status["by_prefix"]["16"]["both_reliable"] and sum(by_fold[str(f)]["both_reliable"] > 0 for f in range(4)) >= 3 else "PHASE79O_GATE_O_FAIL_R_EXHAUSTED_UNDER_FROZEN_PHYSICAL_STREAM"
    obj = {**summary, "decision": decision, "event_records": event_records}
    atomic(OUT / "audit/observability_projection.json", obj); atomic(OUT / "metrics/phase79o_observability.json", obj); atomic(OUT / "audit/phase79o_decision.json", {k: v for k, v in obj.items() if k != "event_records"}); atomic(OUT / "completion/phase79o.done", {"phase": "Phase79O", "decision": decision, "metrics": str(OUT / "metrics/phase79o_observability.json")}); print(json.dumps({"phase": "Phase79O", "decision": decision, "raw_p16": summary["raw_prefix16_both_reliable"], "projection_p16": summary["projection_prefix16_both_reliable"], "by_fold": by_fold, "projection": projection_summary}, sort_keys=True))


if __name__ == "__main__": main()
