#!/usr/bin/env python3
"""Run the unchanged Phase19R controller with the original DINOv2 features.

The Phase26 source proposal is frozen as a provenance comparator; this
diagnostic does not regenerate proposals.  All decisions are made by the
unchanged Phase19R persistent evaluator and controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate, load_events

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase28"
PREFIXES = (1, 2, 4, 8, 16)


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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def correct_decision(decision: dict[str, Any], rec: dict[str, Any], event: dict[str, Any], states: dict[int, dict[str, Any]]) -> bool:
    if decision.get("action") != "EXISTING" or decision.get("semantic_id") is None:
        return False
    state = states.get(int(decision["semantic_id"]))
    # ``simulate`` has already normalized positive/negative manifest field
    # names into the evaluator-side target_category; use that parsed value
    # rather than indexing a condition-specific raw manifest key.
    target_cat = int(rec["target_category"])
    return bool(state and state.get("oracle_birth_category") == target_cat
                and int(state.get("birth_video", -1)) != int(event["target_video"])
                and str(state.get("birth_track", "")) != str(event["target_tracklet_key"]))


def prefix_event_snapshot(rec: dict[str, Any], event: dict[str, Any], prefix: int) -> dict[str, Any]:
    decisions = list(rec.get("target_decisions", []))
    states = {int(x["sid"]): x for x in rec.get("states", [])}
    start = min(int(prefix), len(decisions))
    tail = decisions[start:]
    first = next((x for x in tail if x.get("action") != "DEFER"), None)
    good = bool(first and correct_decision(first, rec, event, states))
    existing = [x for x in tail if x.get("action") == "EXISTING"]
    return {
        "prefix": int(prefix),
        "target_rows": len(decisions),
        "post_prefix_rows": len(tail),
        "first_action": None if first is None else first.get("action"),
        "first_semantic_id": None if first is None else first.get("semantic_id"),
        "correct_commit": good,
        "existing_rows": len(existing),
        "existing_correct_rows": int(sum(correct_decision(x, rec, event, states) for x in existing)),
        "negative_false_merge": bool(event["kind"] == "negative_new" and first and first.get("action") == "EXISTING"),
        "unresolved": first is None,
        "premature": any(x.get("action") != "DEFER" for x in decisions[:start]),
    }


def compact_record(rec: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_key": rec["event_key"],
        "kind": rec["kind"],
        "fold": int(rec["fold"]),
        "target_category": int(rec["target_category"]),
        "target_video": int(rec["target_video"]),
        "source_tracklet_keys": list(event["source_tracklet_keys"]),
        "target_tracklet_key": str(event["target_tracklet_key"]),
        "registered_prefix": int(event["target_first_reliable_prefix_index_gt_only"]),
        "first_commit": rec.get("first_commit"),
        "first_commit_correct": bool(rec.get("first_commit_correct")),
        "negative_false_merge": bool(rec.get("negative_false_merge")),
        "duplicate_target_births": int(rec.get("duplicate_target_births", 0)),
        "premature": bool(rec.get("premature")),
        "unresolved": bool(rec.get("unresolved")),
        "state_count": int(rec.get("state_count", 0)),
        "target_actions": [x.get("action") for x in rec.get("target_decisions", [])],
        "prefix_snapshots": {str(p): prefix_event_snapshot(rec, event, p) for p in PREFIXES},
    }


def prefix_aggregate(records: list[dict[str, Any]], prefix: int) -> dict[str, Any]:
    pos = [r for r in records if r["kind"] == "positive_existing"]
    neg = [r for r in records if r["kind"] == "negative_new"]
    snapshots = [r["prefix_snapshots"][str(prefix)] for r in records]
    ps = [r["prefix_snapshots"][str(prefix)] for r in pos]
    ns = [r["prefix_snapshots"][str(prefix)] for r in neg]
    correct = [r for r, s in zip(pos, ps) if s["correct_commit"]]
    by_cat = defaultdict(list); by_video = defaultdict(list)
    for r, s in zip(pos, ps):
        by_cat[int(r["target_category"])].append(int(s["correct_commit"]))
        by_video[int(r["target_video"])].append(int(s["correct_commit"]))
    ex_total = sum(s["existing_rows"] for s in snapshots); ex_good = sum(s["existing_correct_rows"] for s in snapshots)
    return {
        "prefix": int(prefix),
        "positive_events": len(pos),
        "negative_events": len(neg),
        "commit_ct": {"correct": len(correct), "eligible": len(pos), "recall": len(correct) / max(len(pos), 1)},
        "category_coverage": sum(any(v) for v in by_cat.values()),
        "video_coverage": sum(any(v) for v in by_video.values()),
        "existing_precision": ex_good / max(ex_total, 1),
        "existing_recall": ex_good / max(sum(s["post_prefix_rows"] for s in ps), 1),
        "negative_false_merge_rate": float(np.mean([s["negative_false_merge"] for s in ns])) if ns else 0.0,
        "unresolved_rate": float(np.mean([s["unresolved"] for s in snapshots])) if snapshots else 0.0,
        "premature_rate": float(np.mean([s["premature"] for s in snapshots])) if snapshots else 0.0,
        "duplicate_births": int(sum(r["duplicate_target_births"] for r in records)),
        "failure_event_keys": [r["event_key"] for r, s in zip(pos, ps) if not s["correct_commit"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    fold_results: list[dict[str, Any]] = []
    compact_all: list[dict[str, Any]] = []
    for fold in range(4):
        data = Phase19RData(fold)
        checkpoint = OUT / "checkpoints" / f"controller_f{fold}_best_internal.pt"
        main_result = evaluate_candidate("main", data, checkpoint, device)
        raw_result = evaluate_candidate("raw", data, None, device)
        events = load_events(fold)
        by_key = {str(e["event_key"]): e for e in events}
        main_compact = [compact_record(rec, by_key[rec["event_key"]]) for rec in main_result["records"]]
        raw_compact = [compact_record(rec, by_key[rec["event_key"]]) for rec in raw_result["records"]]
        fold_results.append({
            "fold": fold,
            "main": {"metrics": main_result["metrics"], "known_metrics": main_result["known_metrics"], "events": len(main_result["records"]), "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint)},
            "raw_persistent_comparator": {"metrics": raw_result["metrics"], "events": len(raw_result["records"])},
            "main_prefix_diagnostics": {str(p): prefix_aggregate(main_compact, p) for p in PREFIXES},
            "raw_prefix_diagnostics": {str(p): prefix_aggregate(raw_compact, p) for p in PREFIXES},
            "positive_events": sum(x["kind"] == "positive_existing" for x in main_compact),
            "negative_events": sum(x["kind"] == "negative_new" for x in main_compact),
        })
        compact_all.extend([{**x, "condition": "main"} for x in main_compact])
        compact_all.extend([{**x, "condition": "raw_persistent_comparator"} for x in raw_compact])

    positives = [x for x in compact_all if x["condition"] == "main" and x["kind"] == "positive_existing"]
    if len(positives) != 76:
        raise RuntimeError(f"positive event denominator changed: {len(positives)}")
    main_metrics = [x["main"]["metrics"] for x in fold_results]
    raw_metrics = [x["raw_persistent_comparator"]["metrics"] for x in fold_results]
    aggregate = {
        "protocol": "trackocd_iclr27_phase28_frozen_representation_compatibility",
        "positive_event_denominator": 76,
        "folds": fold_results,
        "main_aggregate": {
            "commit_ct_correct": int(sum(x["commit_ct"]["correct"] for x in main_metrics)),
            "commit_ct_eligible": int(sum(x["commit_ct"]["eligible"] for x in main_metrics)),
            "category_coverage_sum": int(sum(x["category_coverage"] for x in main_metrics)),
            "video_coverage_sum": int(sum(x["video_coverage"] for x in main_metrics)),
            "existing_precision_mean": float(np.mean([x["existing_precision"] for x in main_metrics])),
            "existing_recall_mean": float(np.mean([x["existing_recall"] for x in main_metrics])),
            "negative_false_merge_mean": float(np.mean([x["negative_false_merge_rate"] for x in main_metrics])),
            "duplicate_births": int(sum(x["duplicate_births"] for x in main_metrics)),
            "premature_rate_mean": float(np.mean([x["premature_rate"] for x in main_metrics])),
            "unresolved_rate_mean": float(np.mean([x["unresolved_rate"] for x in main_metrics])),
            "known_micro_mean": float(np.mean([x.get("known_micro", 0.0) for x in [f["main"]["known_metrics"] for f in fold_results]])),
            "known_macro_mean": float(np.mean([x.get("known_macro", 0.0) for x in [f["main"]["known_metrics"] for f in fold_results]])),
        },
        "raw_aggregate": {
            "commit_ct_correct": int(sum(x["commit_ct"]["correct"] for x in raw_metrics)),
            "commit_ct_eligible": int(sum(x["commit_ct"]["eligible"] for x in raw_metrics)),
            "existing_precision_mean": float(np.mean([x["existing_precision"] for x in raw_metrics])),
            "existing_recall_mean": float(np.mean([x["existing_recall"] for x in raw_metrics])),
            "negative_false_merge_mean": float(np.mean([x["negative_false_merge_rate"] for x in raw_metrics])),
            "duplicate_births": int(sum(x["duplicate_births"] for x in raw_metrics)),
        },
        "historical_commit_ct_comparator": "Phase19R/Phase20 2/76",
        "proposal_frozen": True,
        "representation": "original Phase19R normalized fused DINOv2 CLS/ROI",
        "controller_frozen": True,
        "threshold_sweep": False,
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "physical/semantic IDs as features", "semantic text", "held GT as model input"],
    }
    atomic_json(OUT / "metrics/frozen_baseline_persistent.json", aggregate)
    atomic_json(OUT / "metrics/frozen_baseline_prefix_diagnostics.json", {"protocol": aggregate["protocol"], "prefixes": list(PREFIXES), "folds": [{"fold": f["fold"], "main": f["main_prefix_diagnostics"], "raw": f["raw_prefix_diagnostics"]} for f in fold_results]})
    atomic_json(OUT / "audit/frozen_baseline_event_records.json", {"protocol": aggregate["protocol"], "positive_denominator": 76, "records": positives})
    (OUT / "completion/compatibility.done").write_text(json.dumps({"positive_events": 76, "main_commit_ct": aggregate["main_aggregate"]["commit_ct_correct"], "raw_commit_ct": aggregate["raw_aggregate"]["commit_ct_correct"]}, sort_keys=True) + "\n")
    print(json.dumps({"main": aggregate["main_aggregate"], "raw": aggregate["raw_aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
