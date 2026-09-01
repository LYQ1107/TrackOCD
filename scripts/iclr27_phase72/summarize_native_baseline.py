#!/usr/bin/env python3
"""Flatten and audit the frozen Phase19R native 152-event replay."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "outputs/iclr27_phase72/metrics/phase19r_raw_baseline.json"
EVENT_ROOT = ROOT / "outputs/iclr27_phase19r/manifests"
OUT_ROOT = ROOT / "outputs/iclr27_phase72/metrics"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ratio(num: int | float, den: int | float) -> dict[str, Any]:
    den_i = int(den)
    return {"numerator": num, "denominator": den_i, "value": (float(num) / den_i if den_i else None)}


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p + r == 0:
        return 0.0 if p is not None and r is not None else None
    return 2.0 * p * r / (p + r)


def first_action(rec: dict[str, Any]) -> dict[str, Any] | None:
    return rec.get("first_commit") if rec.get("first_commit") else None


def action_name(rec: dict[str, Any]) -> str | None:
    first = first_action(rec)
    return str(first.get("action")) if first else None


def action_position(row: dict[str, Any]) -> int | None:
    if not row:
        return None
    val = row.get("position", row.get("tracklet_position"))
    return int(val) if val is not None else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in records if r.get("kind") == "positive_existing"]
    neg = [r for r in records if r.get("kind") == "negative_new"]
    correct_ct = sum(bool(r.get("first_commit_correct")) for r in pos)
    post_rows = sum(int(r.get("post_prefix_rows", 0)) for r in pos)
    post_good = sum(int(r.get("post_prefix_correct_rows", 0)) for r in pos)
    existing_rows = sum(int(r.get("existing_rows", 0)) for r in records)
    existing_good = sum(int(r.get("existing_correct_rows", 0)) for r in records)
    new_pos = sum(action_name(r) == "NEW" for r in pos)
    new_neg = sum(action_name(r) == "NEW" for r in neg)
    first_positions = [action_position(first_action(r)) for r in records if first_action(r) and action_position(first_action(r)) is not None]
    false_commit = sum(action_name(r) in {"NEW", "EXISTING", "KNOWN"} for r in neg)
    false_merge = sum(bool(r.get("negative_false_merge")) for r in neg)
    defer_rows = sum(sum(str(x.get("action")) == "DEFER" for x in r.get("target_decisions", [])) for r in records)
    target_rows = sum(len(r.get("target_decisions", [])) for r in records)
    pre_defer = sum(int(r.get("pre_prefix_defer_rows", 0)) for r in records)
    pre_rows = sum(int(r.get("pre_prefix_rows", 0)) for r in records)
    premature = sum(bool(r.get("premature")) for r in records)
    unresolved = sum(bool(r.get("unresolved")) for r in records)
    duplicate_births = sum(int(r.get("duplicate_target_births", 0)) for r in records)

    pos_cats = sorted({int(r["target_category"]) for r in pos})
    pos_videos = sorted({int(r["target_video"]) for r in pos})
    correct_cats = sorted({int(r["target_category"]) for r in pos if r.get("first_commit_correct")})
    correct_videos = sorted({int(r["target_video"]) for r in pos if r.get("first_commit_correct")})

    # NMI/ARI preserve the complete positive denominator; an unresolved event
    # is represented by -1 rather than being silently dropped.
    labels = [int(r["target_category"]) for r in pos]
    pred = [int(first_action(r)["semantic_id"]) if first_action(r) and first_action(r).get("semantic_id") is not None else -1 for r in pos]
    nmi = float(normalized_mutual_info_score(labels, pred)) if len(set(labels)) > 1 else 0.0
    ari = float(adjusted_rand_score(labels, pred)) if len(set(labels)) > 1 else 0.0

    correct_delays: list[int] = []
    for r in pos:
        if not r.get("first_commit_correct"):
            continue
        p = action_position(first_action(r))
        prefix = int(next((e.get("target_first_reliable_prefix_index_gt_only", 0) for e in []), 0))
        # The event prefix is not copied into the replay record; use the
        # target pre-prefix row count, which is the same causal cutoff.
        prefix = int(r.get("pre_prefix_rows", 0))
        if p is not None:
            correct_delays.append(p - prefix)

    p_existing = existing_good / existing_rows if existing_rows else None
    r_existing = existing_good / post_rows if post_rows else None
    p_new = new_neg / max(new_neg + new_pos, 1) if (new_neg + new_pos) else None
    r_new = new_neg / len(neg) if neg else None
    return {
        "event_counts": {"positive": len(pos), "negative": len(neg), "total": len(records)},
        "commit_ct": ratio(correct_ct, len(pos)),
        "post_prefix_ct": ratio(post_good, post_rows),
        "existing_precision": ratio(existing_good, existing_rows),
        "existing_recall": ratio(existing_good, post_rows),
        "existing_f1": {"numerator": None, "denominator": None, "value": f1(p_existing, r_existing)},
        "new_precision": ratio(new_neg, new_neg + new_pos),
        "new_recall": ratio(new_neg, len(neg)),
        "new_f1": {"numerator": None, "denominator": None, "value": f1(p_new, r_new)},
        "negative_false_merge": ratio(false_merge, len(neg)),
        "negative_false_commit": ratio(false_commit, len(neg)),
        "premature_rate": ratio(premature, len(records)),
        "unresolved_rate": ratio(unresolved, len(records)),
        "defer_rate": ratio(defer_rows, target_rows),
        "pre_prefix_defer_rate": ratio(pre_defer, pre_rows),
        "first_action_position": {"numerator": float(sum(first_positions)) if first_positions else None, "denominator": len(first_positions), "value": float(np.mean(first_positions)) if first_positions else None},
        "assignment_delay_after_prefix": {"numerator": float(sum(correct_delays)) if correct_delays else None, "denominator": len(correct_delays), "value": float(np.mean(correct_delays)) if correct_delays else None},
        "duplicate_births": {"numerator": duplicate_births, "denominator": len(records), "value": float(duplicate_births / len(records)) if records else None},
        "fragmentation": {"numerator": duplicate_births, "denominator": len(records), "value": float(duplicate_births / len(records)) if records else None},
        "merge_error": ratio(false_merge, len(neg)),
        "nmi": {"numerator": None, "denominator": len(pos), "value": nmi},
        "ari": {"numerator": None, "denominator": len(pos), "value": ari},
        "category_coverage": ratio(len(correct_cats), len(pos_cats)),
        "video_coverage": ratio(len(correct_videos), len(pos_videos)),
        "category_ids": pos_cats,
        "covered_category_ids": correct_cats,
        "video_ids": pos_videos,
        "covered_video_ids": correct_videos,
        "first_action_counts": {
            f"{kind}:{action}": int(count)
            for (kind, action), count in sorted(
                Counter((r.get("kind"), action_name(r) or "UNRESOLVED") for r in records).items(),
                key=lambda x: str(x[0]),
            )
        },
    }


def main() -> None:
    payload = json.loads(BASELINE.read_text())
    records = [r for fold in payload["folds"] for r in fold["records"]]
    pos_manifest = [json.loads(x) for x in (EVENT_ROOT / "held_known_positive_events.jsonl").read_text().splitlines() if x.strip()]
    neg_manifest = [json.loads(x) for x in (EVENT_ROOT / "held_known_negative_events.jsonl").read_text().splitlines() if x.strip()]
    manifest_keys = {e["event_key"] for e in pos_manifest + neg_manifest}
    record_keys = [r["event_key"] for r in records]
    duplicate_keys = sorted(k for k, n in Counter(record_keys).items() if n > 1)
    by_fold: dict[str, Any] = {}
    for fold in range(4):
        fr = [r for r in records if int(r["fold"]) == fold]
        by_fold[str(fold)] = summarize_records(fr)

    # Validate the causal ordering without changing any replay output.
    chronology = []
    for r in records:
        src_pos = [int(x.get("tracklet_position", x.get("position", -1))) for x in r.get("source_decisions", [])]
        tgt_pos = [int(x.get("tracklet_position", x.get("position", -1))) for x in r.get("target_decisions", [])]
        chronology.append({"event_key": r["event_key"], "source_monotonic": src_pos == sorted(src_pos), "target_monotonic": tgt_pos == sorted(tgt_pos), "source_positions": src_pos, "target_positions": tgt_pos})
    event_contract = {
        "positive_manifest_count": len(pos_manifest),
        "negative_manifest_count": len(neg_manifest),
        "record_count": len(records),
        "expected_total": 152,
        "counts_match": len(pos_manifest) == 76 and len(neg_manifest) == 76 and len(records) == 152,
        "manifest_record_key_sets_equal": manifest_keys == set(record_keys),
        "duplicate_record_keys": duplicate_keys,
        "missing_record_keys": sorted(manifest_keys - set(record_keys)),
        "unexpected_record_keys": sorted(set(record_keys) - manifest_keys),
        "chronology_all_monotonic": all(x["source_monotonic"] and x["target_monotonic"] for x in chronology),
        "chronology_by_event": chronology,
    }
    optional = {
        "status": "NOT_APPLICABLE_INTERFACE_MISMATCH",
        "reason": "No strict per-track prediction exporter maps Q0/P71 TAO rows to TrackOCDEvaluator prediction_type/semantic_category_id/virtual_category_id fields.",
        "metrics": {k: None for k in [
            "supported_known_acc", "zero_shot_known_acc", "overall_known_acc", "known_to_novel_error",
            "known_misclassification_rate", "known_unresolved_rate", "novel_routing_recall",
            "novel_routing_precision", "false_known_absorption_rate", "unresolved_novel_rate",
            "route_aware_novel_acc", "conditional_novel_acc", "novel_only_nmi", "novel_only_ari",
            "macro_novel_class_acc", "predicted_novel_count", "novel_count_abs_error",
            "mean_fragmentation", "merge_error", "duplicate_creation_rate", "duplicate_avg_extra",
            "mean_assignment_delay", "all_track_acc", "macro_known_novel_harmonic",
        ]},
    }
    summary = {
        "protocol": "phase72_phase19r_native_frozen_baseline_summary_v1",
        "candidate": "phase19r_native_frozen_baseline_diagnostic",
        "input_baseline": str(BASELINE),
        "input_baseline_sha256": sha256(BASELINE),
        "event_contract": event_contract,
        "aggregate": summarize_records(records),
        "by_fold": by_fold,
        "optional_trackocd_evaluator": optional,
        "causal_inputs": {"future_rows_or_tracks": False, "held_gt_as_model_input": False, "category_text_or_id_feature": False, "physical_id_as_semantic_feature": False},
        "sealed_public_q1_accessed": False,
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    flat = OUT_ROOT / "phase19r_raw_event_records.jsonl"
    tmp_flat = flat.with_name(flat.name + ".tmp")
    with tmp_flat.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    tmp_flat.replace(flat)
    summary["event_records_jsonl"] = str(flat)
    summary["event_records_jsonl_sha256"] = sha256(flat)
    out = OUT_ROOT / "phase19r_raw_metrics_summary.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps({"summary": str(out), "event_records": str(flat), "events": len(records), "contract": event_contract}, indent=2))


if __name__ == "__main__":
    main()
