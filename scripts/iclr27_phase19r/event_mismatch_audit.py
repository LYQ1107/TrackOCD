#!/usr/bin/env python
"""Attribute persistent-event errors in the frozen Phase19R main runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from src.iclr27_phase19r.data.stream import Phase19RData, OUT
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate, load_events


def classify(record: dict) -> list[str]:
    labels: list[str] = []
    first = record.get("first_commit") or {}
    action = first.get("action")
    if record.get("unresolved"):
        labels.append("unresolved_over_defer")
    if record.get("premature"):
        labels.append("premature_pre_prefix_commit")
    if action == "KNOWN":
        labels.append("known_misroute")
    if record.get("kind") == "negative_new" and record.get("negative_false_merge"):
        labels.append("false_merge_existing")
    if record.get("kind") == "negative_new" and action not in {"NEW", None} and not record.get("negative_false_merge"):
        labels.append("negative_not_new")
    if record.get("kind") == "positive_existing" and action == "EXISTING" and not record.get("first_commit_correct"):
        labels.append("wrong_existing_state")
    if record.get("kind") == "positive_existing" and action != "EXISTING":
        labels.append("positive_not_existing")
    if record.get("state_count", 0) >= 16:
        labels.append("state_capacity_pressure")
    if record.get("duplicate_target_births", 0) > 0:
        labels.append("duplicate_target_birth")
    source = record.get("source_decisions", [])
    if not any(x.get("action") in {"NEW", "EXISTING"} for x in source):
        labels.append("source_never_materialized")
    if not labels:
        labels.append("no_error")
    return labels


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--device", default="cpu"); p.add_argument("--out", type=Path, default=OUT / "audit/event_mismatch.json")
    a = p.parse_args(); device = torch.device(a.device)
    folds = []; aggregate = Counter(); event_rows = []
    for fold in range(4):
        data = Phase19RData(fold); ck = OUT / "checkpoints" / f"fold{fold}_best_internal.pt"
        result = evaluate_candidate("main", data, ck, device, load_events(fold))
        counts = Counter()
        for rec in result["records"]:
            labels = classify(rec); counts.update(labels)
            event_rows.append({"fold": fold, "event_key": rec["event_key"], "kind": rec["kind"],
                               "target_category": rec["target_category"], "first_action": (rec.get("first_commit") or {}).get("action"),
                               "first_commit_correct": bool(rec.get("first_commit_correct")), "labels": labels,
                               "state_count": rec.get("state_count"), "pre_prefix_rows": rec.get("pre_prefix_rows"),
                               "pre_prefix_defer_rows": rec.get("pre_prefix_defer_rows"), "duplicate_target_births": rec.get("duplicate_target_births")})
        aggregate.update(counts)
        folds.append({"fold": fold, "metrics": result["metrics"], "known_metrics": result["known_metrics"],
                      "error_counts": dict(counts), "events": len(result["records"])})
    payload = {"protocol": "trackocd_iclr27_phase19r_persistent_event_mismatch_audit_v1",
               "source": "fold*_best_internal.pt and held-known event manifests; no public labels",
               "folds": folds, "aggregate_error_counts": dict(aggregate), "events": event_rows,
               "interpretation": {"synthetic_episode_proxy": "existing_precision and proxy validation are not persistent-event outcomes",
                                  "primary_split_candidates": ["over-defer/unresolved", "wrong existing state", "false merge", "capacity pressure"]}}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); tmp.replace(a.out)
    print(json.dumps({"out": str(a.out), "aggregate_error_counts": dict(aggregate), "fold_metrics": [x["metrics"] for x in folds]}, indent=2), flush=True)


if __name__ == "__main__": main()

