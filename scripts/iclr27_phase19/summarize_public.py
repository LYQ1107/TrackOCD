"""Build the extended frozen-public metric table used by the Phase 19 report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def extended(scored: dict[str, Any], known: dict[str, Any] | None) -> dict[str, Any]:
    records = scored["records"]
    pos = [r for r in records if r["kind"] == "positive_existing"]
    neg = [r for r in records if r["kind"] == "negative_new"]
    post_rows = [(r, d) for r in records for d in r["target_decisions"][int(r["pre_prefix_rows"]):]]
    new_rows = [(r, d) for r, d in post_rows if d["action"] == "NEW"]
    existing_rows = [(r, d) for r, d in post_rows if d["action"] == "EXISTING"]
    # In a positive event the registered action is EXISTING; in a negative
    # paired event it is NEW.  These are event/row rates on the fixed public
    # development population, not training targets or selection signals.
    new_correct_rows = sum(1 for r, _ in new_rows if r["kind"] == "negative_new")
    new_first_correct = sum(1 for r in neg if r.get("first_commit_after_prefix") and
                            r["first_commit_after_prefix"]["action"] == "NEW")
    known = None if known is None else known["candidates"].get(scored["candidate"])
    strata = {}
    for label, subset in {
        "reliable_first_prefix": [r for r in pos if int(r["pre_prefix_rows"]) == 0],
        "unreliable_early_prefix": [r for r in pos if int(r["pre_prefix_rows"]) > 0],
    }.items():
        strata[label] = {
            "events": len(subset),
            "commit_ct_correct": int(sum(r["first_commit_correct"] for r in subset)),
            "commit_ct_recall": float(np.mean([r["first_commit_correct"] for r in subset])) if subset else None,
            "post_prefix_correct_rows": int(sum(r["post_prefix_correct_rows"] for r in subset)),
            "post_prefix_rows": int(sum(r["post_prefix_rows"] for r in subset)),
        }
    return {
        "candidate": scored["candidate"],
        "positive_events": len(pos), "negative_events": len(neg),
        "commit_ct_correct": int(sum(r["first_commit_correct"] for r in pos)),
        "commit_ct_eligible": len(pos),
        "post_prefix_correct_rows": int(sum(r["post_prefix_correct_rows"] for r in pos)),
        "post_prefix_rows": int(sum(r["post_prefix_rows"] for r in pos)),
        "existing_precision": float(sum(r["existing_correct_rows"] for r in records) /
                                    max(sum(r["existing_rows"] for r in records), 1)),
        "existing_recall_rows": float(sum(r["existing_correct_rows"] for r in pos) /
                                      max(sum(r["post_prefix_rows"] for r in pos), 1)),
        "new_precision_rows": float(new_correct_rows / max(len(new_rows), 1)),
        "new_recall_events": float(new_first_correct / max(len(neg), 1)),
        "new_predicted_rows": len(new_rows), "new_correct_rows": int(new_correct_rows),
        "negative_false_merge_count": int(sum(r["existing_rows"] > 0 for r in neg)),
        "negative_false_merge_rate": float(np.mean([r["existing_rows"] > 0 for r in neg])) if neg else 0.0,
        "duplicate_births": int(sum(r["duplicate_target_births"] for r in records)),
        "duplicate_event_rate": float(np.mean([r["duplicate_target_births"] > 0 for r in records])) if records else 0.0,
        "premature_rate": float(np.mean([r["premature_commit"] for r in records])) if records else 0.0,
        "deferral_rate_pre_prefix": float(sum(r["pre_prefix_defer_rows"] for r in records) /
                                           max(sum(r["pre_prefix_rows"] for r in records), 1)),
        "unresolved_rate": float(np.mean([r["unresolved"] for r in records])) if records else 0.0,
        "latency_mean_rows": (float(np.mean([r["latency"] for r in pos if r["latency"] is not None]))
                              if any(r["latency"] is not None for r in pos) else None),
        "category_coverage": len({r["target_category_evaluator_only"] for r in pos if r["first_commit_correct"]}),
        "video_coverage": len({r["target_video"] for r in pos if r["first_commit_correct"]}),
        "known_micro_accuracy": None if known is None else known["micro_accuracy"],
        "known_category_macro_accuracy": None if known is None else known["category_macro_accuracy"],
        "known_rows": None if known is None else known["rows"],
        "known_tracks": None if known is None else known["tracks"],
        "positive_prefix_strata": strata,
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--known", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); known = json.loads(a.known.read_text())
    result = {"protocol": "trackocd_iclr27_phase19_frozen_public_extended_metrics",
              "candidates": {}}
    for path in sorted(a.predictions.glob("*_scored.json")):
        x = json.loads(path.read_text()); result["candidates"][x["candidate"]] = extended(x, known)
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"); tmp.replace(a.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
