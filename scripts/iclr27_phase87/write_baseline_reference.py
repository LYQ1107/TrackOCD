#!/usr/bin/env python3
"""Record the pre-C0 comparator without making a new model selection path."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "iclr27_phase87" / "metrics" / "phase19r_baseline_reference.json"


def lines(path: pathlib.Path) -> list[dict]:
    return [json.loads(x) for x in path.open() if x.strip()]


def main() -> None:
    rows = []
    for fold in range(4):
        events = lines(ROOT / "outputs" / "iclr27_phase87" / "manifests" / f"val_events_f{fold}.jsonl")
        positive = sum(e.get("polarity") == "positive" for e in events)
        negative = len(events) - positive
        rows.append({"fold": fold, "events": len(events), "positive_events": positive, "negative_events": negative, "defer_all_commit_ct": 0, "defer_all_negative_false_merge": 0.0})
    # The exact Phase19R persistent event comparator is retained by hash.  Its
    # event schema is not interchangeable with these newly materialized TRAIN
    # validation events, so this reference is never used for checkpoint choice.
    historical = ROOT / "outputs" / "iclr27_phase72" / "metrics" / "phase19r_raw_baseline.json"
    result = {"schema_version": "trackocd.phase87.baseline_reference.v1", "phase": 87, "protocol": "TRAIN_VALIDATION_BASELINE_BEFORE_C0", "rows": rows, "comparable_policy": "all_defer_safe_fallback", "phase19r_historical_path": str(historical), "phase19r_historical_exists": historical.exists(), "phase19r_event_protocol_equal": False, "diagnostic_only": True, "checkpoint_selection_used": False, "public_dev_q1_sealed_accessed": False}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
