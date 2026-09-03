#!/usr/bin/env python3
"""Phase82P read-only wrapper for the frozen Phase75B strict O contract.

The historical evaluator is imported without invoking its ``main`` function,
so no Phase75B artifact is modified.  This wrapper writes a separate parity
record and requires the native Q0 prefix-16 result to remain exactly 25/76.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase82p"
OLD = ROOT / "scripts/iclr27_phase75b/run_observability.py"
PREFIXES = (1, 2, 4, 8, 16)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_old() -> Any:
    spec = importlib.util.spec_from_file_location("phase75b_observability_readonly", OLD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen evaluator: {OLD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    old = load_old()
    prepared, _ = old.load_event_rows()
    native, frame_keys = old.load_replay_lookup(prepared)
    by_prefix: dict[str, dict[str, Any]] = {}
    event_records: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        positives = []
        negatives = []
        for item in prepared:
            event = item["event"]
            source = old.evaluate_rows(item["source_rows"], native, frame_keys)
            target = old.evaluate_rows(item["target_rows"], native, frame_keys, prefix_rows=prefix)
            rec = {
                "event_key": event["event_key"],
                "polarity": "positive" if event["kind"] == "positive_existing" else "negative",
                "fold": int(event["fold"]),
                "prefix": prefix,
                "source_reliable": bool(source["joint_reliable_rows"] > 0),
                "target_reliable": bool(target["joint_reliable_rows"] > 0),
                "both_reliable": bool(source["joint_reliable_rows"] > 0 and target["joint_reliable_rows"] > 0),
                "source_candidate_rows": int(source["candidate_rows"]),
                "target_candidate_rows": int(target["candidate_rows"]),
                "source_max_iou": float(source["max_iou"]),
                "target_max_iou": float(target["max_iou"]),
                "strict_rule": "assigned == 1 AND transformed IoU >= 0.5 AND Q0 max IoU >= 0.5",
            }
            event_records.append(rec)
            (positives if rec["polarity"] == "positive" else negatives).append(rec)
        by_prefix[str(prefix)] = {
            "positive_events": len(positives),
            "negative_events": len(negatives),
            "positive_source_reliable": sum(r["source_reliable"] for r in positives),
            "positive_target_reliable": sum(r["target_reliable"] for r in positives),
            "positive_both_reliable": sum(r["both_reliable"] for r in positives),
            "negative_both_reliable": sum(r["both_reliable"] for r in negatives),
            "failure_reasons": dict(Counter(
                "both_unreliable" if not r["source_reliable"] and not r["target_reliable"] else
                "source_unreliable" if not r["source_reliable"] else
                "target_unreliable" if not r["target_reliable"] else "reliable"
                for r in positives
            )),
        }
    p16 = by_prefix["16"]
    expected = 25
    actual = int(p16["positive_both_reliable"])
    summary = {
        "schema_version": "trackocd.phase82p.strict_o_parity.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_evaluator": str(OLD),
        "prefixes": list(PREFIXES),
        "event_count": len(prepared),
        "positive_event_count": sum(r["event"]["kind"] == "positive_existing" for r in prepared),
        "negative_event_count": sum(r["event"]["kind"] == "negative_new" for r in prepared),
        "native_path": str(old.NATIVE),
        "frame_trace_path": str(old.FRAMES),
        "by_prefix": by_prefix,
        "p16_expected_positive_both_reliable": expected,
        "p16_actual_positive_both_reliable": actual,
        "parity_pass": actual == expected and len(prepared) == 152,
        "denominator": 76,
        "public_dev_q1_sealed_accessed": False,
        "future_rows_or_tracks": False,
        "model_labels_joined_before_inference": False,
        "physical_ids_used_as_model_input": False,
        "next_action": "build per-video TRAIN residual manifest" if actual == expected else "stop and repair evaluator/data contract",
    }
    atomic_json(OUT / "audit/strict_o_parity.json", summary)
    atomic_json(OUT / "audit/strict_o_event_records.json", event_records)
    (OUT / "completion/strict_o_parity.done").write_text("PASS 25/76\n" if summary["parity_pass"] else "FAIL\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["parity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
