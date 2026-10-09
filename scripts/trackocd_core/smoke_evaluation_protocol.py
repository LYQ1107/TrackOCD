#!/usr/bin/env python3
"""Execute only synthetic evaluator fixtures, never dataset/model experiments."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_core.evaluation import (
    DecisionEvent, Target, TrackKey, evaluate_persistent, evaluate_standard,
    join_evaluation, seal_decisions,
)
from src.trackocd_v2.io import atomic_json


def run() -> dict:
    started = time.monotonic()
    toy_orders = ((1, 2, 3, 4), (3, 1, 4, 2), (4, 3, 2, 1), (2, 4, 1, 3))
    prefixes = (1, 2, 4, 8, 16)
    targets = tuple(Target(TrackKey(v, "7"), 10, "novel") for v in range(1, 5))
    by_video = {t.key.video_id: t for t in targets}
    checked = []
    for order_index, order in enumerate(toy_orders):
        for prefix in prefixes:
            events = tuple(DecisionEvent(i, by_video[v].key, min(prefix, 2),
                                         "NEW" if i == 0 else "EXISTING", token="x")
                           for i, v in enumerate(order))
            replay = seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=(1, 2))
            joined = join_evaluation(replay, targets, {t.key: t.key for t in targets})
            standard, persistent = evaluate_standard(joined), evaluate_persistent(joined)
            if standard["new_acc"] != 1 or persistent["correct_commit_ct"] != 1:
                raise AssertionError("Constructed pure fixture must have exact expected counts")
            if persistent["commit_ct_denominator"] != 3 or len(replay.created_tokens) != 1:
                raise AssertionError("Independent replay denominator/token mismatch")
            checked.append({"toy_order_index": order_index, "prefix_cap": prefix,
                            "fixed_opportunities": 3, "expected_correct_reuse_count": 3})
    a, b = targets[:2]
    missing = join_evaluation(seal_decisions([DecisionEvent(0, b.key, 1, "NEW", token="x")],
                                            video_order=(1, 2), prefix_cap=1, known_ids=(1, 2)),
                              [a, b], {b.key: b.key})
    missing_result = evaluate_persistent(missing)
    if missing_result["commit_ct_denominator"] != 1 or missing_result["false_split_new_count"] != 1:
        raise AssertionError("Missing source must not remove GT reuse eligibility")
    files = ["configs/trackocd_core/evaluation_protocol.json", "tests/trackocd_core/test_evaluation_protocol.py",
             "scripts/trackocd_core/smoke_evaluation_protocol.py"]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src/trackocd_core/evaluation").glob("*.py"))]
    return {
        "schema_version": "trackocd.core.evaluation_smoke.v1",
        "status": "PASS_ENGINEERING_SYNTHETIC_ONLY",
        "primary_predicted_protocol_complete": False,
        "toy_grid_cases": checked,
        "toy_grid_cases_passed": len(checked),
        "toy_orders_are_not_real_TAO_order_artifacts": True,
        "preconstructed_correct_events_are_not_learned_model_results": True,
        "missing_source_fixture": {"fixed_opportunities": 1, "false_split_new_count": 1},
        "file_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        "worker_count": 1, "gpu_used": False,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_seconds": time.monotonic() - started,
        "GT_annotation_or_dataset_opened": False,
        "training_or_model_inference_started": False,
        "test_accessed": False,
        "external_process_interference": False,
        "scientific_hypothesis_tested": False,
    }


if __name__ == "__main__":
    result = run()
    atomic_json(ROOT / "outputs/trackocd_core/audit/evaluation_protocol_smoke.json", result)
    print(json.dumps({key: result[key] for key in ("status", "toy_grid_cases_passed", "peak_rss_kib", "wall_seconds")}))
