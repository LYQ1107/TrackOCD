#!/usr/bin/env python3
"""Unit tests for the TrackOCD-v1.0 corrected evaluator (9 protocol cases)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def gt_rows(role_map):
    rows = []
    for sid, (cat, role) in role_map.items():
        rows.append(
            {
                "sample_id": sid,
                "ground_truth_category_id": cat,
                "protocol_role": role,
            }
        )
    return rows


def preds(plist):
    return [
        {"sample_id": sid, **p}
        for sid, p in plist
    ]


def run_case(name, role_map, plist, checks, subset_ids=None):
    ev = TrackOCDEvaluator(gt_rows(role_map))
    res = ev.evaluate(preds(plist), subset_ids=subset_ids)
    ok = all(abs(res[k] - v) < 1e-9 for k, v in checks.items())
    return {
        "case": name,
        "passed": ok,
        "checks": {k: res[k] for k in checks},
        "expected": checks,
        "details": {k: res[k] for k in (
            "supported_known_acc", "zero_shot_known_acc", "overall_known_acc",
            "route_aware_novel_acc", "conditional_novel_acc",
            "novel_routing_recall", "false_known_absorption_rate",
            "unresolved_novel_rate", "merge_error", "mean_fragmentation",
            "duplicate_creation_rate", "known_to_novel_error",
            "known_misclassification_rate",
        )},
    }


def main():
    report = []

    # Case 1: known fully correct
    report.append(run_case(
        "case1_known_correct",
        {"a": (12, "supported_known"), "b": (12, "supported_known")},
        [("a", {"prediction_type": "known", "semantic_category_id": 12}),
         ("b", {"prediction_type": "known", "semantic_category_id": 12})],
        {"supported_known_acc": 1.0, "overall_known_acc": 1.0},
    ))

    # Case 2: wrong known semantic id must not be rescued by Hungarian
    report.append(run_case(
        "case2_known_wrong_id",
        {"a": (12, "supported_known"), "b": (34, "supported_known")},
        [("a", {"prediction_type": "known", "semantic_category_id": 34}),
         ("b", {"prediction_type": "known", "semantic_category_id": 12})],
        {"supported_known_acc": 0.0, "known_misclassification_rate": 1.0},
    ))

    # Case 3: novel cluster permutation is resolved by Hungarian
    report.append(run_case(
        "case3_novel_permutation",
        {"a": (101, "novel"), "b": (102, "novel"), "c": (101, "novel")},
        [("a", {"prediction_type": "novel", "virtual_category_id": 7}),
         ("b", {"prediction_type": "novel", "virtual_category_id": 2}),
         ("c", {"prediction_type": "novel", "virtual_category_id": 7})],
        {"route_aware_novel_acc": 1.0, "conditional_novel_acc": 1.0,
         "novel_routing_recall": 1.0},
    ))

    # Case 4: novel predicted as known must be an error and never enter Hungarian
    report.append(run_case(
        "case4_novel_as_known",
        {"a": (101, "novel"), "b": (102, "novel")},
        [("a", {"prediction_type": "known", "semantic_category_id": 12}),
         ("b", {"prediction_type": "novel", "virtual_category_id": 5})],
        {"false_known_absorption_rate": 0.5, "route_aware_novel_acc": 0.5,
         "conditional_novel_acc": 1.0, "novel_routing_recall": 0.5},
    ))

    # Case 5: known predicted as novel must be an error
    report.append(run_case(
        "case5_known_as_novel",
        {"a": (12, "supported_known"), "b": (34, "supported_known")},
        [("a", {"prediction_type": "novel", "virtual_category_id": 9}),
         ("b", {"prediction_type": "known", "semantic_category_id": 34})],
        {"known_to_novel_error": 0.5, "supported_known_acc": 0.5},
    ))

    # Case 6: two novel classes merged into one virtual id
    report.append(run_case(
        "case6_novel_merge",
        {"a": (101, "novel"), "b": (102, "novel")},
        [("a", {"prediction_type": "novel", "virtual_category_id": 1}),
         ("b", {"prediction_type": "novel", "virtual_category_id": 1})],
        {"merge_error": 1.0, "route_aware_novel_acc": 0.5},
    ))

    # Case 7: one novel class split into multiple virtual ids
    report.append(run_case(
        "case7_novel_fragmentation",
        {"a": (101, "novel"), "b": (101, "novel"), "c": (101, "novel")},
        [("a", {"prediction_type": "novel", "virtual_category_id": 1}),
         ("b", {"prediction_type": "novel", "virtual_category_id": 2}),
         ("c", {"prediction_type": "novel", "virtual_category_id": 3})],
        {"mean_fragmentation": 3.0, "duplicate_creation_rate": 1.0,
         "route_aware_novel_acc": 1.0 / 3.0},
    ))

    # Case 8: unresolved counts as routing failure
    report.append(run_case(
        "case8_unresolved",
        {"a": (101, "novel"), "b": (12, "supported_known")},
        [("a", {"prediction_type": "unresolved"}),
         ("b", {"prediction_type": "known", "semantic_category_id": 12})],
        {"unresolved_novel_rate": 1.0, "route_aware_novel_acc": 0.0,
         "novel_routing_recall": 0.0},
    ))

    # Case 9: zero-shot official known is novel under Pure, known under OV
    zs_role_pure = {"a": (45, "novel"), "b": (45, "novel")}
    zs_role_ov = {"a": (45, "zero_shot_known"), "b": (45, "zero_shot_known")}
    plist = [
        ("a", {"prediction_type": "novel", "virtual_category_id": 1}),
        ("b", {"prediction_type": "novel", "virtual_category_id": 1}),
    ]
    r_pure = TrackOCDEvaluator(gt_rows(zs_role_pure)).evaluate(preds(plist))
    r_ov = TrackOCDEvaluator(gt_rows(zs_role_ov)).evaluate(preds(plist))
    case9_ok = (
        abs(r_pure["route_aware_novel_acc"] - 1.0) < 1e-9
        and abs(r_ov["zero_shot_known_acc"] - 0.0) < 1e-9
        and abs(r_ov["known_to_novel_error"] - 1.0) < 1e-9
    )
    report.append({
        "case": "case9_pure_vs_ov_role",
        "passed": case9_ok,
        "pure_route_aware_novel_acc": r_pure["route_aware_novel_acc"],
        "ov_zero_shot_known_acc": r_ov["zero_shot_known_acc"],
        "ov_known_to_novel_error": r_ov["known_to_novel_error"],
    })

    all_ok = all(r["passed"] for r in report)
    out = {
        "version": "TrackOCD-v1.0",
        "all_passed": all_ok,
        "num_cases": len(report),
        "cases": report,
    }
    out_path = PROJECT_ROOT / "outputs" / "trackocd_v1" / "tests" / "evaluator_test_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
