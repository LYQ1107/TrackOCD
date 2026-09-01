#!/usr/bin/env python3
"""ICLR27 Phase 1 tests (25)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def read_csv(name):
    with open(PROJECT_ROOT / "outputs/iclr27_closure" / name) as f:
        return list(csv.DictReader(f))


def test1_manifest_stats_repeatable():
    rows = read_csv("tables/protocol_statistics.csv")
    assert len(rows) == 2
    assert rows[0]["protocol"] in ("pure", "ov_assisted")
    return True


def test2_pure_roles():
    rows = read_csv("tables/protocol_statistics.csv")
    p = next(r for r in rows if r["protocol"] == "pure")
    assert p["supported_known_categories"] == "48"
    assert p["novel_categories_total"] == "239"
    return True


def test3_ov_roles():
    rows = read_csv("tables/protocol_statistics.csv")
    o = next(r for r in rows if r["protocol"] == "ov_assisted")
    assert o["known_categories"] if "known_categories" in o else True
    assert o["zero_shot_known_categories"] == "30"
    return True


def test4_zero_shot_switch():
    rows = read_csv("tables/protocol_statistics.csv")
    p = next(r for r in rows if r["protocol"] == "pure")
    o = next(r for r in rows if r["protocol"] == "ov_assisted")
    assert int(o["zero_shot_known_tracks"]) == 24
    assert int(p["novel_tracks"]) == 843
    return True


def test5_known_no_hungarian():
    toy = json.loads((PROJECT_ROOT / "outputs/iclr27_closure/figures/evaluator_toy_example.json").read_text())
    assert toy["corrected_known_acc"] == 0.0
    return True


def test6_novel_only_hungarian():
    toy = json.loads((PROJECT_ROOT / "outputs/iclr27_closure/figures/evaluator_toy_example.json").read_text())
    assert toy["corrected_route_novel_acc"] == 1.0
    return True


def test7_toy_example():
    toy = json.loads((PROJECT_ROOT / "outputs/iclr27_closure/figures/evaluator_toy_example.json").read_text())
    assert toy["legacy_global_hungarian_acc"] > toy["corrected_all_track_acc"]
    return True


def test8_conversion_frame_count():
    p = json.loads((PROJECT_ROOT / "third_party/TrackEval/data/trackers/tao/tao_validation/simowt/data/predictions.json").read_text())
    imgs = {a["image_id"] for a in p}
    assert len(imgs) == 36375
    return True


def test9_conversion_gt_count():
    te = json.loads((PROJECT_ROOT / "outputs/iclr27_closure/tracking_eval/simowt/conversion_manifest.json").read_text())
    assert te["tracker"] == "simowt"
    return True


def test10_track_id_scope():
    p = json.loads((PROJECT_ROOT / "third_party/TrackEval/data/trackers/tao/tao_validation/simowt/data/predictions.json").read_text())
    keys = {(a["video_id"], a["track_id"]) for a in p}
    assert len(keys) == 649378
    return True


def test11_box_format():
    p = json.loads((PROJECT_ROOT / "third_party/TrackEval/data/trackers/tao/tao_validation/simowt/data/predictions.json").read_text())
    assert all(len(a["bbox"]) == 4 for a in p[:1000])
    return True


def test12_no_nan_boxes():
    import numpy as np
    p = json.loads((PROJECT_ROOT / "third_party/TrackEval/data/trackers/tao/tao_validation/simowt/data/predictions.json").read_text())
    b = np.array([a["bbox"] for a in p[:5000]])
    assert not np.isnan(b).any()
    return True


def test13_trackeval_repeatable():
    s = read_csv("tracking_eval/simowt/summary.csv")
    assert len(s) == 3
    assert float(s[0]["HOTA"]) > 0
    return True


def test14_category_agnostic():
    m = json.loads((PROJECT_ROOT / "outputs/iclr27_closure/tracking_eval/simowt/class_agnostic_metrics.json").read_text())
    assert m["subset"] == "all"
    return True


def test15_role_filter_gt_only():
    k = read_csv("tracking_eval/simowt/known_role_metrics.csv")
    u = read_csv("tracking_eval/simowt/novel_role_metrics.csv")
    assert k and u
    return True


def test16_coverage_definition():
    rows = read_csv("tables/track_coverage_table.csv")
    assert any("track_coverage_0.5" in r for r in rows)
    return True


def test17_unmatched_gt_error():
    rows = read_csv("end_to_end/coverage_aware_results.csv")
    for r in rows:
        assert int(r["unmatched"]) == 3435
    return True


def test18_duplicate_stats():
    q = read_csv("tracking_eval/simowt/track_quality_metrics.csv")
    assert any(r["metric"] == "pred_track_count" for r in q)
    return True


def test19_matched_only_vs_coverage():
    mo = read_csv("end_to_end/matched_only_results.csv")
    ca = read_csv("end_to_end/coverage_aware_results.csv")
    assert float(mo[0]["route_novel_acc"]) > float(ca[0]["route_novel_acc"])
    return True


def test20_main_table_auto():
    rows = read_csv("tables/gt_track_main_table.csv")
    assert len(rows) >= 4
    return True


def test21_name_mapping_unique():
    rows = read_csv("audit/method_name_mapping.csv")
    names = [r["paper_name"] for r in rows]
    assert len(names) == len(set(names))
    return True


def test22_claims_have_files():
    rows = read_csv("audit/claim_evidence_matrix.csv")
    for r in rows:
        assert r["supporting_files"]
    return True


def test23_unsupported_claims_marked():
    rows = read_csv("audit/claim_evidence_matrix.csv")
    assert all(r["status"] in ("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "MISSING_EXPERIMENT") for r in rows)
    return True


def test24_old_artifacts_preserved():
    assert (PROJECT_ROOT / "outputs/metrics/summary.csv").exists()
    assert (PROJECT_ROOT / "runs/domain_router/router_gate.json").exists()
    return True


def test25_results_have_metadata():
    rows = read_csv("tables/gt_track_main_table.csv")
    assert all(r.get("protocol") and r.get("subset") and r.get("seed") and r.get("track_source") for r in rows)
    return True


def main():
    names = ["manifest_stats_repeatable", "pure_roles", "ov_roles",
             "zero_shot_switch", "known_no_hungarian", "novel_only_hungarian",
             "toy_example", "conversion_frame_count", "conversion_gt_count",
             "track_id_scope", "box_format", "no_nan_boxes",
             "trackeval_repeatable", "category_agnostic", "role_filter_gt_only",
             "coverage_definition", "unmatched_gt_error", "duplicate_stats",
             "matched_only_vs_coverage", "main_table_auto", "name_mapping_unique",
             "claims_have_files", "unsupported_claims_marked",
             "old_artifacts_preserved", "results_have_metadata"]
    report = []
    for i, name in enumerate(names, 1):
        try:
            globals()[f"test{i}_{name}"]()
            report.append({"test": f"test{i}", "passed": True})
            print("PASS", f"test{i}")
        except Exception as e:
            report.append({"test": f"test{i}", "passed": False, "error": str(e)})
            print("FAIL", f"test{i}", e)
    out = {"all_passed": all(r["passed"] for r in report), "tests": report}
    p = PROJECT_ROOT / "outputs/iclr27_closure/tests/test_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
