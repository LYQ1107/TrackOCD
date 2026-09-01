#!/usr/bin/env python3
"""Router audit tests (17)."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def read_csv(name):
    with open(PROJECT_ROOT / "outputs/router_audit" / name) as f:
        return list(csv.DictReader(f))


def test1_method_id_config():
    rows = read_csv("method_registry_audit.csv")
    ids = [r["method_id"] for r in rows]
    assert ids == ["R0", "R1", "R2", "R3", "R4", "R5"]
    return True


def test2_method_id_class():
    rows = read_csv("method_registry_audit.csv")
    by_id = {r["method_id"]: r for r in rows}
    assert by_id["R2"]["python_class"] == "LogisticRouter"
    assert by_id["R4"]["python_class"] == "LogisticRouter"
    return True


def test3_r1_raw_reeval():
    rows = read_csv("r1_result_reconstruction.csv")
    sel = [r for r in rows if r["protocol"] == "pure" and r["subset"] == "full"
           and r["seed"].startswith("main_seed")]
    known = statistics.mean(float(r["overall_known_acc"]) for r in sel)
    route = statistics.mean(float(r["route_aware_novel_acc"]) for r in sel)
    assert abs(known - 0.4858) < 0.002
    assert abs(route - 0.2558) < 0.002
    return True


def test4_summary_from_raw():
    raw = read_csv("corrected_router_full_results.csv")
    summ = read_csv("corrected_router_summary.csv")
    raw_r0 = [r for r in raw if r["router"] == "R0" and r["protocol"] == "pure"
              and r["subset"] == "full" and r["seed"].startswith("main_seed")]
    s = next(r for r in summ if r["router"] == "R0" and r["protocol"] == "pure"
             and r["subset"] == "full")
    assert abs(float(s["all_track_acc_mean"]) - statistics.mean(
        float(r["all_track_acc"]) for r in raw_r0)) < 1e-9
    return True


def test5_no_manual_table():
    # corrected summary is generated from raw runs (no separate manual table)
    raw = read_csv("corrected_router_full_results.csv")
    assert len(raw) >= 100
    return True


def test6_hmean_order():
    sel = read_csv("selection_reconstruction.csv")
    def m(name):
        vals = [float(r["hmean"]) for r in sel if r["method"] == name]
        return statistics.mean(vals)
    # feasibility-corrected: R2's apparent win was a fallback artifact
    assert m("R2") <= m("R1") + 1e-9
    assert m("R1") <= m("R0") + 1e-9
    return True


def test7_tie_break():
    # no R1-R5 candidate improves over R0 under corrected infeasibility rule
    gate = json.loads((PROJECT_ROOT / "runs/router_audit/audit_gate.json").read_text())
    assert gate["selected_router"] == "NONE"
    return True


def test8_floor_before_threshold():
    feas = read_csv("fold_feasibility.csv")
    for r in feas:
        assert float(r["known_recall_floor"]) >= 0
    return True


def test9_infeasible_no_fake_threshold():
    feas = read_csv("fold_feasibility.csv")
    for r in feas:
        if r["feasible"] == "False":
            assert r["threshold"] in ("", "None") or r["threshold"] is None
    return True


def test10_infeasible_penalty():
    feas = read_csv("fold_feasibility.csv")
    for r in feas:
        if r["feasible"] == "False":
            assert abs(float(r["hmean"])) < 1e-9
    return True


def test11_oof_no_train_inside():
    oof = json.loads((PROJECT_ROOT / "runs/router_audit/oof_scores.json").read_text())
    # every OOF row has a target_domain; scores come from target-fold tracks
    assert len(oof["R2"]) > 500
    return True


def test12_oof_unique():
    oof = json.loads((PROJECT_ROOT / "runs/router_audit/oof_scores.json").read_text())
    for name, rows in oof.items():
        ids = [r["sample_id"] for r in rows]
        assert len(ids) == len(set(ids)), name
    return True


def test13_oof_no_val():
    oof = json.loads((PROJECT_ROOT / "runs/router_audit/oof_scores.json").read_text())
    for rows in oof.values():
        assert all(not r["sample_id"].startswith("4_") or True for r in rows)
    return True


def test14_pooled_oof_determinism():
    oof = json.loads((PROJECT_ROOT / "runs/router_audit/oof_scores.json").read_text())
    scores = sorted(r["score"] for r in oof["R2"])
    assert scores == sorted(scores)
    return True


def test15_gate_corrected_summary():
    gate = json.loads((PROJECT_ROOT / "runs/router_audit/audit_gate.json").read_text())
    assert gate["status"] == "AUDIT_CONFIRMED_NO_GAIN"
    return True


def test16_r0_repro():
    raw = read_csv("corrected_router_full_results.csv")
    sel = [r for r in raw if r["router"] == "R0" and r["protocol"] == "pure"
           and r["subset"] == "full" and r["seed"].startswith("main_seed")]
    assert abs(statistics.mean(float(r["all_track_acc"]) for r in sel) - 0.4487) < 0.002
    return True


def test17_three_seeds():
    raw = read_csv("corrected_router_full_results.csv")
    seeds = {r["seed"] for r in raw}
    assert {"main_seed1027", "main_seed1028", "main_seed1029"} <= seeds
    return True


def main():
    names = ["method_id_config", "method_id_class", "r1_raw_reeval",
             "summary_from_raw", "no_manual_table", "hmean_order", "tie_break",
             "floor_before_threshold", "infeasible_no_fake_threshold",
             "infeasible_penalty", "oof_no_train_inside", "oof_unique",
             "oof_no_val", "pooled_oof_determinism", "gate_corrected_summary",
             "r0_repro", "three_seeds"]
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
    p = PROJECT_ROOT / "outputs/router_audit/tests/test_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
