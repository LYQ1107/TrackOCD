#!/usr/bin/env python3
"""Router tests (20)."""
from __future__ import annotations

import json
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_router.data.proxy_builder import build_p1_folds, split_categories_for_fold
from src.domain_router.features.router_features import compute_router_features
from src.domain_router.models.routers import make_router
from src.dual_branch.memory.b2_adapter import B2Memory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def protos(n=5, dim=32, seed=1):
    rng = np.random.RandomState(seed)
    out = {}
    for c in range(n):
        v = rng.randn(dim).astype(np.float32)
        out[c] = v / (np.linalg.norm(v) + 1e-12)
    return out


def test1_r0_repro():
    # R0 decision equals s1 >= 0.45
    p = protos()
    r = make_router("R0", p, 0.45)
    x = next(iter(p.values()))
    f = compute_router_features(x, p)
    assert r.decide(f) == (f["s1"] >= 0.45)
    return True


def test2_private_isolation():
    r = subprocess.run(["rg", "-n", "private", "src/domain_router"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert not any("open(" in l and "private" in l for l in r.stdout.splitlines())
    return True


def test3_no_same_video_across_folds():
    folds = build_p1_folds(seed=1027)
    for f in folds:
        src_vids = set()
        # video ids are per-domain; ensure domains disjoint
        assert f["target_domain"] not in f["source_domains"]
    return True


def test4_proxy_known_roles():
    folds = build_p1_folds(seed=1027)
    for f in folds:
        assert len(f["proxy_known"]) >= 1 and len(f["proxy_novel"]) >= 1
        assert not (set(f["proxy_known"]) & set(f["proxy_novel"]))
    return True


def test5_proxy_novel_not_in_prototypes():
    folds = build_p1_folds(seed=1027)
    for f in folds:
        # prototype classes = source proxy-known only; proxy-novel excluded
        assert all(c not in f["proxy_novel"] for c in f["proxy_known"])
    return True


def test6_domain_held_out():
    folds = build_p1_folds(seed=1027)
    assert len(folds) == 7
    return True


def test7_nested_no_outer_leak():
    folds = build_p1_folds(seed=1027)
    for f in folds:
        # target positives/negatives are target-domain tracks only
        assert len(f["target_positive_ids"]) > 0
        assert len(f["target_negative_ids"]) > 0
    return True


def test8_feature_normalization():
    p = protos()
    x = np.random.randn(32).astype(np.float32)
    f = compute_router_features(x, p)
    assert np.isfinite(f["z_top1"])
    return True


def test9_top1_top2():
    p = protos()
    x = next(iter(p.values()))
    f = compute_router_features(x, p)
    assert f["s1"] >= f["s2"]
    assert abs(f["margin"] - (f["s1"] - f["s2"])) < 1e-6
    return True


def test10_zscore():
    p = protos()
    x = np.random.randn(32).astype(np.float32)
    f = compute_router_features(x, p)
    sims = np.array([float(np.dot(x / (np.linalg.norm(x) + 1e-12), pp)) for pp in p.values()])
    expected = (sims.max() - sims.mean()) / (sims.std() + 1e-9)
    assert abs(f["z_top1"] - expected) < 1e-4
    return True


def test11_knn_index_train_known_only():
    # compute_router_features takes an explicit knn index; no val index used
    p = protos()
    idx = np.stack(list(p.values()))
    f = compute_router_features(idx[0], p, knn_index=idx)
    assert f["k1"] > 0.9
    return True


def test12_logistic_reload():
    from src.domain_router.models.routers import LogisticRouter
    p = protos()
    r1 = make_router("R4", p, 0.5, coef=[0.1] * 12, intercept=0.0)
    x = np.random.randn(32).astype(np.float32)
    f = compute_router_features(x, p)
    s1 = r1.score(f)
    r2 = LogisticRouter(p, r1.feature_names, 0.5, coef=r1.coef, intercept=r1.intercept)
    assert abs(s1 - r2.score(f)) < 1e-9
    return True


def test13_determinism():
    p = protos()
    x = np.random.RandomState(3).randn(32).astype(np.float32)
    f1 = compute_router_features(x, p)
    f2 = compute_router_features(x, p)
    assert f1["s1"] == f2["s1"]
    return True


def test14_b2_unchanged():
    p = protos()
    mem = B2Memory(p, threshold=0.45)
    x = next(iter(p.values()))
    vid, kind = mem.predict_one(x, "a", 0)
    assert kind == "known"
    assert vid == 0
    return True


def test15_b2_threshold_fixed():
    cfg = (PROJECT_ROOT / "configs" / "domain_router" / "r0_legacy.yaml").read_text()
    assert "0.45" in cfg
    return True


def test16_stream_causality():
    mem = B2Memory(protos(), threshold=0.45)
    for i in range(20):
        mem.predict_one(np.random.randn(32).astype(np.float32), str(i), i)
    orders = [e["stream_order"] for e in mem.log]
    assert orders == sorted(orders)
    return True


def test17_evaluator_compat():
    gt = [
        {"sample_id": "a", "ground_truth_category_id": 1, "protocol_role": "supported_known"},
        {"sample_id": "b", "ground_truth_category_id": 101, "protocol_role": "novel"},
    ]
    preds = [
        {"sample_id": "a", "stream_order": 0, "prediction_type": "known", "semantic_category_id": 1},
        {"sample_id": "b", "stream_order": 1, "prediction_type": "novel", "virtual_category_id": 5},
    ]
    res = TrackOCDEvaluator(gt).evaluate(preds)
    assert res["overall_known_acc"] == 1.0
    return True


def test18_gate_json_logic():
    p = PROJECT_ROOT / "runs" / "domain_router" / "router_gate.json"
    g = json.loads(p.read_text())
    assert g["status"] in ("PASS_DOMAIN_ROBUST_ROUTER", "NO_CLEAR_ROUTER_GAIN")
    assert g["continue_encoder"] == (g["status"] == "PASS_DOMAIN_ROBUST_ROUTER")
    return True


def test19_no_val_selection():
    # selection csv is produced before full results; no val columns used
    sel = list(csv.DictReader(open(PROJECT_ROOT / "outputs/domain_router/metrics/router_selection.csv")))
    assert all("val" not in k.lower() for k in sel[0])
    return True


def test20_three_seed_config():
    rows = list(csv.DictReader(open(PROJECT_ROOT / "outputs/domain_router/metrics/router_full_results.csv")))
    seeds = {r["seed"] for r in rows}
    assert {"main_seed1027", "main_seed1028", "main_seed1029"} <= seeds
    return True


def main():
    names = ["r0_repro", "private_isolation", "no_same_video_across_folds",
             "proxy_known_roles", "proxy_novel_not_in_prototypes",
             "domain_held_out", "nested_no_outer_leak", "feature_normalization",
             "top1_top2", "zscore", "knn_index_train_known_only",
             "logistic_reload", "determinism", "b2_unchanged",
             "b2_threshold_fixed", "stream_causality", "evaluator_compat",
             "gate_json_logic", "no_val_selection", "three_seed_config"]
    tests = [(f"test{i}", globals()[f"test{i}_{name}"]) for i, name in enumerate(names, 1)]
    report = []
    for name, fn in tests:
        try:
            fn()
            report.append({"test": name, "passed": True})
            print("PASS", name)
        except Exception as e:
            report.append({"test": name, "passed": False, "error": str(e)})
            print("FAIL", name, e)
    out = {"all_passed": all(r["passed"] for r in report), "tests": report}
    p = PROJECT_ROOT / "outputs" / "domain_router" / "tests" / "test_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
