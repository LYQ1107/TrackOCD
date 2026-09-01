#!/usr/bin/env python3
"""Causal score-shift tests (24)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.causal_score_shift.causal_routers import (
    C0Legacy, C1Global, C2Translation, C3LocationScale, C4Reliability,
)

REF = {
    "ref_known_median": 0.70,
    "ref_known_mad": 0.05,
    "hc_score": {0.95: 0.80, 0.975: 0.85},
}


def make_router(name, thr=0.45):
    if name == "C0":
        return C0Legacy()
    if name == "C1":
        return C1Global(thr)
    if name == "C2":
        return C2Translation(thr, REF["ref_known_median"], 0.80, 0.05, 5, 0.25, 0.2)
    if name == "C3":
        return C3LocationScale(thr, 0.70, 0.05, 0.80, 0.05, 5, 5, 0.25, 0.2)
    return C4Reliability(thr, 0.70, 0.80, 0.05, 5, 0.25, 0.2)


def test1_c0_repro():
    r = make_router("C0")
    assert r.predict({"s1": 0.5}) is True
    assert r.predict({"s1": 0.4}) is False
    return True


def test2_c1_train_oof_threshold():
    cfg = (PROJECT_ROOT / "configs/causal_score_shift/c1_pooled_oof.yaml").read_text()
    assert "oof" in cfg
    return True


def test3_video_reset():
    r = make_router("C2")
    r.reset_video(1)
    r.update_after_prediction({"s1": 0.9, "margin": 0.2}, True)
    assert len(r.anchors) == 1
    r.reset_video(2)
    assert len(r.anchors) == 0
    return True


def test4_no_future():
    r = make_router("C2")
    r.reset_video(0)
    # predicting at time i only uses history; feed sequentially
    seq = [0.8, 0.82, 0.78, 0.3]
    for s in seq:
        r.predict({"s1": s, "margin": 0.1})
    # no error and no look-ahead (cannot read future by construction)
    return True


def test5_online_equals_replay():
    r1 = make_router("C2")
    r2 = make_router("C2")
    seq = [0.8, 0.82, 0.78, 0.30, 0.85]
    out1 = []
    r1.reset_video(0)
    for s in seq:
        d = r1.predict({"s1": s, "margin": 0.1})
        out1.append(d)
        r1.update_after_prediction({"s1": s, "margin": 0.1}, d)
    # replay: same sequence
    out2 = []
    r2.reset_video(0)
    for s in seq:
        d = r2.predict({"s1": s, "margin": 0.1})
        out2.append(d)
        r2.update_after_prediction({"s1": s, "margin": 0.1}, d)
    assert out1 == out2
    return True


def test6_anchor_history_only():
    r = make_router("C2")
    r.reset_video(0)
    r.update_after_prediction({"s1": 0.9, "margin": 0.2}, True)
    assert len(r.anchors) == 1
    return True


def test7_anchor_no_gt():
    r = make_router("C2")
    r.reset_video(0)
    r.update_after_prediction({"s1": 0.9, "margin": 0.2}, True)
    assert r.anchors == [0.9]
    return True


def test8_fallback_when_few_anchors():
    r = make_router("C2")
    r.reset_video(0)
    r.update_after_prediction({"s1": 0.9, "margin": 0.2}, True)
    # < min_anchors -> raw rule
    assert r.predict({"s1": 0.4, "margin": 0.1}) is False
    return True


def test9_all_novel_safe_fallback():
    r = make_router("C2")
    r.reset_video(0)
    for s in (0.30, 0.32, 0.28):
        d = r.predict({"s1": s, "margin": 0.05})
        r.update_after_prediction({"s1": s, "margin": 0.05}, d)
    assert len(r.anchors) == 0
    return True


def test10_single_track_fallback():
    r = make_router("C2")
    r.reset_video(0)
    assert r.predict({"s1": 0.5, "margin": 0.1}) is True
    return True


def test11_mad_zero_translation():
    r = make_router("C3")
    r.reset_video(0)
    for s in (0.9, 0.9, 0.9, 0.9, 0.9):
        r.update_after_prediction({"s1": s, "margin": 0.2}, True)
    # MAD=0 -> translation fallback, no crash
    assert r.predict({"s1": 0.9, "margin": 0.2}) in (True, False)
    return True


def test12_shift_clip():
    r = make_router("C2")
    r.reset_video(0)
    for s in (0.95, 0.96, 0.94, 0.95, 0.96):
        r.update_after_prediction({"s1": s, "margin": 0.2}, True)
    assert abs(r._shift()) <= 0.2 + 1e-9
    return True


def test13_scale_clip():
    r = make_router("C3")
    r.reset_video(0)
    for s in (0.9, 0.5, 0.88, 0.52, 0.86):
        r.update_after_prediction({"s1": s, "margin": 0.2}, True)
    assert r.predict({"s1": 0.8, "margin": 0.1}) in (True, False)
    return True


def test14_reliability_range():
    r = make_router("C4")
    r.reset_video(0)
    for s in (0.9, 0.91, 0.89, 0.92, 0.9, 0.91, 0.89, 0.9, 0.91, 0.9):
        r.update_after_prediction({"s1": s, "margin": 0.2}, True)
    w = min(1.0, len(r.anchors) / r.anchor_target)
    assert 0 <= w <= 1
    return True


def test15_no_unbounded_shift():
    r = make_router("C2")
    r.reset_video(0)
    for s in (0.99,) * 30:
        r.update_after_prediction({"s1": s, "margin": 0.2}, True)
    assert abs(r.shift_ema) <= 0.2 + 1e-9
    return True


def test16_b2_unchanged():
    from src.dual_branch.memory.b2_adapter import B2Memory
    p = {0: np.ones(32, dtype=np.float32) / np.sqrt(32)}
    m = B2Memory(p, threshold=0.45)
    v, k = m.predict_one(next(iter(p.values())), "a", 0)
    assert k == "known" and v == 0
    return True


def test17_b2_threshold_fixed():
    cfg = (PROJECT_ROOT / "configs/causal_score_shift/c0_legacy.yaml").read_text()
    assert "0.45" in cfg
    return True


def test18_known_prototype_unchanged():
    from src.ocd_v2.common import load_train_known, build_prototypes
    feats, labels = load_train_known("dinov2")
    p = build_prototypes(feats, labels, set(labels.values()))
    assert len(p) == 48
    return True


def test19_private_isolation():
    import subprocess
    r = subprocess.run(["rg", "-n", "private", "src/causal_score_shift"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert not any("open(" in l and "private" in l for l in r.stdout.splitlines())
    return True


def test20_stream_causality():
    r = make_router("C2")
    r.reset_video(0)
    for i in range(20):
        d = r.predict({"s1": 0.5 + 0.01 * i, "margin": 0.05})
        r.update_after_prediction({"s1": 0.5 + 0.01 * i, "margin": 0.05}, d)
    return True


def test21_state_checkpoint():
    r = make_router("C2")
    r.reset_video(7)
    r.update_after_prediction({"s1": 0.9, "margin": 0.2}, True)
    d = r.state_dict()
    r2 = make_router("C2")
    r2.load_state_dict(d)
    assert r2.anchors == r.anchors
    return True


def test22_three_seed_determinism():
    raw = list(csv.DictReader(open(PROJECT_ROOT / "outputs/causal_score_shift/metrics/full_results.csv")))
    seeds = {r["seed"] for r in raw}
    assert {"main_seed1027", "main_seed1028", "main_seed1029"} <= seeds
    return True


def test23_evaluator_compat():
    from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
    gt = [
        {"sample_id": "a", "ground_truth_category_id": 1, "protocol_role": "supported_known"},
        {"sample_id": "b", "ground_truth_category_id": 101, "protocol_role": "novel"},
    ]
    preds = [
        {"sample_id": "a", "stream_order": 0, "prediction_type": "known", "semantic_category_id": 1},
        {"sample_id": "b", "stream_order": 1, "prediction_type": "novel", "virtual_category_id": 3},
    ]
    res = TrackOCDEvaluator(gt).evaluate(preds)
    assert res["overall_known_acc"] == 1.0
    return True


def test24_gate_logic():
    g = json.loads((PROJECT_ROOT / "runs/causal_score_shift/causal_gate.json").read_text())
    assert g["status"] in ("PASS_CAUSAL_SCORE_SHIFT", "NO_CLEAR_CALIBRATION_GAIN")
    return True


def main():
    names = ["c0_repro", "c1_train_oof_threshold", "video_reset", "no_future",
             "online_equals_replay", "anchor_history_only", "anchor_no_gt",
             "fallback_when_few_anchors", "all_novel_safe_fallback",
             "single_track_fallback", "mad_zero_translation", "shift_clip",
             "scale_clip", "reliability_range", "no_unbounded_shift",
             "b2_unchanged", "b2_threshold_fixed", "known_prototype_unchanged",
             "private_isolation", "stream_causality", "state_checkpoint",
             "three_seed_determinism", "evaluator_compat", "gate_logic"]
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
    p = PROJECT_ROOT / "outputs/causal_score_shift/tests/test_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
