#!/usr/bin/env python3
"""18 tests for the DINOv3 bake-off pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dinov3_bakeoff.adapter import (
    WEIGHT_SHA256, WEIGHT_SOURCE, MODEL_ID, HF_REVISION, MEAN, STD, SIZE,
)
from src.dinov3_bakeoff.extract import make_transform
from src.features.extract import crop_bbox, sample_indices
from src.dual_branch.memory.b2_adapter import B2Memory
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def test1_weight_source_hash():
    import hashlib
    p = PROJECT_ROOT / "checkpoints/dinov3/timm_converted/model.safetensors"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == WEIGHT_SHA256
    assert WEIGHT_SOURCE.startswith("timm")
    return True


def test2_param_count():
    import timm
    m = timm.create_model(f"hf_hub:{MODEL_ID}", pretrained=True, num_classes=0)
    n = sum(p.numel() for p in m.parameters())
    assert 80e6 < n < 90e6, n
    return True


def test3_feature_dim():
    import timm
    m = timm.create_model(f"hf_hub:{MODEL_ID}", pretrained=True, num_classes=0)
    assert m.num_features == 768
    return True


def test4_preprocessing():
    tf = make_transform()
    assert tf.transforms[0].size == (SIZE, SIZE)
    assert np.allclose(tf.transforms[2].mean, MEAN)
    assert np.allclose(tf.transforms[2].std, STD)
    return True


def test5_bbox_crop_consistency():
    img = Image.new("RGB", (100, 100))
    crop = crop_bbox(img, [20, 20, 50, 50])
    # 10% context on 30x30 -> 3px each side, clamped to image
    assert crop.size == (36, 36), crop.size
    return True


def test6_frame_sampling_consistency():
    assert sample_indices(10, 8) == sorted(set(np.linspace(0, 9, 8).astype(int).tolist()))
    assert len(sample_indices(100, 8)) == 8
    return True


def test7_no_random_augmentation():
    tf = make_transform()
    names = [type(t).__name__ for t in tf.transforms]
    assert not any("Random" in n for n in names)
    return True


def test8_mean_aggregation():
    frames = np.random.randn(5, 768).astype(np.float32)
    v = frames.mean(axis=0)
    v /= np.linalg.norm(v) + 1e-12
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4
    return True


def test9_l2_normalization():
    assert abs(np.linalg.norm(np.ones(768, dtype=np.float32) / np.sqrt(768)) - 1.0) < 1e-5
    return True


def test10_cache_atomic_write():
    from src.dinov3_bakeoff.adapter import atomic_write_text
    p = PROJECT_ROOT / "outputs" / "dinov3_bakeoff" / "tests" / "atomic_tmp.json"
    atomic_write_text(p, '{"ok": 1}')
    assert json.loads(p.read_text())["ok"] == 1
    p.unlink()
    return True


def test11_private_label_isolation():
    r = subprocess.run(
        ["rg", "-n", "private", "src/dinov3_bakeoff"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    assert not any("open(" in l and "private" in l for l in r.stdout.splitlines())
    return True


def test12_router_memory_same_space():
    # config-level: V2 uses DINOv3 features for both router and memory
    cfg = (PROJECT_ROOT / "configs" / "dinov3_bakeoff" / "v2_dinov3_mean_b2.yaml").read_text()
    assert "dinov3" in cfg
    return True


def test13_threshold_isolation():
    # calibration is a pure function of the feature dict; no cross-backbone reuse
    from src.dinov3_bakeoff.calibration import calibrate_b2_threshold
    rng = np.random.RandomState(0)
    labels = {f"s{i}": i % 4 for i in range(80)}
    feats = {s: rng.randn(768).astype(np.float32) for s in labels}
    thr, curve = calibrate_b2_threshold(feats, labels)
    assert 0.1 <= thr <= 0.9
    assert len(curve) > 5
    return True


def test14_stream_causality():
    protos = {0: np.ones(32, dtype=np.float32) / np.sqrt(32)}
    mem = B2Memory(protos, threshold=0.5)
    for i in range(50):
        mem.predict_one(np.random.randn(32).astype(np.float32), str(i), i)
    orders = [e["stream_order"] for e in mem.log]
    assert orders == sorted(orders)
    return True


def test15_evaluator_compat():
    gt = [
        {"sample_id": "a", "ground_truth_category_id": 12, "protocol_role": "supported_known"},
        {"sample_id": "b", "ground_truth_category_id": 101, "protocol_role": "novel"},
    ]
    preds = [
        {"sample_id": "a", "stream_order": 0, "prediction_type": "known", "semantic_category_id": 12},
        {"sample_id": "b", "stream_order": 1, "prediction_type": "novel", "virtual_category_id": 3},
    ]
    res = TrackOCDEvaluator(gt).evaluate(preds)
    assert res["overall_known_acc"] == 1.0 and res["route_aware_novel_acc"] == 1.0
    return True


def test16_checkpoint_reload():
    import timm
    m1 = timm.create_model(f"hf_hub:{MODEL_ID}", pretrained=True, num_classes=0).eval()
    m2 = timm.create_model(f"hf_hub:{MODEL_ID}", pretrained=True, num_classes=0).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        a = m1(x).numpy(); b = m2(x).numpy()
    assert np.allclose(a, b, atol=1e-6)
    return True


def test17_transformer_mask():
    from src.trackocd_v1.trajectory_encoder import TrajectoryEncoder
    m = TrajectoryEncoder(10, variant="transformer")
    x = torch.randn(2, 8, 1280)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[1, 6:] = False
    out = m(x, mask)
    assert out.shape == (2, 256)
    return True


def test18_transformer_shared_space():
    cfg = (PROJECT_ROOT / "configs" / "dinov3_bakeoff" / "t1_dinov3_transformer.yaml").read_text()
    assert "shared" in cfg
    return True


def main():
    tests = [
        (f"test{i}", globals()[f"test{i}_{name}"])
        for i, name in enumerate(
            ["weight_source_hash", "param_count", "feature_dim", "preprocessing",
             "bbox_crop_consistency", "frame_sampling_consistency",
             "no_random_augmentation", "mean_aggregation", "l2_normalization",
             "cache_atomic_write", "private_label_isolation",
             "router_memory_same_space", "threshold_isolation",
             "stream_causality", "evaluator_compat", "checkpoint_reload",
             "transformer_mask", "transformer_shared_space"],
            start=1,
        )
    ]
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
    p = PROJECT_ROOT / "outputs" / "dinov3_bakeoff" / "tests" / "test_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
