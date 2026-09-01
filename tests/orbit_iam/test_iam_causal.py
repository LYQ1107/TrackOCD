"""Functional tests for ORBIT-IAM causal memory and pair construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_memory_update_causal():
    from src.orbit_iam.iam_memory import IamMemory
    protos = {1: np.ones(8, dtype=np.float32) / np.sqrt(8)}
    mem = IamMemory(protos)
    z = np.ones(8, dtype=np.float32) / np.sqrt(8)
    vid = mem.create_novel(z, created_at=0)
    assert vid is not None
    st = mem.state(vid)
    assert st["support"] >= 1
    assert 0.0 <= st["conf"] <= 1.0
    mem.update_novel(vid, z, cos_to_center=1.0, update_radius=False, margin=0.1)
    assert mem.support(vid) >= 2


def test_compat_feature_legal():
    from src.orbit_iam.compat import build_compat_features, compat_feature_spec
    feats = compat_feature_spec("sim,margin,radius,support,conf,mem,rel")
    z = np.ones(8, dtype=np.float32) / np.sqrt(8)
    vals = build_compat_features(z, z, 0.3, 5, 0.5, 100, 0.9, 0.05, feats)
    assert len(vals) == 7
    assert all(np.isfinite(v) for v in vals)


def test_hard_negative_selection_is_memory_conditioned():
    src = (ROOT / "src/orbit_iam/train_iam.py").read_text()
    assert "hard_neg_k" in src
    assert "neg_vids[:k_neg]" in src
    assert "v != own_vid" in src


def test_first_occurrence_all_negative():
    src = (ROOT / "src/orbit_iam/train_iam.py").read_text()
    assert 'if q["first"]:' in src
    assert "targets = [0.0]" in src


def test_no_gt_purity_in_memory():
    src = (ROOT / "src/orbit_iam/iam_memory.py").read_text()
    assert "self.purity" not in src
    assert "purity_offline" not in src


if __name__ == "__main__":
    import traceback
    results = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results.append({"test": name, "status": "PASS"})
            except Exception as e:
                results.append({"test": name, "status": "FAIL",
                                "error": str(e), "trace": traceback.format_exc()})
    out = ROOT / "outputs/orbit_iam/tests/test_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"phase": "4E", "suite": "orbit_iam",
                               "results": results,
                               "passed": sum(1 for r in results if r["status"] == "PASS"),
                               "total": len(results)}, indent=1))
    print(json.dumps(results, indent=1))
