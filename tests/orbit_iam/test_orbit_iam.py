"""ORBIT-IAM unit tests: pair labels, hard negatives, causal memory."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_compat_feature_spec_order():
    from src.orbit_iam.compat import compat_feature_spec
    names = compat_feature_spec("sim,margin,radius,support,conf,mem,rel")
    assert names == ["sim", "margin", "radius", "support", "conf", "mem", "rel"]
    with pytest.raises(AssertionError):
        compat_feature_spec("sim,wat")


def test_build_compat_features_values():
    from src.orbit_iam.compat import build_compat_features
    z = np.array([1.0, 0.0], dtype=np.float32)
    p = np.array([0.8, 0.6], dtype=np.float32)
    feats = build_compat_features(z, p, radius=0.5, support=20, conf=0.3,
                                  mem_size=150, rel=0.9, margin=0.04,
                                  feat_names=["sim", "margin", "radius",
                                              "support", "conf", "mem", "rel"])
    assert feats[0] == pytest.approx(0.8, abs=1e-6)
    assert feats[1] == pytest.approx(0.04)
    assert feats[2] == pytest.approx(0.5)
    assert feats[3] == pytest.approx(math.log1p(20) / math.log1p(300))
    assert feats[4] == pytest.approx(0.3)
    assert feats[5] == pytest.approx(math.log1p(150) / math.log1p(300))
    assert feats[6] == pytest.approx(0.9)


def test_compat_head_forward_shape():
    from src.orbit_iam.model import ORBITIAMModel
    model = ORBITIAMModel(dim=8, bottleneck=4, gate_dim=5, reuse_dim=5,
                          compat_dim=6)
    x = torch.randn(3, 6)
    q = model.compat_forward(x)
    assert q.shape == (3,)


def test_hard_negative_selection_prefers_nearest_wrong():
    # emulate train_iam negative ordering: top-k of sim-sorted vids excluding own
    rng = np.random.RandomState(0)
    z = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    protos = np.array([
        [0.99, 0.1, 0.0],
        [0.8, 0.5, 0.2],
        [0.2, 0.9, 0.3],
        [-0.3, 0.9, 0.4],
    ], dtype=np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    ns = protos @ z
    order = np.argsort(ns)[::-1]
    vids = [int(o) for o in order]
    own = 1
    negs = [v for v in vids if v != own][:3]
    assert negs[0] == 0
    assert ns[negs[0]] > ns[negs[1]] > ns[negs[2]]
    assert own not in negs


def test_iam_memory_causal_stats():
    from src.orbit_iam.iam_memory import IamMemory
    known = {1: np.array([1.0, 0.0], dtype=np.float32)}
    mem = IamMemory(known, {1: 0.3})
    z1 = np.array([0.9, 0.3], dtype=np.float32)
    z1 /= np.linalg.norm(z1)
    vid = mem.create_novel(z1, created_at=0)
    st = mem.state(vid)
    assert st["support"] == 1
    assert 0.0 <= st["conf"] <= 1.0
    z2 = np.array([0.95, 0.2], dtype=np.float32)
    z2 /= np.linalg.norm(z2)
    mem.update_novel(vid, z2, cos_to_center=float(np.dot(mem.novel[vid]["proto"], z2)),
                     update_radius=True, margin=0.1)
    st2 = mem.state(vid)
    assert st2["support"] == 2
    assert st2["dispersion"] >= 0.0
    assert st2["mean_margin"] == pytest.approx(0.1)
    assert st2["low_margin_rate"] == 0.0
    # confidence must not use GT purity
    assert "purity" not in st2


def test_decision_policy_no_oracle_k():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    # reuse decision only depends on q and current memory size
    assert "compat_margin" in src
    assert "len(mem.novel)" in src


def test_checkpoint_has_compat_head():
    ck = torch.load(ROOT / "runs/orbit_iam/iam_i1/model.pth",
                    map_location="cpu", weights_only=False)
    assert ck["compat_dim"] == 6
    assert any(k.startswith("compat.") for k in ck["state_dict"])
    assert ck["compat_feats"] == "sim,margin,radius,support,mem,rel"
