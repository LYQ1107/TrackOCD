"""ORBIT-MSRouting unit tests: state legality, init correctness, gating."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def make_mem():
    from src.orbit_iam.iam_memory import IamMemory
    protos = {1: np.ones(16, dtype=np.float32) / math.sqrt(16)}
    mem = IamMemory(protos, {1: 0.3})
    mem.create_novel(np.ones(16, dtype=np.float32) / math.sqrt(16),
                     created_at=0)
    mem.create_novel(-np.ones(16, dtype=np.float32) / math.sqrt(16),
                     created_at=1)
    return mem


def test_state_feature_legal_bounds():
    from src.orbit_msrouting.state_features import MemoryStateTracker, \
        STATE_FEAT_ORDER
    mem = make_mem()
    tr = MemoryStateTracker(window=8)
    v = tr.compute(mem, STATE_FEAT_ORDER)
    assert len(v) == len(STATE_FEAT_ORDER)
    for x, name in zip(v, STATE_FEAT_ORDER):
        assert 0.0 <= x <= 1.0, name
    assert v[STATE_FEAT_ORDER.index("log_mem")] > 0.0
    assert v[STATE_FEAT_ORDER.index("mean_support")] > 0.0


def test_state_tracker_uses_recent_actions():
    from src.orbit_msrouting.state_features import MemoryStateTracker
    mem = make_mem()
    tr = MemoryStateTracker(window=4)
    tr.note_action("NEW_NOVEL", 1)
    tr.note_action("KNOWN")
    tr.note_action("EXISTING_NOVEL", 2)
    v = tr.compute(mem, ["recent_birth_rate", "recent_reuse_rate"])
    assert abs(v[0] - 1 / 3) < 1e-6
    assert abs(v[1] - 1 / 3) < 1e-6


def test_g2_zero_calibration_init():
    from src.orbit_msrouting.model import build_msrouting_model
    ck = ROOT / "runs/orbit_mdc/mdc_m2/model.pth"
    model, _ = build_msrouting_model(str(ck), "G2", ["log_mem"], "cpu")
    with torch.no_grad():
        x = torch.randn(1, 1)
        b = model.calib(x)
    assert torch.allclose(b, torch.zeros_like(b), atol=1e-6)


def test_g1_gate_copy_and_state_path():
    from src.orbit_msrouting.model import build_msrouting_model
    ck = ROOT / "runs/orbit_mdc/mdc_m2/model.pth"
    model, _ = build_msrouting_model(str(ck), "G1", ["log_mem"], "cpu")
    base = torch.load(ck, map_location="cpu")["state_dict"]
    w = model.gate.net[0].weight
    assert w.shape[1] == 12
    assert torch.allclose(w[:, :11], base["gate.net.0.weight"])
    assert torch.allclose(w[:, 11:], torch.zeros_like(w[:, 11:]))


def test_gate_modes_reject_missing_state():
    from src.orbit_msrouting.model import build_msrouting_model
    ck = ROOT / "runs/orbit_mdc/mdc_m2/model.pth"
    model, _ = build_msrouting_model(str(ck), "G2", ["log_mem"], "cpu")
    ev = torch.randn(1, 11)
    try:
        model.gate_logit(ev, None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("G2 must require state features")


def test_g0_is_m2_gate():
    from src.orbit_msrouting.model import build_msrouting_model
    ck = ROOT / "runs/orbit_mdc/mdc_m2/model.pth"
    model, _ = build_msrouting_model(str(ck), "G0", [], "cpu")
    base = torch.load(ck, map_location="cpu")["state_dict"]
    for k, v in base.items():
        if k in model.state_dict():
            assert torch.allclose(model.state_dict()[k], v, atol=1e-6), k


def test_bucket_rows_outputs_columns():
    from src.orbit_msrouting.evaluate_msrouting import bucket_rows
    rows = [{
        "sample_id": "a", "arrival_index": 0, "role": "novel",
        "class": 1, "true_role": "novel", "true_class": 1,
        "first_occurrence": True, "predicted_action": "KNOWN",
        "predicted_known_id": 1, "predicted_virtual_novel_id": None,
        "active_novel_prototypes": 10,
    }]
    gt = [{"sample_id": "a", "protocol_role": "novel",
           "ground_truth_category_id": 1}]
    out = bucket_rows(rows, gt, "test")
    assert out and out[0]["bucket"] == "0-32"
    assert "n2k" in out[0] and "ari" in out[0] and "count_error" in out[0]
