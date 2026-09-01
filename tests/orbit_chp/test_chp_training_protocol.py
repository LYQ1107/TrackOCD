"""Real-risk tests for CHP training: frozen architecture, causal memory,
train-side hardness, shared evaluator."""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def _load(path):
    return torch.load(str(path), map_location="cpu")["state_dict"]


def test_aggregator_frozen_during_chp_training():
    anchor = _load(ROOT / "runs" / "orbit_mdc" / "mdc_m2" / "model.pth")
    for variant in ("chp_h1", "chp_h2", "chp_h3"):
        sd = _load(ROOT / "runs" / "orbit_chp" / variant / "model.pth")
        for k, v in anchor.items():
            if k.startswith("aggregator"):
                assert torch.equal(v, sd[k]), (variant, k)


def test_checkpoint_metadata_records_mode_and_seed():
    for variant, mode in (("chp_h1", "random"),
                          ("chp_h2", "hard"),
                          ("chp_h3", "mixed")):
        ck = torch.load(str(ROOT / "runs" / "orbit_chp" / variant
                            / "model.pth"), map_location="cpu")
        assert ck["episode_mode"] == mode
        assert ck["seed"] == 1027
        assert ck["compat_dim"] == 7


def test_on_policy_memory_not_gt_fixed():
    src = (ROOT / "src" / "orbit_chp" / "train_chp.py").read_text()
    # memory updates must be conditioned on the model's predicted action,
    # not on the pseudo-novel ground-truth label.
    assert 'if action == "KNOWN":' in src
    assert 'elif action == "EXISTING" and nid is not None:' in src
    assert 'else:' in src
    # GT labels are only consumed by losses (gate/compat/known supervision).
    assert 'gate_target = torch.tensor([1.0 if q["known"] else 0.0]' in src
    # no oracle-history / GT memory repair field anywhere in the episode path
    assert "gt_memory" not in src
    assert "oracle_memory" not in src
    assert "relabel" not in src


def test_tier_boundaries_train_side_only():
    from src.orbit_chp.eval_proxy import pool_hardness_distribution
    dist = pool_hardness_distribution()
    assert len(dist) == 38
    src = (ROOT / "src" / "orbit_chp" / "eval_proxy.py").read_text()
    # no official Pure Full GT/stream is read for proxy evaluation
    assert "load_gt" not in src
    assert "main_seed1027" not in src


def test_evaluator_is_shared_not_forked():
    src = (ROOT / "src" / "orbit_chp" / "eval_proxy.py").read_text()
    assert "from src.orbit_msr.evaluate import attach_gt, summarize" in src
    # the shared MDC evaluator is imported (multi-line), never reimplemented
    import_block = src.split("from src.orbit_mdc.evaluate_mdc import", 1)[1]
    import_block = import_block.split(")", 1)[0]
    assert "evaluate_long_mdc" in import_block
    assert "run_mdc_stream" in import_block


def test_freeze_record_contains_sha256_and_protocol():
    src = (ROOT / "src" / "orbit_chp" / "freeze_candidate.py").read_text()
    assert '"sha256": sha256(args.checkpoint)' in src
    assert '"protocol": "Pure Full, seed1027, strict online causal, GT-track"' in src
