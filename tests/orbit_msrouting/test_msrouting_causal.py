"""ORBIT-MSRouting causal-contract tests."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_state_gate_in_onpolicy_rollout():
    src = (ROOT / "src/orbit_mdc/train_onpolicy.py").read_text()
    assert "gate_mode" in src
    assert "mem.state_summary()" in src
    assert "action = \"NEW\"" in src


def test_residual_bias_is_additive_not_reclassification():
    src = (ROOT / "src/orbit_iam/model.py").read_text()
    assert "gate_logit_with_bias" in src
    assert "logit - b" in src


def test_no_oracle_or_future():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    code = src.split('"""', 2)[2]
    assert "oracle" not in code.lower()
    assert "rows[i +" not in code
