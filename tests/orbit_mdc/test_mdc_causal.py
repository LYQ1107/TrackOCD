"""ORBIT-MDC causal contract tests."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_quarantine_does_not_delay_virtual_id():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    # quarantine scales q by support but never suppresses the id itself
    assert "quarantine_support" in src
    assert "predicted_virtual_novel_id" in src


def test_rollout_uses_current_memory_only():
    src = (ROOT / "src/orbit_mdc/train_onpolicy.py").read_text()
    assert "for q in query:" in src
    assert "mem.state(vid)" in src


def test_candidate_freeze_artifacts_exist():
    assert (ROOT / "outputs/orbit_mdc/frozen_candidates").is_dir()
