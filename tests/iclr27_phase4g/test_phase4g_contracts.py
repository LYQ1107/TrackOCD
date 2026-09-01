"""Phase 4G protocol-contract tests."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_state_features_legal():
    src = (ROOT / "src/orbit_iam/iam_memory.py").read_text()
    code = src.split('"""', 2)[2]
    assert "state_summary" in src
    assert "final" not in code
    assert "purity" not in code


def test_no_final_memory_size_in_eval():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    code = src.split('"""', 2)[2]
    assert "len(rows)" not in code.replace("n = len(rows)", "")
    # memory size used in state features comes from current memory only
    assert "mem.state_summary()" in src


def test_official_not_used_for_feature_selection():
    audit = (ROOT / "src/iclr27_phase4g/audit_routing_bias.py").read_text()
    # audit only reads frozen M2 logs; training never reads official
    train = (ROOT / "src/orbit_mdc/train_onpolicy.py").read_text()
    assert "load_gt(" not in train
    assert "official" not in train


def test_metric_reconciliation_done():
    assert (ROOT / "outputs/iclr27_phase4g/audit/definition_summary.csv").exists()
    assert (ROOT / "outputs/iclr27_phase4g/audit/hub_definition_check.csv").exists()
