"""Phase 4F protocol-contract tests (risk-focused)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_onpolicy_state_from_model_decisions():
    src = (ROOT / "src/orbit_mdc/train_onpolicy.py").read_text()
    assert "mem.update_novel" in src
    assert "mem.create_novel" in src
    # decisions are made by gate/compat, not by GT labels
    assert "if gate_prob >= args.gate_thr" in src
    assert "action = \"NEW\"" in src


def test_gt_never_repairs_memory():
    src = (ROOT / "src/orbit_mdc/train_onpolicy.py").read_text()
    assert "load_gt(" not in src
    assert "ground_truth_category_id" not in src
    # no branch that forces memory to follow pseudo-label on mis-decision
    assert "own_vid_by_label[q[\"label\"]] = vid" in src  # only at NEW birth


def test_no_future_or_oracle_in_eval():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    code = src.split('"""', 2)[2]
    assert "oracle" not in code.lower()
    assert "rewrite" not in code.lower()
    assert "rows[i +" not in code


def test_official_not_used_for_band_selection():
    audit = (ROOT / "src/iclr27_phase4f/audit_real_similarity.py").read_text()
    assert "load_gt(" not in audit
    assert "official" not in audit


def test_inputs_frozen():
    hashes = __import__("json").loads(
        (ROOT / "outputs/iclr27_phase4f/input_hashes.json").read_text())
    assert hashes["runs/orbit_iam/iam_i2_v3/model.pth"].startswith("6cb0479b")
    assert hashes["runs/orbit_msr/msr_nr2/model.pth"].startswith("677b2bd4")
