"""Phase 4G risk checks: causality, official-free training, definitions."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_train_source_has_no_gt_or_future():
    src = (ROOT / "src/orbit_msrouting/train_msrouting.py").read_text()
    assert "load_gt(" not in src
    assert "main_seed1027" not in src
    assert "val_gt_track_stream" not in src
    assert "reversed(" not in src
    assert "oracle_k" not in src.lower()
    assert "retroactive_relabeling" not in src.lower()
    assert "official" not in src


def test_eval_source_uses_official_only_for_validation():
    src = (ROOT / "src/orbit_msrouting/evaluate_msrouting.py").read_text()
    assert "main_seed1027" in src  # official path exists
    assert "assignment_from_preds" in src
    assert "reversed(" not in src
    assert "oracle_k" not in src.lower()
    assert "load_gt(" in src


def test_state_features_are_legal():
    from src.orbit_msrouting.state_features import (
        STATE_FEAT_ORDER,
        MemoryStateTracker,
    )
    assert "log_mem" in STATE_FEAT_ORDER
    assert "mean_support" in STATE_FEAT_ORDER
    assert "low_support_ratio" in STATE_FEAT_ORDER
    assert "mean_dispersion" in STATE_FEAT_ORDER
    forbidden = ["gt", "true", "purity", "final", "future", "oracle"]
    for f in STATE_FEAT_ORDER:
        assert not any(x in f.lower() for x in forbidden)
    tr = MemoryStateTracker()
    feats = set(tr.compute(None)) if False else None


def test_metric_reconciliation_numbers():
    path = ROOT / "outputs/iclr27_phase4g/audit/definition_summary.csv"
    rows = {r["stream"]: r for r in csv.DictReader(path.open())
            if r["method"] == "c1"}
    o = rows["official"]
    assert int(o["birth_origin_known"]) == 259
    assert int(o["ever_contaminated_by_known"]) == 308
    assert int(o["novel_free"]) == 180
    assert int(o["global_absorption_hub"]) == 178
    assert int(o["novel_absorption_hub"]) == 83


def test_hub_origin_split():
    path = ROOT / "outputs/iclr27_phase4g/audit/hub_definition_check.csv"
    hubs = [r for r in csv.DictReader(path.open())
            if r["method"] == "c1" and r["stream"] == "official"
            and r["novel_absorption_hub"] == "1"]
    known = sum(1 for r in hubs if r["birth_origin_role"] == "known")
    novel = sum(1 for r in hubs if r["birth_origin_role"] == "novel")
    assert len(hubs) == 83
    assert known == 37
    assert novel == 46


def test_static_threshold_pareto_exists():
    p = ROOT / "outputs/iclr27_phase4g/audit/static_threshold_pareto.csv"
    assert p.exists()
    rows = list(csv.DictReader(p.open()))
    assert len(rows) >= 4
    assert rows[0]["gate_threshold"] in {"0.4", "0.45", "0.5", "0.55"}


def test_model_zero_init_preserves_m2():
    import torch
    from src.orbit_msrouting.model import load_msrouting_model
    m0, _ = load_msrouting_model("runs/orbit_mdc/mdc_m2/model.pth", "cpu")
    e = torch.randn(1, 11)
    with torch.no_grad():
        g0 = m0.gate_logit(e)
    m1, _ = load_msrouting_model("runs/orbit_mdc/mdc_m2/model.pth", "cpu",
                                 gate_mode="G1", state_dim=4)
    with torch.no_grad():
        g1 = m1.gate_logit(e, torch.zeros(1, 4))
    assert float((g0 - g1).abs().max()) < 1e-5
    m2, _ = load_msrouting_model("runs/orbit_mdc/mdc_m2/model.pth", "cpu",
                                 gate_mode="G2", state_dim=4)
    with torch.no_grad():
        g2 = m2.gate_logit(e, torch.zeros(1, 4))
    assert float((g0 - g2).abs().max()) < 1e-5


def test_freeze_sha_format():
    p = ROOT / "outputs/orbit_msrouting/frozen_candidates/candidate_a.json"
    if not p.exists():
        return  # freeze happens after training; test is conditional
    import json
    doc = json.loads(p.read_text())
    assert doc["checkpoint_sha256"]
    assert len(doc["checkpoint_sha256"]) == 64
    assert doc["official_validation"] == "PENDING"


def test_no_evaluator_modification():
    # evaluator source was frozen in Phase 4B/4C; check its hash is unchanged
    # from the phase-4C manifest if present; otherwise skip.
    import json
    manifest = ROOT / "outputs/iclr27_phase4c/audit/input_hashes.json"
    if not manifest.exists():
        return
    doc = json.loads(manifest.read_text())
    for rel, sha in doc.items():
        if "evaluator" not in rel:
            continue
        p = ROOT / rel
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == sha
