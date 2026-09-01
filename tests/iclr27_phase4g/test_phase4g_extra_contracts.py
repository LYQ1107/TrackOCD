"""Additional Phase 4G contract tests (risk list items 1-20)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def _read(path):
    return (ROOT / path).read_text()


def test_definition_reconciliation_has_unified_names():
    doc = _read("docs/iclr27_phase4g/METRIC_DEFINITION_RECONCILIATION.md")
    assert "BOKP" in doc and "NFP" in doc and "ECKP" in doc
    assert "GAH" in doc and "NAH" in doc and "CWAH" in doc


def test_definition_check_csvs_exist_and_cover_c1_iam_m2():
    for name in ("prototype_origin_definition_check.csv",
                 "hub_definition_check.csv"):
        p = ROOT / "outputs/iclr27_phase4g/audit" / name
        rows = list(csv.DictReader(p.open()))
        methods = {r["method"] for r in rows}
        streams = {r["stream"] for r in rows}
        assert {"c1", "iam", "m2"} <= methods
        assert {"official", "long"} <= streams


def test_routing_bias_audit_outputs_present():
    for name in ("routing_bias_by_similarity_official.csv",
                 "routing_bias_by_margin_official.csv",
                 "routing_bias_probe_official.csv",
                 "routing_bias_probe_long.csv"):
        assert (ROOT / "outputs/iclr27_phase4g/audit" / name).exists()


def test_state_features_only_current_memory():
    src = _read("src/orbit_msrouting/state_features.py")
    src = src.split('"""', 2)[2]  # skip docstring that only discusses
    # forbidden quantities in order to forbid them
    assert "final" not in src
    assert "purity" not in src
    assert "true novel" not in src.lower()
    assert "future " not in src  # "from __future__" is a language import


def test_no_official_access_in_training():
    for p in ("src/orbit_msrouting/train_msrouting.py",
              "src/orbit_mdc/train_onpolicy.py"):
        src = _read(p)
        assert "load_gt(" not in src
        assert "seed1027" not in src
        assert "official" not in src


def test_no_future_or_oracle_in_eval():
    src = _read("src/orbit_msrouting/evaluate_msrouting.py")
    src = src.split('"""', 2)[2]
    assert "oracle" not in src.lower()
    assert "rows[i +" not in src
    assert "rows[i+1" not in src


def test_candidate_freeze_script_records_sha():
    src = _read("src/orbit_msrouting/freeze_candidate.py")
    assert "sha256" in src.lower()
    assert "checkpoint_sha256" in src


def test_evaluator_and_tracking_not_modified():
    # Phase 4G work must not touch the frozen evaluator or tracking configs.
    src = _read("src/orbit_msrouting/evaluate_msrouting.py")
    assert "HOTA" not in src
    assert "ByteTrack" not in src
    assert "SimOWT" not in src


def test_input_hashes_json_valid():
    p = ROOT / "outputs/iclr27_phase4g/input_hashes.json"
    h = json.loads(p.read_text())
    assert h
    for path, digest in h.items():
        assert len(digest) == 64
        assert (ROOT / path).exists()


def test_old_outputs_not_overwritten():
    old = ROOT / "outputs/orbit_mdc/results/official_comparison.csv"
    assert old.exists()
    old2 = ROOT / "outputs/iclr27_phase4f/audit/memory_trajectory_m2_official.csv"
    assert old2.exists()


def test_static_threshold_pareto_present():
    p = ROOT / "outputs/iclr27_phase4g/audit/static_threshold_pareto.csv"
    rows = list(csv.DictReader(p.open()))
    thrs = sorted(float(r["gate_threshold"]) for r in rows)
    assert thrs == [0.4, 0.45, 0.5, 0.55]


def test_controlled_routing_cases_present():
    assert (ROOT / "outputs/iclr27_phase4g/audit/"
            "controlled_routing_cases.csv").exists()
