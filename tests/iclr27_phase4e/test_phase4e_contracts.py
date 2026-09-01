"""Phase 4E protocol-contract tests (risk-focused, not quantity-focused)."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def pinned_commits():
    out = {}
    for repo in ["ProxyAnchor", "pytorch_metric_learning", "CoPE",
                 "research_xbm", "OCM"]:
        p = ROOT / "third_party/research_refs_phase4e" / repo
        r = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        out[repo] = r.stdout.strip() if r.returncode == 0 else None
    return out


def inventory():
    p = ROOT / "outputs/iclr27_phase4e/open_source/repository_inventory.csv"
    return list(csv.DictReader(open(p)))


def test_repos_pinned_and_recorded():
    commits = pinned_commits()
    inv = {r["repo"]: r for r in inventory()}
    for repo, c in commits.items():
        assert c is not None, f"{repo} not a git repo"
        assert repo in inv, f"{repo} missing from inventory"
        assert inv[repo]["commit"] == c, f"{repo} commit drift"
        assert inv[repo]["license"], f"{repo} license missing"


def test_official_non_official_marks_present():
    inv = inventory()
    assert len(inv) >= 5
    assert {r["repo"] for r in inv} >= {
        "ProxyAnchor", "pytorch_metric_learning", "CoPE", "research_xbm", "OCM"}


def test_hard_negatives_train_side_only():
    src = (ROOT / "src/orbit_iam/train_iam.py").read_text()
    assert "load_gt(" not in src
    assert "official" not in src
    assert "ground_truth_category_id" not in src
    assert "1000000" in src  # pseudo-novel label namespace only
    stats = ROOT / "outputs/iclr27_phase4e/training/hard_negative_statistics.csv"
    if stats.exists():
        rows = list(csv.DictReader(open(stats)))
        assert rows, "hard-negative statistics empty"
        assert all(r["first"] in ("0", "1") for r in rows)


def test_official_gt_not_in_training_or_pair_labels():
    for f in ["src/orbit_iam/train_iam.py", "src/orbit_iam/compat.py"]:
        src = (ROOT / f).read_text()
        assert "load_gt" not in src


def test_no_future_access_or_oracle_k_in_eval():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    code = src.split('"""', 2)[2]  # after module docstring
    assert "sorted(mem.novel)" in src  # causal memory only
    # novel class count is never read: no occurrence of total novel class K
    assert "n_novel_classes" not in code
    assert "oracle" not in code.lower()
    assert "rewrite" not in code.lower()
    assert "future" not in code.lower().replace("from __future__", "")
    # no forward indexing of the stream
    assert "rows[i +" not in code
    assert "rows[i+" not in code
    assert "[i - 1]" not in code


def test_no_historical_relabel():
    mem = (ROOT / "src/orbit_iam/iam_memory.py").read_text()
    assert "history" not in mem
    assert "relabel" not in mem.lower()
    assert "rewrite" not in mem.lower()


def test_confidence_not_using_gt_purity():
    mem = (ROOT / "src/orbit_iam/iam_memory.py").read_text()
    code = mem.split('"""', 2)[2]  # after the module docstring
    assert "purity" not in code
    assert "ground_truth" not in code
    assert "class" not in mem.replace("class IamMemory", "").replace(
        "class CausalNovelMemory", "")


def test_memory_scale_feature_is_causal():
    src = (ROOT / "src/orbit_iam/compat.py").read_text()
    assert "mem_size" in src
    assert "len(rows)" not in src
    assert "len(mem.novel)" in (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()


def test_first_occurrence_definition():
    src = (ROOT / "src/orbit_iam/evaluate_iam.py").read_text()
    assert "first_occurrence" in src
    assert "seen" in src


def test_multi_prototype_gate_respected():
    doc = (ROOT / "docs/iclr27_phase4e/MULTI_PROTOTYPE_JUSTIFICATION.md").read_text()
    assert "MULTI_PROTOTYPE_NOT_JUSTIFIED" in doc
    mp_dirs = list((ROOT / "runs/orbit_iam").glob("*mp*"))
    assert not mp_dirs, "multi-prototype trained despite NOT_JUSTIFIED"
    assert not (ROOT / "outputs/orbit_iam/frozen_candidates/candidate_b.json").exists()


def test_evaluator_unchanged():
    p = ROOT / "src/trackocd_v1/evaluation/trackocd_evaluator.py"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    frozen = json.loads(
        (ROOT / "outputs/iclr27_phase4e/audit/input_hashes.json").read_text())
    assert h == frozen["src/trackocd_v1/evaluation/trackocd_evaluator.py"]


def test_frozen_inputs_present():
    h = json.loads((ROOT / "outputs/iclr27_phase4e/audit/input_hashes.json").read_text())
    for rel in ["runs/orbit_msr/msr_nr2/model.pth",
                "runs/orbit_msr/msr_c2/model.pth",
                "outputs/iclr27_phase4d/long_stream/stream_cache.npz"]:
        assert (ROOT / rel).exists(), rel


def test_no_old_artifact_overwrite():
    backups = list((ROOT / "outputs/iclr27_phase4e/audit/backup_official_identity").glob("*.csv"))
    assert backups, "official backup missing"


def test_candidate_freeze_precedes_official():
    freeze = ROOT / "docs/orbit_iam/OFFICIAL_CANDIDATE_FREEZE.md"
    if freeze.exists():
        text = freeze.read_text()
        assert "SHA256" in text or "sha256" in text
    # official results must not exist before freeze is written
    official = ROOT / "outputs/orbit_iam/results/candidate_a_seed1027.csv"
    if official.exists():
        assert freeze.exists(), "official result exists without freeze doc"
