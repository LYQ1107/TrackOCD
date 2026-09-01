"""Phase 4J risk-focused contract tests (task section 83)."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.semantic import SemanticStateManager


def _mgr(commit_mode="M1"):
    return SemanticStateManager(
        model=None, known_protos={1: np.ones(4, dtype=np.float32)},
        radii={1: 0.1}, device=torch.device("cpu"),
        prefix_mode="P1", theta_novel=0.6, memo_tracklet_frames=10,
        decision_threshold=0.15, commit_mode=commit_mode,
        commit_min_age=2, commit_min_support=2)


def _obs(p_known=0.1):
    return {
        "z": np.random.RandomState(5).randn(4).astype(np.float32),
        "p_known": p_known, "class_dist": np.ones(1, dtype=np.float32),
        "best_known": 0.3, "best_novel": 0.0, "novel_id": None,
        "novel_conf": 0.0, "reliability": 1.0, "rel": 1.0,
    }


def test_observation_before_association():
    src = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "sem_cost = sem_manager.semantic_cost_matrix(" in src
    assert src.index("sem_cost = sem_manager") < \
        src.index("tracker.match(")


def test_uncommitted_semantics_still_affect_association():
    m = _mgr()
    m._apply_association(0, [(1, _obs(), 1.0)])
    t = m.tracks[1]
    assert t.committed_sem_id is None
    # a novel-like detection of the same physical track inherits the
    # track-local provisional identity, so soft semantics stay positive
    M = m.semantic_cost_matrix([_obs(p_known=0.1)], [1])
    assert M[0, 0] > 0.0


def test_uncommitted_fp_does_not_enter_global_memory():
    m = _mgr()
    m._apply_association(0, [(7, _obs(), 1.0)])
    assert m.memory.size() == 0
    assert m.tracks[7].novel_id == "L7"


def test_committed_novel_enters_global_memory():
    m = _mgr()
    m._apply_association(0, [(7, _obs(), 1.0)])
    m._apply_association(1, [(7, _obs(), 1.0)])
    assert m.memory.size() == 1
    assert m.tracks[7].committed_sem_id in m.memory.protos


def test_existing_novel_matching_uses_committed_history_only():
    m = _mgr()
    m._apply_association(0, [(7, _obs(), 1.0)])
    # provisional candidate is not in global memory
    assert all(not isinstance(k, str) for k in m.memory.protos)


def test_no_future_or_full_track_in_semantic_source():
    src = (ROOT / "src/frame_online_trackocd/semantic.py").read_text()
    assert "reversed(" not in src
    assert "feats_all" not in src
    assert "val_gt" not in src and "oracle" not in src


def test_detector_frozen_and_b0_unchanged():
    src = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "det_bboxes" in src and "torch.from_numpy(z[\"det_bboxes\"])" in src
    eq = json.loads((ROOT / "outputs/frame_online_trackocd/subset/"
                     "b0_equivalence.json").read_text())
    assert len(eq) == 20 and all(r["ok"] for r in eq)


def test_calibration_uses_train_side_only():
    src = (ROOT / "src/iclr27_phase4j/"
           "train_side_gate_calibration.py").read_text()
    assert "load_frame_features(\"train_known_mean\")" in src
    assert "validation_20" not in src
    assert "tao_subset" not in src


def test_candidate_config_frozen_before_full_val():
    cfg = json.loads((ROOT / "outputs/iclr27_phase4j/semantic_logs/"
                      "J1_config.json").read_text())
    assert cfg["decision_threshold"] == "0.15"
    assert cfg["checkpoint"] == "runs/orbit_mdc/mdc_m2/model.pth"


def test_open_source_commits_verified_and_licenses_recorded():
    inv = list(csv.DictReader(open(
        ROOT / "outputs/iclr27_phase4j/open_source/repository_inventory.csv")))
    phase4j_rows = [r for r in inv if r["phase"] == "phase4j"
                    and r["repo"].startswith(("weilllllls", "Fulin-Gao",
                                              "inha-vllab", "fanlyu",
                                              "ynanwu", "Ashengl"))]
    assert phase4j_rows
    for r in phase4j_rows:
        name = r["repo"].split("/")[-1]
        if name == "AGE":
            repo = ROOT / "third_party/research_refs_phase4f/AGE"
        else:
            cands = [p for p in (ROOT / "third_party/iclr27_phase4j").iterdir()
                     if p.is_dir() and p.name.lower() == name.lower()]
            assert cands, (name, r["repo"])
            repo = cands[0]
        got = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True).strip()
        assert got.startswith(r["commit"]), (name, got, r["commit"])
        assert r["license"] != ""


def test_old_outputs_preserved():
    old = ROOT / "outputs/frame_online_trackocd/subset/B2/l0.1/0000003423.json"
    assert old.exists()
    old_log = ROOT / "outputs/iclr27_phase4i/audit/semantic_logs/" \
        "B2_l0.1/88.jsonl"
    assert old_log.exists()
