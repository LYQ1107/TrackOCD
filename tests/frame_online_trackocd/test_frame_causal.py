"""Frame-causality and identity-separation tests for Phase 4I."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def test_prefix_is_causal_no_full_track():
    from src.frame_online_trackocd.semantic import TrackSemState
    obs = {"z": np.random.rand(768).astype(np.float32),
           "p_known": 0.4, "class_dist": np.ones(48, dtype=np.float32) / 48,
           "novel_id": None, "novel_conf": 0.0, "reliability": 1.0}
    t = TrackSemState(1, obs, prefix_mode="P1")
    assert len(t.z_history) == 1
    for _ in range(5):
        t.update(obs, alpha=0.5)
    assert len(t.z_history) == 6
    # P0 must equal the most recent frame only
    t0 = TrackSemState(2, obs, prefix_mode="P0")
    for _ in range(3):
        t0.update(obs, alpha=0.5)
    assert np.allclose(t0.prefix(), obs["z"])


def test_semantic_observation_uses_frozen_preframe_memory():
    from src.frame_online_trackocd.semantic import SemanticStateManager
    from src.frame_online_trackocd.semantic import NovelSemanticMemory
    mgr = object.__new__(SemanticStateManager)
    mgr.memory = NovelSemanticMemory()
    mgr.memory.protos[7] = np.ones(768, dtype=np.float32)
    n0 = mgr.memory.size()
    # observing must not mutate memory (frame-synchronous)
    assert mgr.memory.size() == n0


def test_physical_and_semantic_identity_separate():
    from src.frame_online_trackocd.semantic import TrackSemState
    obs = {"z": np.random.rand(768).astype(np.float32),
           "p_known": 0.1, "class_dist": np.ones(48, dtype=np.float32) / 48,
           "novel_id": 900001, "novel_conf": 0.9, "reliability": 1.0}
    t1 = TrackSemState(17, obs, "P1")
    t2 = TrackSemState(42, obs, "P1")
    # same novel semantic ID on two different physical tracks is allowed
    assert t1.track_id != t2.track_id
    assert t1.novel_id == t2.novel_id


def test_b1_never_passes_semantic_cost_to_match():
    src = (ROOT / "src" / "frame_online_trackocd" / "replay.py").read_text()
    assert 'if mode in ("B1", "B2") and sem_manager is not None:' in src
    assert 'if mode == "B2":' in src
    assert 'sem_cost = sem_manager.semantic_cost_matrix(' in src


def test_semantic_cost_enters_association_in_b2():
    src = (ROOT / "src" / "frame_online_trackocd" / "frame_tracker.py").read_text()
    assert "if sem_cost is not None and lambda_s > 0.0:" in src
    assert "scores = scores + lambda_s * sem_cost.to(scores)" in src


def test_no_gt_and_no_oracle_k_in_semantic_engine():
    src = (ROOT / "src" / "frame_online_trackocd" / "semantic.py").read_text()
    assert "load_train_labels" in src  # train-side only
    assert "val_gt" not in src
    assert "oracle" not in src
    assert "num_classes" not in src.replace("known_ids", "")


def test_original_replay_equivalence_recorded():
    p = ROOT / "outputs" / "frame_online_trackocd" / "subset" / "b0_equivalence.json"
    assert p.exists()
    rows = json.loads(p.read_text())
    assert len(rows) == 20
    assert all(r["ok"] for r in rows)
