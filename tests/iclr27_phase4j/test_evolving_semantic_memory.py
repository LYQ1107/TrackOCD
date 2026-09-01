"""Phase 4J: observation/commitment separation tests (risk-focused)."""
from __future__ import annotations

import numpy as np
import torch

from src.frame_online_trackocd.semantic import (
    SemanticStateManager,
    TrackSemState,
)


def _mgr(commit_mode="M0", decision_threshold=0.5, commit_min_age=2,
         commit_min_support=2, split_age=None):
    m = SemanticStateManager(
        model=None, known_protos={1: np.ones(4, dtype=np.float32)},
        radii={1: 0.1}, device=torch.device("cpu"),
        prefix_mode="P1", theta_novel=0.6, memo_tracklet_frames=10,
        decision_threshold=decision_threshold,
        decision_split_age=split_age,
        commit_mode=commit_mode, commit_min_age=commit_min_age,
        commit_min_support=commit_min_support)
    return m


def _obs(p_known=0.1, novel_id=None, novel_conf=0.0, z=None):
    if z is None:
        z = np.random.RandomState(3).randn(4).astype(np.float32)
    z = z / (np.linalg.norm(z) + 1e-12)
    return {
        "z": z,
        "p_known": p_known,
        "class_dist": np.ones(1, dtype=np.float32),
        "best_known": 0.3, "best_novel": novel_conf,
        "novel_id": novel_id, "novel_conf": novel_conf,
        "reliability": 1.0,
        "rel": 1.0,
    }


def test_m0_commits_new_novel_immediately():
    m = _mgr(commit_mode="M0")
    o = _obs()
    m._apply_association(0, [(1, o, 1.0)])
    t = m.tracks[1]
    assert t.committed_sem_id is not None
    assert t.commit_action == "NEW_NOVEL"
    assert m.memory.size() == 1
    assert len(m.memory.commit_events) == 1


def test_m1_delays_new_novel_but_keeps_soft_identity():
    m = _mgr(commit_mode="M1", commit_min_age=2, commit_min_support=2)
    o1 = _obs()
    m._apply_association(0, [(1, o1, 1.0)])
    t = m.tracks[1]
    assert t.committed_sem_id is None
    assert m.memory.size() == 0
    assert t.novel_id == "L1"          # provisional identity active
    assert t.last_action == "PROVISIONAL_NOVEL"
    o2 = _obs()
    m._apply_association(1, [(1, o2, 1.0)])
    t = m.tracks[1]
    assert t.committed_sem_id is not None
    assert t.commit_action == "NEW_NOVEL"
    assert t.commit_age == 2
    assert m.memory.size() == 1


def test_uncommitted_existing_novel_uses_global_id_without_update():
    m = _mgr(commit_mode="M1", commit_min_age=3, commit_min_support=2)
    m.memory.protos[900001] = _obs()["z"]
    m.memory.support[900001] = 1
    m.memory.next_id = 900002
    o = _obs(p_known=0.1, novel_id=900001, novel_conf=0.9)
    m._apply_association(0, [(1, o, 1.0)])
    t = m.tracks[1]
    assert t.committed_sem_id is None
    assert t.novel_id == 900001          # global soft identity
    assert m.memory.support[900001] == 1  # no write yet


def test_provisional_identity_is_track_local():
    m = _mgr(commit_mode="M1")
    o = _obs()
    m._apply_association(0, [(1, o, 1.0), (2, o, 1.0)])
    assert m.tracks[1].novel_id == "L1"
    assert m.tracks[2].novel_id == "L2"


def test_calibrated_threshold_changes_routing_decision():
    m = _mgr(commit_mode="M1", decision_threshold=0.2)
    o = _obs(p_known=0.35)
    m._apply_association(0, [(1, o, 1.0)])
    t = m.tracks[1]
    assert t.last_action == "KNOWN"
    assert t.novel_id is None


def test_two_band_threshold_uses_age_split():
    m = _mgr(commit_mode="M1", decision_threshold=(0.5, 0.2),
             split_age=2)
    assert m.decision_threshold(1) == 0.5
    assert m.decision_threshold(2) == 0.5
    assert m.decision_threshold(3) == 0.2


def test_commit_events_record_age_and_support():
    m = _mgr(commit_mode="M1", commit_min_age=2, commit_min_support=2)
    m._apply_association(0, [(1, _obs(), 1.0)])
    m._apply_association(1, [(1, _obs(), 1.0)])
    ev = m.memory.commit_events
    assert len(ev) == 1
    assert ev[0]["action"] == "NEW_NOVEL"
    assert ev[0]["age"] == 2 and ev[0]["support"] == 2
    assert ev[0]["frame_id"] == 1


def test_log_row_reports_provisional_vs_committed_state():
    m = _mgr(commit_mode="M1", commit_min_age=2, commit_min_support=2)
    m._apply_association(0, [(1, _obs(), 1.0)])
    row = m.log_row(0, 0, [_obs()], 1, 1.0, [0, 0, 1, 1])
    assert row["semantic_action"] == "PROVISIONAL_NOVEL"
    assert row["commit_state"] == "provisional"
    assert row["global_novel_id"] is None
    m._apply_association(1, [(1, _obs(), 1.0)])
    row = m.log_row(1, 0, [_obs()], 1, 1.0, [0, 0, 1, 1])
    assert row["semantic_action"] == "NEW_NOVEL"
    assert row["commit_state"] == "committed"
    assert row["global_novel_id"] is not None


def test_no_future_gt_oracle_in_engine_source():
    src = __import__("pathlib").Path(
        "src/frame_online_trackocd/semantic.py").read_text()
    assert "val_gt" not in src
    assert "oracle" not in src


def test_m0_novel_known_novel_births_by_current_evidence_not_stale_id():
    """Phase 4I M0 regression: a track that flips known clears its active
    novel identity; a later novel phase with no compatible prototype must
    birth a NEW id, not silently reuse the stale id (P0 prefix)."""
    m = SemanticStateManager(
        model=None, known_protos={1: np.ones(4, dtype=np.float32)},
        radii={1: 0.1}, device=torch.device("cpu"),
        prefix_mode="P0", theta_novel=0.6, memo_tracklet_frames=10,
        decision_threshold=0.5, commit_mode="M0")
    m._apply_association(0, [(1, _obs(p_known=0.1), 1.0)])
    first = m.tracks[1].committed_sem_id
    assert first is not None
    m._apply_association(1, [(1, _obs(p_known=0.9), 1.0)])
    assert m.tracks[1].novel_id is None
    assert m.tracks[1].last_action == "KNOWN"
    z2 = -_obs()["z"]               # far from the first prototype
    m._apply_association(2, [(1, _obs(p_known=0.1, z=z2), 1.0)])
    second = m.tracks[1].novel_id
    assert isinstance(second, int)
    assert second != first          # new evidence, not stale id reuse
    assert m.tracks[1].last_action == "NEW_NOVEL"
    assert m.memory.size() == 2
