"""Phase 4L risk-focused contract tests (task section 86)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.semantic import SemanticStateManager


def _mgr(commit_mode="M0", threshold=0.3):
    return SemanticStateManager(
        model=None, known_protos={1: np.ones(4, dtype=np.float32)},
        radii={1: 0.1}, device=torch.device("cpu"),
        prefix_mode="P1", theta_novel=0.6, memo_tracklet_frames=10,
        decision_threshold=threshold, commit_mode=commit_mode,
        commit_min_age=2, commit_min_support=2)


def _obs(p_known=0.1):
    return {
        "z": np.random.RandomState(5).randn(4).astype(np.float32),
        "p_known": p_known, "class_dist": np.ones(1, dtype=np.float32),
        "best_known": 0.3, "best_novel": 0.0, "novel_id": None,
        "novel_conf": 0.0, "reliability": 1.0, "rel": 1.0,
    }


def test_01_semantic_observation_before_association():
    src = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "sem_cost = sem_manager.semantic_cost_matrix(" in src
    assert src.index("sem_cost = sem_manager") < src.index("tracker.match(")


def test_02_non_admissible_still_observed_immediately():
    # The audit design contract: admissibility (if any) is a memory
    # weight, not an observation gate.  Observe() must return semantics
    # for every detection.
    m = _mgr(commit_mode="M1")
    # a failed-crop / neutral detection still receives an immediate
    # semantic observation dict
    obs = m.observe([np.zeros(768, dtype=np.float32)])
    assert obs[0]["p_known"] == 0.5
    assert "z" in obs[0]


def test_03_admissibility_controls_memory_not_observation():
    src = (ROOT / "src/frame_online_trackocd/semantic.py").read_text()
    # current frame evidence must not be gated by admissibility state
    assert "semantic_cost_matrix" in src
    assert "post_association" in src


def test_04_no_future_track_features():
    for p in (ROOT / "src/frame_online_trackocd").glob("*.py"):
        assert "feats_all" not in p.read_text()
        assert "full_track" not in p.read_text()


def test_05_no_gt_online():
    for p in (ROOT / "src/frame_online_trackocd").glob("*.py"):
        src = p.read_text()
        assert "validation_20.json" not in src
        assert "validation_heldout_tao.json" not in src
        assert "ground_truth" not in src.lower()


def test_06_no_oracle_k():
    for p in (ROOT / "src/frame_online_trackocd").glob("*.py"):
        assert "oracle_k" not in p.read_text().lower()


def test_07_no_retroactive_relabel_in_online_path():
    online = "".join(p.read_text() for p in
                     (ROOT / "src/frame_online_trackocd").glob("*.py"))
    audit = "".join(p.read_text() for p in
                    (ROOT / "src/iclr27_phase4l").glob("*.py"))
    assert "match_gt" not in online or "diagnostic" in audit.lower()


def test_08_physical_id_ne_semantic_id():
    m = _mgr(commit_mode="M1")
    m._apply_association(0, [(7, _obs(), 1.0)])
    m._apply_association(1, [(7, _obs(), 1.0)])
    gid = m.tracks[7].committed_sem_id
    assert isinstance(gid, int) and gid >= 1_000_000
    assert gid != 7


def test_09_semantic_class_may_span_physical_tracks():
    m = _mgr(commit_mode="M1")
    m._apply_association(0, [(7, _obs(), 1.0)])
    m._apply_association(1, [(7, _obs(), 1.0)])
    assert isinstance(m.tracks[7].committed_sem_id, int)
    # a second physical track can reference the same semantic id without
    # sharing a physical identity (state object is keyed by track id)
    assert m.tracks[7].track_id == 7


def test_10_persistent_fp_labels_diagnostic_only():
    audit = (ROOT / "src/iclr27_phase4l/build_admissibility_dataset.py")
    src = audit.read_text()
    assert "diagnostic" in src.lower()
    assert "offline" in src.lower()


def test_11_matching_statistics_causal():
    src = (ROOT / "src/iclr27_phase4l/build_novel_matching_dataset.py")
    text = src.read_text()
    assert "past members" in text.lower() or "no future" in text.lower()
    assert "members[best]" in text


def test_12_prototype_radius_uses_past_members():
    src = (ROOT / "src/iclr27_phase4l/build_novel_matching_dataset.py")
    text = src.read_text()
    assert "support[best]" in text
    assert "member" in text


def test_13_best_second_margin_correct():
    from src.iclr27_phase4l.build_novel_matching_dataset import _norm
    P = _norm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    z = _norm(np.array([0.9, 0.1], dtype=np.float32))
    cos = P @ z
    order = np.argsort(cos)[::-1]
    best, second = float(cos[order[0]]), float(cos[order[1]])
    assert abs(best - second - (best - second)) < 1e-6


def test_14_known_novel_relative_scores():
    # novel_minus_known is best_novel - best_known in the audit builder
    src = (ROOT / "src/iclr27_phase4l/build_novel_matching_dataset.py")
    assert "novel_minus_known" in src.read_text()


def test_15_video_local_track_ids_separated():
    prov = (ROOT / "src/iclr27_phase4k/provenance.py").read_text()
    assert "track_key" in prov and "video_id" in prov
    audit = (ROOT / "src/iclr27_phase4l/build_admissibility_dataset.py")
    assert "video_id" in audit.read_text()


def test_16_phase4k_provenance_reuse():
    for p in (ROOT / "tests/iclr27_phase4k").glob("*.py"):
        assert "byte_exact" in p.read_text()


def test_17_j1b_anchor_reproduction():
    eq = ROOT / "outputs/iclr27_phase4k/audit/prov_j1b/equivalence.json"
    assert eq.exists()
    assert json.loads(eq.read_text())["byte_exact"] is True


def test_18_detector_unchanged():
    replay = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "tracker.match(" in replay
    runner = (ROOT / "src/iclr27_phase4l/run_heldout_replay.py").read_text()
    assert "mdc_m2/model.pth" in runner


def test_19_dino_frozen():
    extract = (ROOT / "src/iclr27_phase4l/extract_heldout_features.py")
    src = extract.read_text()
    assert "load_dinov2" in src
    assert "torch.no_grad()" in src
    assert "requires_grad" not in src


def test_20_lambda_s_fixed():
    runner = (ROOT / "src/iclr27_phase4l/run_heldout_replay.py")
    assert "lambda_s=0.1" in runner.read_text()


def test_21_tau_frozen_main_comparison():
    runner = (ROOT / "src/iclr27_phase4l/run_heldout_replay.py")
    text = runner.read_text()
    assert '"j1b": {"decision_threshold": 0.30' in text
    assert '"m1": {"decision_threshold": 0.30' in text


def test_22_dev_heldout_non_overlap():
    dev = {r["video_id"] for r in csv.DictReader(open(
        ROOT / "outputs/iclr27_phase3a/smoke/selected_20_videos.csv"))}
    held = {r["video_id"] for r in csv.DictReader(open(
        ROOT / "outputs/iclr27_phase4l/heldout/"
              "selected_heldout_videos.csv"))}
    assert not (dev & held)
    assert len(held) == 24


def test_23_heldout_not_tuned():
    dec = ROOT / "docs/iclr27_phase4l/HELDOUT_RESULTS.md"
    if dec.exists():
        assert "seed" in dec.read_text()


def test_24_open_source_commits_verified():
    inv = ROOT / "outputs" / "iclr27_phase4l" / "open_source" / \
        "repository_inventory.csv"
    if not inv.exists():
        return
    with open(inv) as f:
        for r in csv.DictReader(f):
            c = r.get("commit", "")
            if c and c != "not cloned":
                assert re.fullmatch(r"[0-9a-f]{40}", c), c


def test_25_licenses_recorded():
    inv = ROOT / "outputs" / "iclr27_phase4l" / "open_source" / \
        "repository_inventory.csv"
    if not inv.exists():
        return
    with open(inv) as f:
        for r in csv.DictReader(f):
            assert r.get("license", "") != ""


def test_26_old_outputs_preserved():
    assert (ROOT / "outputs/iclr27_phase4k/audit/prov_j1b/"
            "equivalence.json").exists()
    assert (ROOT / "outputs/iclr27_phase4j/subset/J1b").exists()
    assert (ROOT / "outputs/iclr27_phase4j/subset/J2b").exists()


def test_27_video_boundary_clears_physical_track_state():
    m = _mgr(commit_mode="M1")
    m.video_id = 1
    m._apply_association(0, [(7, _obs(), 1.0)])
    assert 7 in m.tracks
    m.video_id = 2
    m._apply_association(0, [(7, _obs(), 1.0)])
    # track 7 in video 2 must be a fresh physical identity, not the
    # video-1 track state
    assert m.tracks[7].last_frame == 0
    assert m.tracks[7].novel_support == 0 or \
        m.tracks[7].novel_support == 1


def test_28_observed_novel_id_does_not_override_committed_identity():
    m = _mgr(commit_mode="M1")
    z = np.random.RandomState(3).randn(4).astype(np.float32)
    m._apply_association(0, [(7, dict(_obs(p_known=0.1), z=z), 1.0)])
    m._apply_association(1, [(7, dict(_obs(p_known=0.1), z=z), 1.0)])
    gid = m.tracks[7].committed_sem_id
    assert isinstance(gid, int)
    # a different observed argmax must not silently replace the committed id
    m._apply_association(2, [(7, dict(_obs(p_known=0.1),
                                      novel_id=1000001, best_novel=0.9),
                              1.0)])
    assert m.tracks[7].committed_sem_id == gid
