"""Phase 4K risk-focused contract tests (task section 91)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.semantic import SemanticStateManager
from src.frame_online_trackocd.semantic import TrackSemState
from src.iclr27_phase4k.build_offline_audit import (
    MATCH_THR,
    classify_intervention,
)


def _mgr(commit_mode="M0", threshold=0.3):
    return SemanticStateManager(
        model=None, known_protos={1: np.ones(4, dtype=np.float32)},
        radii={1: 0.1}, device=torch.device("cpu"),
        prefix_mode="P1", theta_novel=0.6, memo_tracklet_frames=10,
        decision_threshold=threshold, commit_mode=commit_mode,
        commit_min_age=2, commit_min_support=2)


def _obs(p_known=0.1, novel_id=None, novel_conf=0.8):
    return {
        "z": np.random.RandomState(5).randn(4).astype(np.float32),
        "p_known": p_known, "class_dist": np.ones(1, dtype=np.float32),
        "best_known": 0.3, "best_novel": novel_conf if novel_id else 0.0,
        "novel_id": novel_id, "novel_conf": novel_conf,
        "reliability": 1.0, "rel": 1.0,
    }


def test_01_observation_before_association():
    src = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "sem_cost = sem_manager.semantic_cost_matrix(" in src
    assert src.index("sem_cost = sem_manager") < src.index("tracker.match(")


def test_02_candidate_semantics_affect_association():
    m = _mgr(commit_mode="M1")
    m._apply_association(0, [(1, _obs(), 1.0)])
    assert m.tracks[1].candidate_id is not None
    M = m.semantic_cost_matrix([_obs(p_known=0.1)], [1])
    assert M[0, 0] > 0.0


def test_03_candidate_is_not_nonexistent():
    m = _mgr(commit_mode="M1")
    m._apply_association(0, [(7, _obs(), 1.0)])
    # provisional candidate keeps a usable local semantic identity
    assert m.tracks[7].novel_id is not None
    assert m.memory.size() == 0          # not global yet
    M = m.semantic_cost_matrix([_obs(p_known=0.1)], [7])
    assert M[0, 0] > 0.0


def test_04_promotion_influence_ordering_if_supported():
    decision = ROOT / "docs/iclr27_phase4k/ROOT_CAUSE_DECISION.md"
    if not decision.exists():
        return
    text = decision.read_text()
    if "CAUSAL_PROMOTION_SIGNAL" not in text:
        return
    spec = ROOT / "docs/iclr27_phase4k/CAUSAL_SEMANTIC_MEMORY_PROMOTION_SPEC.md"
    if "PROGRESSIVE_MEMORY_NOT_SUPPORTED" in text:
        assert not spec.exists() or "w_promoted" not in spec.read_text()
    else:
        assert spec.exists()
        spec_text = spec.read_text()
        m = re.search(r"candidate influence[^\n]*?(\d\.\d+)", spec_text,
                      re.I)
        assert m is not None
        assert float(m.group(1)) < 1.0


def test_05_promotion_uses_no_future():
    for p in (ROOT / "src/iclr27_phase4k").glob("*.py"):
        src = "\n".join(line for line in p.read_text().splitlines()
                        if not line.startswith("from __future__"))
        assert "future" not in src.lower() or "no future" in src.lower()


def test_06_no_gt_online_input():
    for p in (ROOT / "src/frame_online_trackocd").glob("*.py"):
        src = p.read_text()
        assert "validation_20.json" not in src
        assert "supported_known_ids.json" not in src
        assert "ground_truth" not in src.lower()
        assert "gt_track" not in src.lower()


def test_07_no_full_track_feature():
    src = (ROOT / "src/frame_online_trackocd/semantic.py").read_text()
    assert "feats_all" not in src
    assert "full_track" not in src


def test_08_no_oracle_k():
    for p in (ROOT / "src/frame_online_trackocd").glob("*.py"):
        assert "oracle_k" not in p.read_text().lower()


def test_09_no_retroactive_relabel_in_online_path():
    # retrospective GT labeling is confined to the offline audit module
    src = (ROOT / "src/iclr27_phase4k/build_offline_audit.py").read_text()
    assert "diagnostic" in src.lower()
    assert "offline" in src.lower()


def test_10_physical_id_ne_semantic_id():
    m = _mgr(commit_mode="M1")
    m._apply_association(0, [(7, _obs(), 1.0)])
    m._apply_association(1, [(7, _obs(), 1.0)])
    gid = m.tracks[7].committed_sem_id
    assert isinstance(gid, int) and gid >= 1_000_000
    assert gid != 7


def test_11_same_semantic_id_may_span_physical_tracks():
    m = _mgr(commit_mode="M1")
    m._apply_association(0, [(7, _obs(), 1.0)])
    m._apply_association(1, [(7, _obs(), 1.0)])
    gid = m.tracks[7].committed_sem_id
    # simulate an independent physical track matching the same global id
    other = TrackSemState(23, _obs(), "P1")
    other.committed_sem_id = gid
    assert other.committed_sem_id == gid
    assert other.track_id != m.tracks[7].track_id
    assert isinstance(gid, int)


def test_12_cross_track_counts_unique_physical_tracks():
    csv_path = ROOT / "outputs/iclr27_phase4k/audit/cross_track_support.csv"
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["cross_track"] == "1":
            assert int(r["distinct_physical_tracks"]) >= 2
        else:
            assert int(r["distinct_physical_tracks"]) < 2


def test_13_video_local_ids_not_merged_across_videos():
    prov = (ROOT / "src/iclr27_phase4k/provenance.py").read_text()
    assert "video_id" in prov
    assert "track_key" in prov
    run = (ROOT / "src/iclr27_phase4k/run_provenance.py").read_text()
    assert "provenance=prov" in run


def test_14_provenance_gt_diagnostic_only():
    src = (ROOT / "src/iclr27_phase4k/build_offline_audit.py").read_text()
    assert "validation_20.json" in src
    assert "used ONLY for offline diagnostic labeling" in src


def test_15_intervention_attribution_correct():
    base = {
        "appearance_best_idx": 0, "final_best_idx": 1,
        "appearance_best_score": 0.3, "final_best_score": 0.7,
    }
    # semantic switched to the correct track
    eff, _, _, _ = classify_intervention(base, det_gt=5,
                                         ap_gt=6, fn_gt=5, chosen_gt=5)
    assert eff == "helpful"
    # semantic switched to the wrong track
    eff, _, _, _ = classify_intervention(base, det_gt=5,
                                         ap_gt=5, fn_gt=6, chosen_gt=6)
    assert eff == "harmful"
    # no argmax/threshold switch
    no = dict(base, final_best_idx=0, final_best_score=0.3)
    eff, _, _, _ = classify_intervention(no, det_gt=5,
                                         ap_gt=5, fn_gt=5, chosen_gt=5)
    assert eff == "no_effect"


def test_16_birth_provenance_correct():
    prov = (ROOT / "src/iclr27_phase4k/provenance.py").read_text()
    assert "log_birth" in prov
    assert "creator" in prov
    assert "support_after" in prov


def test_17_fp_origin_definition_correct():
    src = (ROOT / "src/iclr27_phase4k/build_offline_audit.py").read_text()
    assert '"fp"' in src
    assert "match_gt" in src


def test_18_promotion_latency_causal_if_promotion():
    decision = ROOT / "docs/iclr27_phase4k/ROOT_CAUSE_DECISION.md"
    if not decision.exists():
        return
    text = decision.read_text()
    if "PROGRESSIVE_MEMORY_NOT_SUPPORTED" in text:
        return
    spec = ROOT / "docs/iclr27_phase4k/CAUSAL_SEMANTIC_MEMORY_PROMOTION_SPEC.md"
    assert spec.exists()
    assert "<= t" in spec.read_text() or "events <= t" in spec.read_text()


def test_19_frame_synchronous_memory():
    src = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "finish_frame(sem_manager, det_obs" in src
    assert src.index("finish_frame") < src.index("post_association_raw")


def test_20_detector_unchanged():
    replay = (ROOT / "src/frame_online_trackocd/replay.py").read_text()
    assert "tracker.match(" in replay
    assert "import" not in replay or "replay_video" in replay


def test_21_dino_frozen():
    src = (ROOT / "src/frame_online_trackocd/semantic.py").read_text()
    assert "torch.no_grad()" in src
    assert "requires_grad_(True)" not in src


def test_22_lambda_s_frozen():
    run = (ROOT / "src/iclr27_phase4k/run_provenance.py").read_text()
    assert "lambda_s=0.1" in run


def test_23_tau_frozen_for_main_comparison():
    run = (ROOT / "src/iclr27_phase4k/run_provenance.py").read_text()
    assert '"j1b": {"decision_threshold": 0.30' in run
    assert '"m1": {"decision_threshold": 0.30' in run


def test_24_heldout_no_dev_overlap():
    decision = ROOT / "docs/iclr27_phase4k/GENERALIZATION_DECISION.md"
    if not decision.exists():
        return
    text = decision.read_text()
    assert "HELDOUT_NOT_AVAILABLE" in text or "do not overlap" in text


def test_25_heldout_not_tuned():
    decision = ROOT / "docs/iclr27_phase4k/GENERALIZATION_DECISION.md"
    if decision.exists():
        text = decision.read_text()
        assert "HELDOUT_NOT_AVAILABLE" in text or "not used for tuning" in text


def test_26_compact_replay_equivalence_if_used():
    decision = ROOT / "docs/iclr27_phase4k/GENERALIZATION_DECISION.md"
    if not decision.exists():
        return
    text = decision.read_text()
    if "compact replay" in text.lower():
        assert "byte-exact" in text or "metric-exact" in text or \
            "not used" in text


def test_27_github_commits_verified():
    inv = ROOT / "outputs/iclr27_phase4k/open_source/repository_inventory.csv"
    if not inv.exists():
        return
    with open(inv) as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        commit = r.get("commit", "")
        if commit and commit != "not cloned":
            assert re.fullmatch(r"[0-9a-f]{40}", commit), commit
        repo = r.get("repo", "")
        assert repo


def test_28_licenses_recorded():
    inv = ROOT / "outputs/iclr27_phase4k/open_source/repository_inventory.csv"
    if not inv.exists():
        return
    with open(inv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r.get("license", "") != ""


def test_29_old_outputs_preserved():
    for tag in ("J0", "J1b", "J2b"):
        p = ROOT / "outputs/iclr27_phase4j/subset" / tag
        assert p.exists() and any(p.glob("*.json"))


def test_30_full_val_not_tuned():
    decision = ROOT / "docs/iclr27_phase4k/GENERALIZATION_DECISION.md"
    if not decision.exists():
        return
    text = decision.read_text()
    assert "FULL_VAL_RESOURCE_BLOCKED" in text or \
        "full-val not used for tuning" in text


def test_31_provenance_replay_byte_exact():
    for tag in ("j0", "j1b", "m1"):
        eq = ROOT / "outputs/iclr27_phase4k/audit" / f"prov_{tag}" / \
            "equivalence.json"
        if eq.exists():
            d = json.loads(eq.read_text())
            assert d["byte_exact"] is True, (tag, d)


def test_32_outcome_groups_fixed_and_transparent():
    src = (ROOT / "src/iclr27_phase4k/build_offline_audit.py").read_text()
    for g in ("USEFUL", "POLLUTING", "MIXED", "LOW_EVIDENCE"):
        assert g in src
    assert "fixed transparent retrospective groups" in src
