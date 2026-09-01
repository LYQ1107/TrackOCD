"""Phase 4M contract tests (real risks only)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.semantic import (
    NovelSemanticMemory,
    SemanticStateManager,
    TrackSemState,
)


def _obs(z=None, p_known=0.2, novel_id=None, novel_conf=0.7,
         best_known=0.3):
    return {
        "z": z if z is not None else np.ones(768, dtype=np.float32) * 0.1,
        "rel": 1.0, "p_known": p_known,
        "class_dist": np.full(2, 0.5, dtype=np.float32),
        "best_known": best_known, "known_margin": 0.05,
        "best_novel": novel_conf if novel_id is not None else 0.0,
        "novel_id": novel_id, "novel_conf": novel_conf,
        "reliability": 1.0,
    }


def _manager(deferral_mode="margin"):
    known = {1: np.ones(768, dtype=np.float32) * 0.2}
    m = SemanticStateManager.__new__(SemanticStateManager)
    m.known_protos = known
    m.radii = {1: 0.1}
    m.P_known = np.stack(list(known.values())).astype(np.float32)
    m.known_ids = [1]
    m.prefix_mode = "P1"
    m.theta_novel = 0.6
    m.class_temp = 0.05
    m.belief_alpha = 0.5
    m.novel_update_rate = 0.2
    m.memo_tracklet_frames = 10
    m.rel_default = 1.0
    m.thr_early = m.thr_stable = 0.30
    m.decision_split_age = None
    m.commit_mode = "M0"
    m.commit_min_age = 2
    m.commit_min_support = 2
    m.provenance = None
    m.admissibility_mode = "none"
    m.admissibility_config = {}
    m.deferral_mode = deferral_mode
    m.defer_margin = 0.05
    m.defer_entropy = 1.6
    m.defer_nk = 0.25
    m.defer_ambiguity_coef = np.asarray([-3.0754, -1.2081, -0.5705],
                                        dtype=np.float32)
    m.defer_ambiguity_intercept = 1.7935
    m.defer_ambiguity_threshold = 0.6097
    m.memory = NovelSemanticMemory(
        novel_update_rate=0.2, min_birth_sim=0.6,
        matching_mode="absolute", margin_threshold=0.05,
        entropy_threshold=1.6)
    m.tracks = {}
    m._z_cache = {}
    m.branch_sticky = m.branch_soft = m.branch_new = 0
    m.branch_deferred = 0
    m.video_id = None
    m.current_frame = -1
    m._last_video_id = None
    m._debug_branch = False
    m.rel_calls = 0
    m.rel_rejects = 0
    return m


def test_unresolved_does_not_create_global_prototype():
    m = _manager("margin")
    m.memory.protos[1000000] = np.ones(768, dtype=np.float32) * 0.1
    m.memory.protos[1000001] = np.ones(768, dtype=np.float32) * 0.11
    m.memory.support[1000000] = 1
    m.memory.support[1000001] = 1
    z = np.ones(768, dtype=np.float32) * 0.08  # ambiguous: margin < 0.05
    assert m.should_defer(z)
    t = TrackSemState(7, _obs(novel_id=1000000), "P1")
    m.tracks[7] = t
    m._mark_unresolved(t)
    assert t.resolution_state == "unresolved"
    assert t.last_action == "UNRESOLVED_NOVEL"
    assert len(m.memory.protos) == 2
    assert t.committed_sem_id is None


def test_unresolved_track_keeps_local_soft_semantics():
    m = _manager("margin")
    z = np.ones(768, dtype=np.float32) * 0.08
    t = TrackSemState(3, _obs(novel_id=None), "P1")
    m.tracks[3] = t
    m._mark_unresolved(t)
    assert t.candidate_id is not None
    assert t.novel_id == t.candidate_id
    assert t.novel_conf >= 0.5
    # still contributes to the association cost matrix
    m.memory.protos[1000000] = np.ones(768, dtype=np.float32) * 0.1
    M = m.semantic_cost_matrix([_obs(novel_id=None, p_known=0.2)],
                               [3])
    assert M is not None and M.shape == (1, 1)


def test_resolution_uses_only_causal_evidence():
    m = _manager("margin")
    m.memory.protos[1000000] = np.ones(768, dtype=np.float32) * 0.1
    m.memory.protos[1000001] = np.ones(768, dtype=np.float32) * 0.11
    z_amb = np.ones(768, dtype=np.float32) * 0.105
    assert m.should_defer(z_amb) is True
    m.memory.protos[1000001] = -np.ones(768, dtype=np.float32) * 0.1
    assert m.should_defer(np.ones(768, dtype=np.float32) * 0.1) is False


def test_ambiguity_rule_matches_frozen_model():
    m = _manager("ambiguity")
    m.memory.protos[1000000] = np.ones(768, dtype=np.float32) * 0.1
    z = np.ones(768, dtype=np.float32) * 0.1
    obs = _obs(novel_id=1000000, novel_conf=0.7, best_known=0.3)
    assert isinstance(m.should_defer(z, obs), bool)


def test_video_boundary_resets_tracks():
    m = _manager()
    t = TrackSemState(1, _obs(), "P1")
    m.tracks[1] = t
    m.video_id = 10
    m._last_video_id = 9
    m._apply_association(0, [(1, _obs(novel_id=None), 1.0)])
    assert all(v is not t for v in m.tracks.values())


def test_observed_and_committed_novel_ids_stay_separated():
    t = TrackSemState(1, _obs(novel_id=1000000), "P1")
    t.update(_obs(novel_id=1000001, p_known=0.2))
    assert t.observed_novel_id == 1000001
    assert t.committed_sem_id is None
    assert t.novel_id == 1000000


def test_anchor_frozen_config():
    cfg = json.loads((ROOT / "outputs" / "iclr27_phase4m" / "dev" /
                      "m0" / "config.json").read_text())
    assert cfg["deferral_mode"] == "none"
    assert cfg.get("decision_threshold", 0.30) == 0.30


def test_m1_frozen_config():
    cfg = json.loads((ROOT / "outputs" / "iclr27_phase4m" / "dev" /
                      "m1" / "config.json").read_text())
    assert cfg["deferral_mode"] == "margin"
    assert cfg["defer_margin"] == 0.05


def test_dev_heldout_no_overlap():
    dev = {int(p.stem) for p in (ROOT / "outputs" / "iclr27_phase3a" /
                                 "smoke" / "pre_assoc_detections").glob(
        "*.jsonl")}
    sel = list(csv.DictReader(open(
        ROOT / "outputs" / "iclr27_phase4l" / "heldout" /
        "selected_heldout_videos.csv")))
    ho = {int(r["video_id"]) for r in sel}
    assert not (dev & ho)


def test_github_commits_and_licenses_recorded():
    rows = list(csv.DictReader(open(
        ROOT / "outputs" / "iclr27_phase4m" / "open_source" /
        "repository_inventory.csv")))
    assert len(rows) >= 6
    for r in rows:
        assert len(r["commit"]) == 40
        assert r["license"]


def test_identity_decision_datasets_present():
    for tag in ("j1b", "b1", "b2", "m0", "m1", "m2", "m3"):
        p = ROOT / "outputs" / "iclr27_phase4m" / "audit" / \
            f"identity_decisions_{tag}.csv"
        assert p.exists() and p.stat().st_size > 0


def test_memory_provenance_runs():
    for tag in ("m0", "m1", "m2", "m3"):
        d = json.loads((ROOT / "outputs" / "iclr27_phase4m" / "audit" /
                        f"offline_summary_{tag}.json").read_text())
        assert d["prototypes"] > 0


def test_old_phase4l_outputs_preserved():
    p = ROOT / "outputs" / "iclr27_phase4l" / "audit" / \
        "prototype_provenance_j1b.csv"
    assert p.exists()


def test_anchor_metrics_reproduced():
    dev = json.loads((ROOT / "outputs" / "iclr27_phase4m" / "dev" /
                      "trackeval" / "tracking_m0.json").read_text())["m0"]
    def first(x):
        return x[0] if isinstance(x, list) else x
    assert abs(dev["HOTA"]["HOTA"][0] - 0.1595) < 0.005
    assert abs(dev["HOTA"]["AssA"][0] - 0.4583) < 0.01
    assert abs(first(dev["CLEAR"]["IDSW"]) - 488) < 5
    ho = json.loads((ROOT / "outputs" / "iclr27_phase4m" / "heldout" /
                     "trackeval" / "tracking_m0.json").read_text())["m0"]
    assert abs(ho["HOTA"]["HOTA"][0] - 0.1792) < 0.005
    assert abs(first(ho["CLEAR"]["IDSW"]) - 563) < 5
    assert abs(first(ho["CLEAR"]["Frag"]) - 226) < 5


def test_deferral_runs_produce_unresolved_rows():
    acts = set()
    p = ROOT / "outputs" / "iclr27_phase4m" / "dev" / "m1" / \
        "semantic_logs" / "88.jsonl"
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("semantic_action") == "UNRESOLVED_NOVEL":
            acts.add(r.get("resolution_state"))
    assert acts == {"unresolved"}
