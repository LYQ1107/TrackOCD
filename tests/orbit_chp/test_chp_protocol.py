"""Real-risk tests for CHP: frozen split, causal episodes, no leakage."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_chp.split import load_chp_split


def test_split_frozen_and_complete():
    split = load_chp_split()
    assert len(split["episode_pool"]) == 38
    assert len(split["heldout"]) == 10
    labels = load_train_labels()
    all_classes = set(labels.values())
    assert set(split["episode_pool"]) | set(split["heldout"]) == all_classes
    assert not (set(split["episode_pool"]) & set(split["heldout"]))


def test_heldout_never_in_episode_training():
    """Episode builder must never emit heldout classes."""
    from src.orbit_chp.episodes import build_episode
    from src.orbit_chp.split import load_chp_split
    split = load_chp_split()
    heldout = set(split["heldout"])
    all_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)
    z_cache = {sid: np.mean(f, axis=0).astype(np.float32)
               for sid, f in all_feats.items()}
    rng = random.Random(7)
    np_rng = np.random.RandomState(7)
    from src.orbit_chp.episodes import global_tiers
    tiers = global_tiers(z_cache, by_class, split["episode_pool"])
    for mode in ("random", "hard", "mixed"):
        ep = build_episode(mode, rng, np_rng, z_cache, by_class,
                           split["episode_pool"], tiers=tiers)
        assert not (set(ep["novel_classes"]) & heldout)
        assert not (set(ep["known_classes"]) & heldout)


def test_no_oracle_k_and_no_future_in_episodes():
    from src.orbit_chp.episodes import build_episode
    split = load_chp_split()
    all_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)
    z_cache = {sid: np.mean(f, axis=0).astype(np.float32)
               for sid, f in all_feats.items()}
    rng = random.Random(1)
    np_rng = np.random.RandomState(1)
    ep = build_episode("random", rng, np_rng, z_cache, by_class,
                       split["episode_pool"])
    # no future/memory-oracle fields in queries
    for q in ep["queries"]:
        assert "future" not in q
        assert "oracle" not in q
    # pseudo-novel classes must be excluded from episode-known prototypes
    assert not (set(ep["novel_classes"]) & set(ep["protos"].keys()))


def test_class_split_manifest_is_frozen():
    split = load_chp_split()
    p = ROOT / "configs" / "orbit_chp" / "class_split.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert data["episode_pool"] == split["episode_pool"]


def test_input_hashes_present():
    p = ROOT / "outputs" / "iclr27_phase4h" / "audit" / "input_hashes.json"
    assert p.exists()
    hashes = json.loads(p.read_text())
    assert "runs/orbit_mdc/mdc_m2/model.pth" in hashes
    assert len(hashes["runs/orbit_mdc/mdc_m2/model.pth"]) == 64
