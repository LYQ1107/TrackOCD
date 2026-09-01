"""Small causal-controller regressions for provenance and overflow rules."""
from __future__ import annotations

import numpy as np

from src.iclr27_phase15s.evaluation import causal_controller as cc


def _rows():
    return [
        {"video_id": 1, "frame_id": 0, "proposal_local_id": 0, "track_id": 1,
         "gt_role": "novel", "gt_category_id": 50},
        {"video_id": 1, "frame_id": 0, "proposal_local_id": 1, "track_id": 2,
         "gt_role": "novel", "gt_category_id": 50},
        {"video_id": 2, "frame_id": 0, "proposal_local_id": 0, "track_id": 3,
         "gt_role": "novel", "gt_category_id": 50},
        {"video_id": 3, "frame_id": 0, "proposal_local_id": 0, "track_id": 4,
         "gt_role": "known", "gt_category_id": 7},
    ]


def _bank():
    # The known bank is deliberately orthogonal to the novel state vector.
    x = np.asarray([[1.0, 0.0]], dtype=np.float32)
    return {
        "categories": [7],
        "prototypes": {7: x[0]},
        "exemplars": {7: [x[0]]},
        "prototype_matrix": x,
        "exemplar_matrix": x,
        "exemplar_categories": np.asarray([7], dtype=np.int64),
        "rows": 1,
        "tracks": 1,
        "mode": "synthetic",
    }


def test_known_precedes_self_state_and_cross_video_reuses():
    rows = _rows()
    feats = np.asarray([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    decisions, internal = cc.replay(
        rows, feats, _bank(),
        {"tau_known": 0.9, "tau_cross_physical_reuse": 0.9, "margin_new": 0.0},
        {7},
    )
    assert internal["valid"] and not internal["overflow"]
    assert decisions[0]["sem_action"] == "new"
    # Same-video cross-track evidence is not legal cross-physical evidence.
    assert decisions[1]["sem_action"] == "new"
    # The other video can reuse the first track's state.
    assert decisions[2]["sem_action"] == "existing"
    assert decisions[2]["evidence_source"] == "cross_physical"
    assert decisions[2]["sem_sid"] == decisions[0]["sem_sid"]
    # Known evidence is evaluated before any self/local novel state.
    assert decisions[3]["sem_action"] == "known"
    assert decisions[3]["sem_sid"] == 7


def test_state_overflow_is_invalid_not_forced_match():
    rows = [{"video_id": 1, "frame_id": 0, "proposal_local_id": 0, "track_id": 1,
             "gt_role": "novel", "gt_category_id": 50}]
    old = cc.MAX_STATES
    try:
        cc.MAX_STATES = 0
        decisions, internal = cc.replay(
            rows, np.asarray([[0.0, 1.0]], dtype=np.float32), _bank(),
            {"tau_known": 0.9, "tau_cross_physical_reuse": 0.9, "margin_new": 0.0},
            {7},
        )
    finally:
        cc.MAX_STATES = old
    assert not internal["valid"] and internal["overflow"]
    assert decisions[0]["sem_action"] == "invalid_overflow"


if __name__ == "__main__":
    test_known_precedes_self_state_and_cross_video_reuses()
    test_state_overflow_is_invalid_not_forced_match()
    print("controller tests passed")
