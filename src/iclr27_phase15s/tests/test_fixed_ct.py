"""Focused fixed-denominator CT regression and controls."""
from __future__ import annotations

from src.iclr27_phase15s.evaluation.fixed_ct import controls, fixed_ct_metrics, fixed_eligibility


def rows():
    # Category 10 appears on physical/video (1,1), (1,2), and (2,3); only
    # the last occurrence is fixed cross-video eligible. Category 20 has no
    # cross-video repeat. Candidate births differ, but denominator cannot.
    return [
        {"video_id": 1, "frame_id": 0, "proposal_local_id": 0, "track_id": 1, "gt_role": "novel", "gt_category_id": 10, "gt_track_id": 11},
        {"video_id": 1, "frame_id": 1, "proposal_local_id": 0, "track_id": 2, "gt_role": "novel", "gt_category_id": 10, "gt_track_id": 12},
        {"video_id": 2, "frame_id": 0, "proposal_local_id": 0, "track_id": 3, "gt_role": "novel", "gt_category_id": 10, "gt_track_id": 13},
        {"video_id": 2, "frame_id": 1, "proposal_local_id": 0, "track_id": 4, "gt_role": "novel", "gt_category_id": 20, "gt_track_id": 14},
    ]


def test_denominator_is_prediction_independent():
    r = rows(); e = fixed_eligibility(r, {10, 20}); assert e == [2]
    a = [{"sem_action": "new", "sem_sid": 100}, {"sem_action": "new", "sem_sid": 101}, {"sem_action": "existing", "sem_sid": 100}, {"sem_action": "new", "sem_sid": 102}]
    b = [{"sem_action": "new", "sem_sid": 200}, {"sem_action": "new", "sem_sid": 201}, {"sem_action": "new", "sem_sid": 202}, {"sem_action": "existing", "sem_sid": 202}]
    ma, mb = fixed_ct_metrics(r, a, {10, 20}), fixed_ct_metrics(r, b, {10, 20})
    assert ma["eligible"] == mb["eligible"] == 1
    assert ma["correct"] == 1 and mb["correct"] == 0


def test_controls_share_denominator():
    r = rows(); c = controls(r, {10, 20}); assert {v["eligible"] for v in c.values()} == {1}
    assert c["correct_label_oracle"]["recall"] == 1.0
    assert c["all_new"]["recall"] == 0.0


if __name__ == "__main__":
    test_denominator_is_prediction_independent(); test_controls_share_denominator(); print("fixed CT tests passed")
