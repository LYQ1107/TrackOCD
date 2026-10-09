from pathlib import Path

import pytest

from src.trackocd_v2.evaluation.persistent import evaluate_persistent
from src.trackocd_v2.evaluation.standard_ocd import evaluate_standard
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed


def _rows():
    return [
        {"sample_key": "1_1", "video_id": 1, "physical_track_id": "1", "gt_category_id": 1},
        {"sample_key": "1_2", "video_id": 1, "physical_track_id": "2", "gt_category_id": 2},
        {"sample_key": "2_3", "video_id": 2, "physical_track_id": "3", "gt_category_id": 1},
        {"sample_key": "2_4", "video_id": 2, "physical_track_id": "4", "gt_category_id": 2},
    ]


def test_standard_uses_one_global_mapping():
    rows = _rows()
    rows[0]["gt_split"] = "new"
    rows[1]["gt_split"] = "old"
    rows[2]["gt_split"] = "new"
    rows[3]["gt_split"] = "old"
    decisions = [{"kind": "NEW", "token": "A:0"}, {"kind": "KNOWN", "token": "K:2"}, {"kind": "EXISTING", "token": "A:0"}, {"kind": "KNOWN", "token": "K:2"}]
    result = evaluate_standard(rows, decisions, known_ids={2}, novel_ids={1})
    assert result["old_acc"] == 1.0
    assert result["new_acc"] == 1.0
    assert result["h_score"] == 1.0


def test_persistent_reuse_and_wrong_merge_are_separate():
    rows = _rows()
    decisions = [
        {"kind": "NEW", "token": "A:0"},
        {"kind": "NEW", "token": "A:1"},
        {"kind": "EXISTING", "token": "A:0"},
        {"kind": "EXISTING", "token": "A:0"},
    ]
    result = evaluate_persistent(rows, decisions, known_ids=set(), novel_ids={1, 2})
    assert result["commit_ct_correct"] == 1
    assert result["commit_ct_denominator"] == 2
    assert result["false_assignment_count"] == 1
    assert result["false_assignment_denominator"] == 2
    assert result["unresolved_count"] == 0


def test_persistent_uses_joined_gt_track_key_for_predicted_rows():
    rows = [
        {"sample_key": "pred_a", "video_id": 1, "physical_track_id": "0", "evaluator_track_key": "gt_1_7", "gt_category_id": 9},
        {"sample_key": "pred_b", "video_id": 2, "physical_track_id": "0", "evaluator_track_key": "gt_2_3", "gt_category_id": 9},
    ]
    decisions = [{"kind": "NEW", "token": "A:0"}, {"kind": "EXISTING", "token": "A:0"}]
    result = evaluate_persistent(rows, decisions, known_ids=set(), novel_ids={9})
    assert result["persistent_category_count"] == 1
    assert result["commit_ct_denominator"] == 1


def test_persistent_eligibility_uses_registered_gt_order_not_match_order():
    rows = [
        {
            "sample_key": "pred_target",
            "video_id": 2,
            "gt_category_id": 9,
            "gt_split": "new",
            "evaluator_track_key": "2_1",
            "gt_stream_order": 1,
        },
        {
            "sample_key": "pred_source",
            "video_id": 1,
            "gt_category_id": 9,
            "gt_split": "new",
            "evaluator_track_key": "1_1",
            "gt_stream_order": 0,
        },
    ]
    decisions = [
        {"kind": "EXISTING", "token": "A:0"},
        {"kind": "NEW", "token": "A:0"},
    ]
    result = evaluate_persistent(rows, decisions, known_ids=(), novel_ids=(9,), distractor_ids=())
    assert result["commit_ct_denominator"] == 1
    assert result["commit_ct_correct"] == 0
    assert result["unresolved_count"] == 0


def test_test_semantic_guard():
    with pytest.raises(RuntimeError, match="TEST_SEMANTIC_LEAKAGE_FORBIDDEN"):
        assert_test_semantic_access_allowed(Path("/tmp/not-a-freeze.json"), "unit test")
