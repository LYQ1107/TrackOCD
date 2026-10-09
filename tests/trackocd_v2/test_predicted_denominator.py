from scripts.trackocd_v2.run_pred_baselines import _fixed_denominator_rows


def test_fixed_predicted_denominator_keeps_unmatched_gt_as_defer():
    gt_rows = [
        {
            "gt_sample_key": "1_1",
            "gt_physical_track_id": "1",
            "video_id": 1,
            "gt_stream_order": 0,
            "gt_category_id": 11,
            "gt_split": "new",
            "evaluator_track_key": "1_1",
        },
        {
            "gt_sample_key": "2_1",
            "gt_physical_track_id": "1",
            "video_id": 2,
            "gt_stream_order": 1,
            "gt_category_id": 11,
            "gt_split": "new",
            "evaluator_track_key": "2_1",
        },
    ]
    matched = [{
        "sample_key": "pred_2_1",
        "gt_sample_key": "2_1",
        "video_id": 2,
        "physical_track_id": "p",
        "stream_order": 0,
    }]
    rows, missing = _fixed_denominator_rows(matched, gt_rows)
    assert missing == 1
    assert len(rows) == 2
    assert rows[0]["causal_decision_available"] is True
    assert rows[1]["observation_status"] == "UNMATCHED_GT_TARGET"
    assert rows[1]["causal_decision_available"] is False
