from scripts.trackocd_v2.build_predicted_evaluator_join import _build_matches


def test_predicted_join_is_geometry_only_and_evaluator_side():
    gt = {
        7: [{
            "sample_key": "7_1",
            "video_id": 7,
            "physical_track_id": "1",
            "boxes": {10: [0, 0, 10, 10], 20: [0, 0, 10, 10]},
            "gt_category_id": 42,
            "gt_split": "new",
        }]
    }
    predicted = {
        7: [{
            "sample_key": "pred_P7_9",
            "source_sample_id": "P7_9",
            "video_id": 7,
            "physical_track_id": "9",
            "stream_order": 12,
            "boxes": {10: [0, 0, 10, 10], 20: [0, 0, 10, 10]},
        }]
    }
    rows = _build_matches(gt, predicted)
    assert len(rows) == 1
    assert rows[0]["temporal_iou"] == 1.0
    assert rows[0]["gt_category_id"] == 42
    assert rows[0]["evaluator_only"] is True
    assert "gt_category_id" not in predicted[7][0]
