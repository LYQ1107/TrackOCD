import json

from scripts.trackocd_v2.build_frontend_evaluator_join import _create_spool, _match, _spool_physical_stream


def test_frontend_join_matches_only_after_causal_stream_is_spooled(tmp_path):
    physical = tmp_path / "physical.jsonl"
    physical.write_text(
        json.dumps({
            "sample_key": "simowt_1_9",
            "source_sample_id": "1_9",
            "video_id": 1,
            "physical_track_id": "9",
            "frame_ids": [1, 2],
            "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
            "image_paths": ["1/1.jpg", "1/2.jpg"],
            "quality": [1.0, 1.0],
            "stream_order": 0,
            "source_split": "val_predicted",
        }) + "\n",
        encoding="utf-8",
    )
    connection, temporary_path = _create_spool()
    try:
        assert _spool_physical_stream(physical, connection) == (1, 2, 1)
        matches = _match(connection, {
            1: [{
                "sample_key": "gt_1",
                "video_id": 1,
                "physical_track_id": "gt-9",
                "boxes": {1: [0.0, 0.0, 10.0, 10.0], 2: [1.0, 1.0, 11.0, 11.0]},
                "gt_category_id": 900,
                "gt_split": "new",
                "stream_order": 0,
            }],
        })
    finally:
        connection.close()
        temporary_path.unlink(missing_ok=True)
    assert len(matches) == 1
    assert matches[0]["sample_key"] == "simowt_1_9"
    assert matches[0]["evaluator_track_key"] == "gt_1"
    assert matches[0]["temporal_iou"] == 1.0
    assert matches[0]["evaluator_only"] is True
