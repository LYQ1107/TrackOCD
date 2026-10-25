from scripts.trackocd_cvpr.causal_physical_reassociate import reassociate_video


def _row(track_id, frames, boxes):
    return {
        "sample_key": track_id,
        "source_sample_id": track_id,
        "video_id": 1,
        "physical_track_id": track_id,
        "frame_ids": frames,
        "boxes_xyxy": boxes,
        "image_paths": [f"train/example/{frame}.jpg" for frame in frames],
        "quality": [0.9 for _ in frames],
        "stream_order": 0,
        "source_split": "train_predicted",
    }


def test_one_frame_prior_iou_bridge_is_causal():
    rows, summary = reassociate_video(
        [
            _row("a", [0, 1], [[0, 0, 10, 10], [0, 0, 10, 10]]),
            _row("b", [2], [[0, 0, 10, 10]]),
        ],
        iou_threshold=0.5,
        max_gap_frames=1,
    )

    assert len(rows) == 1
    assert rows[0]["frame_ids"] == [0, 1, 2]
    assert summary["matched_births"] == 1
    assert summary["output_observations"] == summary["input_observations"]


def test_gap_horizon_and_same_frame_peer_are_not_reused():
    rows, summary = reassociate_video(
        [
            _row("a", [0], [[0, 0, 10, 10]]),
            _row("b", [2], [[0, 0, 10, 10]]),
            _row("c", [2], [[0, 0, 10, 10]]),
        ],
        iou_threshold=0.5,
        max_gap_frames=1,
    )

    assert len(rows) == 3
    assert summary["matched_births"] == 0
    assert summary["output_observations"] == 3
