import json

from scripts.trackocd_cvpr import run_score_gate_velocity_gap30_transfer as velocity
from scripts.trackocd_cvpr.causal_physical_reassociate import reassociate_video
from scripts.trackocd_cvpr.causal_velocity_reassociate import velocity_reassociate_video


def _row(track_id, frames, boxes):
    return {
        "sample_key": f"sample-{track_id}",
        "source_sample_id": f"source-{track_id}",
        "video_id": 72,
        "physical_track_id": str(track_id),
        "frame_ids": list(frames),
        "boxes_xyxy": [list(box) for box in boxes],
        "image_paths": [f"frame-{frame}.jpg" for frame in frames],
        "quality": [0.5 for _ in frames],
        "stream_order": 0 if track_id == "native-a" else 1,
        "source_split": "train_predicted",
    }


def test_velocity_config_is_frozen_and_separate_from_prior_gap30_namespace():
    config_path = velocity.ROOT / "configs/trackocd_cvpr/covtrack_score_gate_velocity_gap30_reassoc_transfer_v1.json"
    prior_path = velocity.ROOT / "configs/trackocd_cvpr/covtrack_score_gate_gap30_reassoc_transfer_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    assert config["fixed_mechanism"]["motion_model"] == "constant_box_velocity_from_last_two_persistent_observations"
    assert prior["fixed_mechanism"].get("motion_model") is None
    assert config["source"]["candidate_outputs"] != prior["source"]["candidate_outputs"]
    assert config["boundaries"]["test_accessed"] is False
    assert config["boundaries"]["training_started"] is False


def test_velocity_bridge_recovers_a_causal_linear_motion_birth():
    rows = [
        _row("native-a", [0, 10], [[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]]),
        _row("native-b", [20], [[20.0, 0.0, 30.0, 10.0]]),
    ]
    last_box, last_summary = reassociate_video(rows, iou_threshold=0.5, max_gap_frames=30)
    velocity_box, velocity_summary = velocity_reassociate_video(rows, iou_threshold=0.5, max_gap_frames=30)
    assert last_summary["matched_births"] == 0
    assert velocity_summary["matched_births"] == 1
    assert velocity_summary["velocity_matches"] == 1
    assert len(last_box) == 2
    assert len(velocity_box) == 1
    assert velocity_box[0]["frame_ids"] == [0, 10, 20]


def test_velocity_bridge_is_prefix_causal():
    prefix = [
        _row("native-a", [0, 10], [[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]]),
        _row("native-b", [20], [[20.0, 0.0, 30.0, 10.0]]),
    ]
    with_future = prefix + [_row("native-c", [30], [[100.0, 0.0, 110.0, 10.0]])]
    left, _ = velocity_reassociate_video(prefix, iou_threshold=0.5, max_gap_frames=30)
    right, _ = velocity_reassociate_video(with_future, iou_threshold=0.5, max_gap_frames=30)
    assert [row for row in right if max(row["frame_ids"]) <= 20] == left
