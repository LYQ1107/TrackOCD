import json

from scripts.trackocd_cvpr import run_score_gate_gap30_reassoc_transfer as gap30
from scripts.trackocd_cvpr.causal_physical_reassociate import reassociate_video


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


def test_gap30_config_is_frozen_and_separate_from_one_frame_namespace():
    config_path = gap30.ROOT / "configs/trackocd_cvpr/covtrack_score_gate_gap30_reassoc_transfer_v1.json"
    old_path = gap30.ROOT / "configs/trackocd_cvpr/covtrack_score_gate_reassoc_transfer_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old = json.loads(old_path.read_text(encoding="utf-8"))
    assert config["fixed_mechanism"]["reassociation_max_gap_frames"] == 30
    assert old["fixed_mechanism"]["reassociation_max_gap_frames"] == 1
    assert config["source"]["candidate_outputs"] != old["source"]["candidate_outputs"]
    assert config["boundaries"]["test_accessed"] is False
    assert config["boundaries"]["training_started"] is False


def test_gap30_bridge_is_causal_and_preserves_all_observations():
    rows = [
        _row("native-a", [0], [[0.0, 0.0, 10.0, 10.0]]),
        _row("native-b", [30], [[0.0, 0.0, 10.0, 10.0]]),
    ]
    one_frame, one_summary = reassociate_video(rows, iou_threshold=0.5, max_gap_frames=1)
    gap_frame, gap_summary = reassociate_video(rows, iou_threshold=0.5, max_gap_frames=30)
    assert one_summary["matched_births"] == 0
    assert gap_summary["matched_births"] == 1
    assert len(one_frame) == 2
    assert len(gap_frame) == 1
    assert sum(len(row["frame_ids"]) for row in one_frame) == 2
    assert sum(len(row["frame_ids"]) for row in gap_frame) == 2
    assert gap_frame[0]["frame_ids"] == [0, 30]
