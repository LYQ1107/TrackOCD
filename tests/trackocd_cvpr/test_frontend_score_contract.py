import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.audit_frontend_score_contract import audit  # noqa: E402


def test_source_contract_detects_upstream_and_final_score_normalization(tmp_path):
    stream_root = tmp_path / "streams"
    stream_root.mkdir()
    row = {
        "sample_key": "sample-0",
        "source_sample_id": "sample-0",
        "video_id": 1,
        "physical_track_id": "p0",
        "frame_ids": [0, 1],
        "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]],
        "image_paths": ["opaque-0", "opaque-1"],
        "quality": [0.003367, 0.004],
        "stream_order": 0,
        "source_split": "train_predicted",
    }
    (stream_root / "part-0" / "physical_tracks.jsonl").parent.mkdir()
    (stream_root / "part-0" / "physical_tracks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = audit(stream_root, tmp_path / "result.json")
    contract = result["source_contract"]
    assert contract["upstream_text_softmax"] is True
    assert contract["final_bbox_head_softmax_on_already_normalized_score"] is True
    assert contract["final_non_background_score_is_class_probability"] is True
    assert contract["foreground_objectness_score_exported"] is False
    assert contract["tracker_obj_score_gate"] is True
    assert contract["tracker_init_score_gate"] is True
    assert contract["tracker_default_distractor_cut"] == 0.5
    assert contract["score_dimension_after_validation_slice"] == 297
    assert result["stream_quality"]["observations"] == 2
