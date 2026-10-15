import json

from scripts.trackocd_cvpr.audit_covtrack_train_dense_shards import (
    PHYSICAL_FIELDS,
    _audit_physical_stream,
)


def _row(track_id="track-1", **extra):
    row = {
        "sample_key": f"7/{track_id}",
        "source_sample_id": f"7/{track_id}",
        "video_id": 7,
        "physical_track_id": track_id,
        "frame_ids": [10, 11],
        "boxes_xyxy": [[0.0, 1.0, 10.0, 12.0], [1.0, 2.0, 11.0, 13.0]],
        "image_paths": ["frames/7/10.jpg", "frames/7/11.jpg"],
        "quality": [0.9, 0.8],
        "stream_order": 0,
        "source_split": "train_predicted",
    }
    row.update(extra)
    return row


def test_stream_audit_accepts_category_free_rows(tmp_path):
    path = tmp_path / "physical_tracks.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")

    result = _audit_physical_stream(path, {7})

    assert result["status"] == "PASS"
    assert result["tracks"] == 1
    assert result["observations"] == 2
    assert result["invalid_boxes"] == 0
    assert set(json.loads(path.read_text(encoding="utf-8")).keys()) == PHYSICAL_FIELDS


def test_stream_audit_rejects_forbidden_category_field(tmp_path):
    path = tmp_path / "physical_tracks.jsonl"
    path.write_text(json.dumps(_row(category_id=1)) + "\n", encoding="utf-8")

    result = _audit_physical_stream(path, {7})

    assert result["status"] == "FAIL"
    assert any("forbidden physical field" in error for error in result["errors"])


def test_stream_audit_rejects_duplicate_physical_track_row(tmp_path):
    path = tmp_path / "physical_tracks.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")

    result = _audit_physical_stream(path, {7})

    assert result["status"] == "FAIL"
    assert any("duplicate physical track row" in error for error in result["errors"])
