import json

import pytest

from scripts.trackocd_v2.normalize_frontend_output import normalize_file


def test_grouped_stream_is_sorted_and_category_free(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join([
            json.dumps({
                "sample_id": "P2_9",
                "video_id": 2,
                "track_id": 9,
                "frame_ids": [20, 10],
                "boxes_xyxy": [[2, 2, 8, 8], [1, 1, 7, 7]],
                "image_paths": ["v2/f20.jpg", "v2/f10.jpg"],
                "scores": [0.7, 0.8],
                "category_ids": [12, 12],
            }),
            json.dumps({
                "sample_id": "P1_3",
                "video_id": 1,
                "track_id": 3,
                "frame_ids": [5],
                "boxes_xyxy": [[0, 0, 4, 4]],
                "image_paths": ["v1/f5.jpg"],
                "scores": [0.9],
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    physical = tmp_path / "physical.jsonl"
    native = tmp_path / "native.jsonl"
    audit = tmp_path / "audit.json"
    result = normalize_file(
        frontend="COVTrack-NoSemantic",
        input_path=source,
        input_format="grouped_jsonl",
        physical_output=physical,
        native_output=native,
        audit_output=audit,
        annotation_path=None,
    )
    rows = [json.loads(line) for line in physical.read_text(encoding="utf-8").splitlines()]
    native_rows = [json.loads(line) for line in native.read_text(encoding="utf-8").splitlines()]
    assert result["counts"] == {"tracks": 2, "observations": 3, "videos": 2, "invalid_boxes": 0}
    assert [row["video_id"] for row in rows] == [1, 2]
    assert rows[1]["frame_ids"] == [10, 20]
    assert all("category_id" not in row for row in rows)
    assert native_rows[0]["category_id"] is None
    assert native_rows[1]["category_id"] == 12


def test_tao_array_is_streamed_and_xywh_is_normalized(tmp_path):
    source = tmp_path / "tao.json"
    source.write_text(
        json.dumps([{
            "image_id": 11,
            "video_id": 7,
            "track_id": 4,
            "bbox": [2, 3, 5, 6],
            "score": 0.4,
            "category_id": 99,
        }]),
        encoding="utf-8",
    )
    annotation = tmp_path / "validation.json"
    annotation.write_text(json.dumps({"images": [{
        "id": 11, "video_id": 7, "frame_index": 2, "file_name": "val/v7/f2.jpg"
    }]}), encoding="utf-8")
    physical = tmp_path / "physical.jsonl"
    native = tmp_path / "native.jsonl"
    result = normalize_file(
        frontend="OVTR-native",
        input_path=source,
        input_format="tao_json",
        physical_output=physical,
        native_output=native,
        audit_output=tmp_path / "audit.json",
        annotation_path=annotation,
    )
    row = json.loads(physical.read_text(encoding="utf-8").strip())
    native_row = json.loads(native.read_text(encoding="utf-8").strip())
    assert result["counts"]["observations"] == 1
    assert row["boxes_xyxy"] == [[2.0, 3.0, 7.0, 9.0]]
    assert native_row["bbox_xyxy"] == [2.0, 3.0, 7.0, 9.0]
    assert native_row["category_id"] == 99
    assert "category_id" not in row


def test_invalid_boxes_require_explicit_retention_policy(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({
        "sample_id": "P1_1", "video_id": 1, "track_id": 1,
        "frame_ids": [1], "boxes_xyxy": [[0, 0, 0, 2]],
        "image_paths": ["v1/f1.jpg"], "scores": [0.2],
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or degenerate"):
        normalize_file(
            frontend="SimOWT/Q0", input_path=source, input_format="grouped_jsonl",
            physical_output=tmp_path / "strict_physical.jsonl",
            native_output=tmp_path / "strict_native.jsonl",
            audit_output=tmp_path / "strict_audit.json",
            annotation_path=None,
        )
    result = normalize_file(
        frontend="SimOWT/Q0", input_path=source, input_format="grouped_jsonl",
        physical_output=tmp_path / "retained_physical.jsonl",
        native_output=tmp_path / "retained_native.jsonl",
        audit_output=tmp_path / "retained_audit.json",
        annotation_path=None,
        allow_invalid_boxes=True,
    )
    assert result["status"] == "COMPLETE_WITH_INVALID_BOXES"
    assert result["counts"]["invalid_boxes"] == 1
    assert len((tmp_path / "retained_native.jsonl").read_text(encoding="utf-8").splitlines()) == 1
