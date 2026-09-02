from src.iclr27_phase74r.asset_identity import build_identity_records, content_asset_key, record_from_manifest


def _record(split, path, image_id):
    return {"dataset_name": "tao", "dataset_split": split, "video_file_name": f"{split}/Scene/clip", "image_file_name": f"{split}/Scene/clip/000001.jpg", "frame_index": 1, "canonical_image_key": f"tao|{split}|{split}/Scene/clip|frame=1", "resolved_path": path, "path_exists": bool(path), "image_id": image_id, "video_id": 1}


def test_content_key_ignores_protocol_split_but_keeps_duplicates():
    assert content_asset_key(_record("train", None, 1)) == content_asset_key(_record("validation", None, 2))
    result = build_identity_records([_record("validation", None, 2), _record("validation", None, 3)], [_record("train", None, 1)])
    assert result["summary"]["duplicate_q0_content_keys"] == 1
    assert len(result["ambiguities"]) == 1


def test_category_and_track_are_not_identity_fields():
    a = _record("train", None, 1); b = dict(a); b["category_id"] = 999; b["track_id"] = 123
    assert record_from_manifest(a, "event_train").content_asset_key == record_from_manifest(b, "event_train").content_asset_key
