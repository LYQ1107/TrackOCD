import json

import numpy as np

from scripts.trackocd_v2.audit_masa_frontend_migration import build_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_candidate(root):
    source = root / "masa"
    shards = source / "shards"
    shards.mkdir(parents=True)
    npz_name = "video_0004.attempt.npz"
    npz_path = shards / npz_name
    np.savez(
        npz_path,
        image_id=np.asarray([100, 101], dtype=np.int64),
        frame_index=np.asarray([0, 1], dtype=np.int64),
        frame_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        track_id=np.asarray([7, 7], dtype=np.int64),
        boxes=np.asarray([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
        score=np.asarray([0.9, 0.8], dtype=np.float32),
        det_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        det_boxes=np.asarray([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
        det_score=np.asarray([0.9, 0.8], dtype=np.float32),
    )
    import hashlib

    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    receipt = {
        "status": "COMPLETE_VIDEO",
        "video_id": 4,
        "npz_filename": npz_name,
        "npz_bytes": npz_path.stat().st_size,
        "npz_sha256": digest,
        "images": 2,
        "detection_rows": 2,
        "prediction_rows": 2,
    }
    _write_json(shards / "video_0004.complete.json", receipt)
    _write_json(source / "prediction_manifest.json", {
        "schema_version": "trackocd.core.masa_full_val_prediction.v1",
        "status": "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM",
        "test_access": False,
        "gt_or_role_model_input": False,
        "training": False,
        "primary_freeze_permitted": False,
        "images": 2,
        "total_detection_rows": 2,
        "total_prediction_rows": 2,
        "videos": [{
            "video_id": 4,
            "status": "COMPLETE_VIDEO",
            "npz_filename": npz_name,
            "config_sha256": "config",
        }],
    })
    return source


def test_complete_historical_stream_is_integrity_checked_but_not_admitted(tmp_path):
    source = _make_candidate(tmp_path)
    report = build_report(source)
    assert report["status"] == "HISTORICAL_MASA_PHYSICAL_CANDIDATE_NOT_V2_ADMISSIBLE"
    assert report["admissible_for_formal_v2"] is False
    assert report["integrity"]["status"] == "PASS"
    assert report["coverage"]["shards_passed"] == 1
    assert report["source_boundary"]["tao_test_accessed"] is False
    assert report["source_boundary"]["source_stream_modified"] is False


def test_semantic_like_npz_field_is_recorded_as_integrity_failure(tmp_path):
    source = _make_candidate(tmp_path)
    path = source / "shards/video_0004.attempt.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    values["category_id"] = np.asarray([1, 1], dtype=np.int64)
    np.savez(path, **values)
    report = build_report(source)
    assert report["integrity"]["status"] == "FAIL"
    assert any("semantic_like_fields_present" in value for value in report["integrity"]["problems"])
