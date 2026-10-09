import json
from pathlib import Path

import pytest

from scripts.trackocd_v2 import run_frontend_metrics as metrics
from scripts.trackocd_v2.run_frontend_metrics import (
    _atomic_native_teta_json,
    _build_tracker_json,
    _observability,
    _record_ovtr_teta_failure,
)


def _gt(sample_key, video_id, stream_order):
    return {
        "sample_key": sample_key,
        "video_id": video_id,
        "physical_track_id": sample_key,
        "boxes": {1: [0.0, 0.0, 10.0, 10.0], 2: [1.0, 1.0, 11.0, 11.0]},
        "gt_category_id": 900,
        "gt_split": "new",
        "stream_order": stream_order,
    }


def test_category_free_conversion_and_persistent_observability(tmp_path):
    native = tmp_path / "native.jsonl"
    native.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"video_id": 1, "image_id": 1, "physical_track_id": "a", "bbox_xyxy": [0, 0, 10, 10], "score": 0.9, "category_id": 1},
                {"video_id": 1, "image_id": 2, "physical_track_id": "a", "bbox_xyxy": [1, 1, 11, 11], "score": 0.8, "category_id": 1},
                {"video_id": 2, "image_id": 1, "physical_track_id": "b", "bbox_xyxy": [0, 0, 10, 10], "score": 0.7, "category_id": 1},
                {"video_id": 2, "image_id": 2, "physical_track_id": "b", "bbox_xyxy": [1, 1, 11, 11], "score": 0.6, "category_id": 1},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    tracker_json = tmp_path / "tao_ow_track.json"
    spool = tmp_path / "observations.sqlite"
    conversion = _build_tracker_json(native, tracker_json, spool)
    assert conversion["rows"] == 4
    assert conversion["physical_tracks"] == 2
    assert conversion["native_category_ids"] == [1]
    converted = json.loads(tracker_json.read_text(encoding="utf-8"))
    assert all(row["category_id"] == 1 for row in converted)
    assert all("_native_category_id" not in row for row in converted)

    gt_by_video = {1: [_gt("g1", 1, 0)], 2: [_gt("g2", 2, 1)]}
    metrics = _observability(spool, gt_by_video)
    assert metrics["novel_track_observability"] == {"numerator": 2, "denominator": 2, "value": 1.0}
    assert metrics["persistent_observability"] == {"numerator": 1, "denominator": 1, "value": 1.0}
    assert metrics["persistent_target_observability"] == {"numerator": 1, "denominator": 1, "value": 1.0}


def test_native_teta_conversion_retains_categories_without_physical_fields(tmp_path):
    native = tmp_path / "native.jsonl"
    native.write_text(
        "\n".join(json.dumps(row) for row in [
            {"video_id": 1, "image_id": 4, "physical_track_id": "local", "bbox_xyxy": [2, 3, 12, 13], "score": 0.8, "category_id": 793},
            {"video_id": 1, "image_id": 5, "physical_track_id": "local", "bbox_xyxy": [3, 4, 13, 14], "score": 0.7, "category_id": 794},
        ]) + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "teta" / "tao_track.json"
    count, categories = _atomic_native_teta_json(native, target)
    assert count == 2
    assert categories == {793, 794}
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert rows[0]["bbox"] == [2.0, 3.0, 10.0, 10.0]
    assert [row["category_id"] for row in rows] == [793, 794]
    assert "physical_track_id" not in rows[0]


def test_test_metrics_guard_precedes_stage_and_annotation_access(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "OUTPUT_TARGET", tmp_path / "project_outputs")
    with pytest.raises(RuntimeError, match="TEST_SEMANTIC_LEAKAGE_FORBIDDEN"):
        metrics.run("simowt", split="test")


def test_ovtr_teta_reference_failure_preserves_physical_audit(tmp_path, monkeypatch):
    native = tmp_path / "native.jsonl"
    native.write_text("{}\n", encoding="utf-8")
    run_root = tmp_path / "run"
    raw = run_root / "teta" / "teta_reference.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"status": "FAILED_TETA_REFERENCE", "error": "KeyError: COMBINED_SEQ"}), encoding="utf-8")
    monkeypatch.setattr(metrics, "_teta_runtime_snapshot", lambda: {"python": "/pinned/teta/python", "versions": {"teta": "test"}})

    result = _record_ovtr_teta_failure(
        exc=RuntimeError("TETA reference failed with return code 1"),
        run_root=run_root,
        native_path=native,
        split="val",
    )

    assert result["status"] == "OVTR_NATIVE_TETA_REFERENCE_EVAL_FAILED"
    assert result["reference_only"] is True
    assert result["physical_metrics_preserved"] is True
    assert result["raw_teta_failure"]["error"] == "KeyError: COMBINED_SEQ"
    assert Path(result["failure_audit"]).is_file()
