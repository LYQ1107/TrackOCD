import json

from scripts.trackocd_v2.recovery_asset_admission import build_report


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_minimal_complete_assets(root):
    data = root / "data"
    output = root / "outputs" / "trackocd_v2"
    (data / "raw/tao/annotations").mkdir(parents=True)
    (data / "raw/tao/annotations/train.json").write_text("not opened", encoding="utf-8")
    (data / "raw/tao/annotations/validation.json").write_text("not opened", encoding="utf-8")
    for name in ("known_ids", "unknown_ids_val", "distractor_ids"):
        _write_json(data / "tao_ow_ocd_v1/splits" / f"{name}.json", [1])
    predicted = data / "tao_ow_ocd_v1/public/pred_track_stream.jsonl"
    predicted.parent.mkdir(parents=True, exist_ok=True)
    predicted.write_text("large stream placeholder", encoding="utf-8")
    for name in (
        "val_gt_track_stream.jsonl",
        "val_gt_track_stream_seed1027.jsonl",
        "val_gt_track_stream_seed1028.jsonl",
        "val_gt_track_stream_seed1029.jsonl",
    ):
        (predicted.parent / name).write_text("order placeholder", encoding="utf-8")

    stage = {
        "status": "BAKEOFF_METRICS_COMPLETE",
        "native_run_complete": True,
        "physical_stream_contract_complete": True,
        "same_v2_physical_metric_protocol": True,
        "same_v2_evaluator_contract": True,
        "requested_metrics_complete": True,
        "test_semantic_accessed": False,
        "normalized_outputs": {"physical_stream": str(predicted)},
    }
    audit = output / "audit"
    _write_json(audit / "frontend_selection.json", {
        "status": "FINAL_PHYSICAL_FRONTEND_SELECTED",
        "selected_frontend": "COVTrack-native",
    })
    _write_json(audit / "representation_decision.json", {
        "status": "FINAL_REPRESENTATION_SELECTED",
        "formal_common_feature_cache_authorized": True,
    })
    _write_json(audit / "autonomous_state.json", {"test_semantic_accessed": False})
    _write_json(audit / "frontend_covtrack_native.json", stage)

    cache = output / "features/formal/covtrack_native"
    shard = cache / "shards/shard-000000"
    shard.mkdir(parents=True)
    for name in ("observations.npy", "prefix_features.npy", "full_features.npy", "index.parquet"):
        (shard / name).write_bytes(b"placeholder")
    _write_json(shard / ".done", {"track_start": 0, "track_count": 1})
    _write_json(cache / "cache_manifest.json", {
        "status": "COMPLETE",
        "split": "val",
        "format": "sharded_numpy_with_parquet_index",
        "cache_root": str(cache),
        "shard_tracks": 1,
        "completed_shards": [{"track_start": 0, "track_count": 1}],
        "encoder": {
            "text_or_category_logits": False,
            "category_id_feature": False,
            "physical_id_feature": False,
            "future_observations_used": False,
        },
        "test_semantic_accessed": False,
    })
    return data, output


def test_missing_assets_is_read_only(tmp_path):
    report = build_report(tmp_path, tmp_path / "data", tmp_path / "outputs")
    assert report["formal_ready"] is False
    assert report["status"] == "BLOCKED_MISSING_OR_INVALID_FORMAL_V2_ASSETS"
    assert report["test_semantic_accessed"] is False
    assert report["test_paths_opened"] == []
    assert not (tmp_path / "outputs").exists()


def test_complete_val_assets_are_admissible_without_opening_large_inputs(tmp_path):
    data, output = _make_minimal_complete_assets(tmp_path)
    report = build_report(tmp_path, data, output)
    assert report["formal_ready"] is True
    assert report["status"] == "READY_FORMAL_V2_PREDICTED_EVAL_INPUTS"
    assert report["test_semantic_accessed"] is False
    assert report["test_paths_opened"] == []
    assert report["data_assets"]["train_annotation"]["content_checked"] is False
    assert report["data_assets"]["val_annotation"]["content_checked"] is False
    assert report["data_assets"]["predicted_stream"]["content_checked"] is False
    assert set(report["data_assets"]["causal_order"]) == {
        "val_gt_track_stream.jsonl",
        "val_gt_track_stream_seed1027.jsonl",
        "val_gt_track_stream_seed1028.jsonl",
        "val_gt_track_stream_seed1029.jsonl",
    }
    assert all(not item["content_checked"] for item in report["data_assets"]["causal_order"].values())
    assert report["formal_cache_manifests"]["covtrack_native"]["integrity_evidence"]["arrays_opened"] is False


def test_test_split_cache_is_never_admitted(tmp_path):
    data, output = _make_minimal_complete_assets(tmp_path)
    manifest = output / "features/formal/covtrack_native/cache_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["split"] = "test"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = build_report(tmp_path, data, output)
    assert report["formal_ready"] is False
    assert "formal_cache_manifest:split_not_val" in report["problems"]
    assert report["test_semantic_accessed"] is False
