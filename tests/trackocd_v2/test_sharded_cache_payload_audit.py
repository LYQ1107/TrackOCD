import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

from scripts.trackocd_v2.audit_sharded_cache_payload import audit


SOURCE_SHA = "a" * 64


def _make_cache(tmp_path, *, nan_prefix=False):
    cache = tmp_path / "cache"
    shard = cache / "shards" / "shard-000000"
    shard.mkdir(parents=True)
    observations = np.ones((2, 768), dtype=np.float16)
    prefixes = np.ones((2, 5, 768), dtype=np.float16)
    full = np.ones((2, 768), dtype=np.float16)
    if nan_prefix:
        prefixes[1, 0, 0] = np.nan
    np.save(shard / "observations.npy", observations, allow_pickle=False)
    np.save(shard / "prefix_features.npy", prefixes, allow_pickle=False)
    np.save(shard / "full_features.npy", full, allow_pickle=False)
    rows = []
    for index in range(2):
        rows.append({
            "sample_key": f"k{index}",
            "source_split": "val_predicted",
            "video_id": 10 + index,
            "physical_track_id": str(index),
            "stream_order": index,
            "global_track_index": index,
            "observation_offset": index,
            "observation_count": 1,
            "frame_ids": [index],
            "image_paths": [f"val/video/{index}.jpg"],
            "boxes_xyxy": [[0.0, 0.0, 1.0, 1.0]],
            "quality": [1.0],
            "prefix_observations_used": [1, 1, 1, 1, 1],
        })
    parquet.write_table(pa.Table.from_pylist(rows), str(shard / "index.parquet"), compression="zstd")
    (shard / ".done").write_text(json.dumps({
        "source_sha256": SOURCE_SHA,
        "track_start": 0,
        "track_count": 2,
        "observation_count": 2,
    }), encoding="utf-8")
    manifest = cache / "cache_manifest.json"
    manifest.write_text(json.dumps({
        "status": "WAITING_RESOURCE",
        "format": "sharded_numpy_with_parquet_index",
        "prefixes": [1, 2, 4, 8, 16],
        "cache_root": str(cache),
        "shard_tracks": 2,
        "total_tracks": 2,
        "total_observations": 2,
        "input": {"sha256": SOURCE_SHA},
        "completed_shards": [{"track_start": 0, "track_count": 2, "observation_count": 2}],
    }), encoding="utf-8")
    return manifest


def test_partial_cache_payload_passes_without_formal_admission(tmp_path):
    result = audit(_make_cache(tmp_path), expected_source_sha256=SOURCE_SHA)

    assert result["status"] == "PASS_PARTIAL_COMPLETED_SHARDS"
    assert result["formal_cache_complete"] is False
    assert result["checked_tracks"] == 2
    assert result["checked_observations"] == 2
    assert result["error_count"] == 0
    assert result["test_content_opened"] is False


def test_nonfinite_payload_fails_closed(tmp_path):
    result = audit(_make_cache(tmp_path, nan_prefix=True), expected_source_sha256=SOURCE_SHA)

    assert result["status"] == "FAIL"
    assert result["error_count"] > 0
    assert any("nonfinite" in error for error in result["errors"])
