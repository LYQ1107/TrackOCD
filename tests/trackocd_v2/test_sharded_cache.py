import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

from src.trackocd_v2.sharded_cache import ShardedFeatureStore


def _write_shard(root, index, sample_key, global_index, observations):
    shard = root / "shards" / f"shard-{index:06d}"
    shard.mkdir(parents=True)
    observation_array = np.arange(observations * 768, dtype=np.float16).reshape(observations, 768)
    prefix_array = np.full((1, 5, 768), float(global_index + 1), dtype=np.float16)
    full_array = np.full((1, 768), float(global_index + 10), dtype=np.float16)
    np.save(shard / "observations.npy", observation_array, allow_pickle=False)
    np.save(shard / "prefix_features.npy", prefix_array, allow_pickle=False)
    np.save(shard / "full_features.npy", full_array, allow_pickle=False)
    row = {
        "sample_key": sample_key,
        "source_split": "val_predicted",
        "video_id": 7 + global_index,
        "physical_track_id": str(global_index),
        "stream_order": global_index,
        "global_track_index": global_index,
        "observation_offset": 0,
        "observation_count": observations,
        "frame_ids": list(range(10, 10 + observations)),
        "image_paths": [f"video/{value}.jpg" for value in range(observations)],
        "boxes_xyxy": [[0.0, 0.0, 10.0, 10.0] for _ in range(observations)],
        "quality": [1.0 for _ in range(observations)],
        "prefix_observations_used": [min(prefix, observations) for prefix in (1, 2, 4, 8, 16)],
    }
    parquet.write_table(pa.Table.from_pylist([row]), str(shard / "index.parquet"), compression="zstd")
    (shard / ".done").write_text(
        json.dumps({"track_start": global_index, "track_count": 1, "observation_count": observations}),
        encoding="utf-8",
    )
    return {"track_start": global_index, "track_count": 1, "observation_count": observations}


def test_sharded_feature_store_round_trip(tmp_path):
    cache = tmp_path / "cache"
    completed = [_write_shard(cache, 0, "k0", 0, 1), _write_shard(cache, 1, "k1", 1, 2)]
    manifest = cache / "cache_manifest.json"
    manifest.write_text(
        json.dumps({
            "status": "COMPLETE",
            "format": "sharded_numpy_with_parquet_index",
            "prefixes": [1, 2, 4, 8, 16],
            "cache_root": str(cache),
            "shard_tracks": 1,
            "total_tracks": 2,
            "completed_shards": completed,
        }),
        encoding="utf-8",
    )

    store = ShardedFeatureStore(manifest)
    assert len(store) == 2
    assert list(store.keys()) == ["k0", "k1"]
    assert "k1" in store
    value = store.get("k1")
    assert value["source_split"] == "val_predicted"
    assert value["frame_embeddings"].shape == (2, 768)
    assert value["prefix_features"]["4"].shape == (768,)
    assert value["prefix_observations_used"]["16"] == 2
    assert np.all(value["full_feature"] == 11.0)
