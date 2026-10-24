import json
import os

import pytest

from scripts.trackocd_v2.build_sharded_feature_cache import (
    _normalize_shard_range,
    _recover_or_reject_markers,
    _worker_failure_path,
    _worker_manifest_path,
)


def test_normalize_shard_range_uses_exclusive_endpoints():
    assert _normalize_shard_range(10, None, None) == (0, 10, False)
    assert _normalize_shard_range(10, 3, 7) == (3, 7, True)

    with pytest.raises(ValueError, match="supplied together"):
        _normalize_shard_range(10, 3, None)
    with pytest.raises(ValueError, match="0 <= start"):
        _normalize_shard_range(10, 7, 3)
    with pytest.raises(ValueError, match="0 <= start"):
        _normalize_shard_range(10, 0, 11)


def test_bounded_worker_does_not_claim_another_workers_marker(tmp_path):
    shard = tmp_path / "shards" / "shard-000001"
    shard.mkdir(parents=True)
    (shard / ".launched").write_text(
        json.dumps({"pid": os.getpid()}), encoding="utf-8"
    )

    _recover_or_reject_markers(tmp_path, {0})
    assert (shard / ".launched").is_file()

    with pytest.raises(RuntimeError, match="already owned"):
        _recover_or_reject_markers(tmp_path, {1})


def test_worker_artifacts_are_range_scoped(tmp_path):
    assert _worker_manifest_path(tmp_path, 0, 181).name == (
        "cache_manifest.worker-000000-000181.json"
    )
    assert _worker_failure_path(tmp_path, 181, 362).name == (
        "cache_failure.worker-000181-000362.json"
    )
