from __future__ import annotations

from pathlib import Path

from scripts.trackocd_v2.finalize_formal_cache_and_baselines import done_and_launched
from scripts.trackocd_v2.resume_formal_cache_supervisor import done_count, gate


def test_dynamic_gate_requires_worker_ram_margin_and_selects_idle_gpu():
    snapshot = {
        "mem_total_kib": 100_000,
        "mem_available_kib": 40_000,
        "gpu_rows": [
            {"index": 0, "memory_free_mib": 20_000, "compute_apps": [123]},
            {"index": 1, "memory_free_mib": 9_000, "compute_apps": []},
            {"index": 2, "memory_free_mib": 7_000, "compute_apps": []},
        ],
    }

    selected = gate(snapshot, -1)
    assert selected["ram_floor_kib"] == 25_000
    assert selected["required_available_kib"] == 25_000 + 8 * 1024 * 1024
    assert selected["worker_gate_pass"] is False
    assert selected["gpu_index"] == 1
    assert selected["gpu_gate_pass"] is True
    assert selected["pass"] is False

    fixed_busy = gate(snapshot, 0)
    assert fixed_busy["gpu_index"] is None
    assert fixed_busy["gpu_gate_pass"] is False


def test_shard_marker_counters_are_range_and_completion_scoped(tmp_path: Path):
    shards = tmp_path / "shards"
    for index in (0, 1, 2, 4):
        shard = shards / f"shard-{index:06d}"
        shard.mkdir(parents=True)
    (shards / "shard-000000" / ".done").touch()
    (shards / "shard-000001" / ".done").touch()
    (shards / "shard-000002" / ".launched").touch()
    (shards / "shard-000004" / ".done").touch()

    assert done_count(tmp_path, 0, 4) == 2
    assert done_count(tmp_path, 4, 5) == 1
    assert done_and_launched(tmp_path, 5) == (3, 1)
