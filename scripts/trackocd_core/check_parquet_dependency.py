#!/usr/bin/env python3
"""Synthetic CPU I/O smoke; does not read real shards or prove v2 recovery."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file


def run_smoke(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    offset = 0
    for index, (video, count) in enumerate(((11, 2), (12, 3))):
        rows.append({
            'sample_key': f'{video}_7', 'video_id': video,
            'physical_track_id': '7', 'global_track_index': index,
            'observation_offset': offset, 'observation_count': count,
            'frame_ids': list(range(1, count + 1)),
            'image_paths': [f'synthetic/{video}/{frame}.jpg' for frame in range(count)],
            'boxes_xyxy': [[0.0, 0.0, 8.0, 8.0] for _ in range(count)],
            'quality': [1.0] * count,
            'prefix_observations_used': [min(prefix, count) for prefix in (1, 2, 4, 8, 16)],
            'source_split': 'synthetic_no_dataset',
        })
        offset += count
    index_path = directory / 'index.parquet'
    pq.write_table(pa.Table.from_pylist(rows), index_path)
    loaded = pq.read_table(index_path).to_pylist()
    if loaded != rows:
        raise AssertionError('Parquet row values changed')
    observations = np.arange(offset * 768, dtype=np.float32).reshape(offset, 768).astype(np.float16)
    array_path = directory / 'observations.npy'
    np.save(array_path, observations, allow_pickle=False)
    mapped = np.load(array_path, mmap_mode='r', allow_pickle=False)
    for row in loaded:
        start, count = row['observation_offset'], row['observation_count']
        if mapped[start:start + count].shape != (count, 768):
            raise AssertionError('Parquet offset/length does not index the NPY array')
    if mapped.dtype != np.float16 or not np.array_equal(mapped, observations):
        raise AssertionError('FP16 NPY roundtrip failed')
    return {
        'status': 'PASS_SYNTHETIC_CPU_IO_ONLY',
        'tracks': len(rows), 'observations': offset, 'dimension': 768,
        'parquet_nested_row_roundtrip_pass': True,
        'float16_npy_memory_map_roundtrip_pass': True,
        'same_local_id_different_video_retained': len({(r['video_id'], r['physical_track_id']) for r in loaded}) == 2,
        'temporary_payload_bytes': index_path.stat().st_size + array_path.stat().st_size,
        'real_nas_shards_validated': False,
        'legacy_sharded_reader_imported': False,
        'feature_contract_compatibility_proved': False,
        'test_annotation_opened': False, 'training_started': False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/trackocd_core/audit/parquet_dependency_smoke.json')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='trackocd-core-parquet-smoke-') as temporary:
        result = run_smoke(Path(temporary))
    versions = {name: metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'pandas', 'pyarrow')}
    expected = {'torch': '2.6.0+cu118', 'torchvision': '0.21.0+cu118', 'numpy': '2.2.6', 'pandas': '2.3.3', 'pyarrow': '25.0.1'}
    if versions != expected:
        raise RuntimeError(f'existing environment differs from audited versions: {versions}')
    installed = metadata.distribution('pyarrow')
    package_bytes = sum(path.stat().st_size for item in installed.files or []
                        if (path := Path(installed.locate_file(item))).is_file())
    result.update(schema_version='trackocd.core.parquet_dependency_smoke.v1',
                  verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  versions=versions, original_stack_versions_preserved=True,
                  installed_pyarrow_distribution_bytes=package_bytes,
                  install_scope='one pinned binary wheel with uv --no-deps --no-cache; task-scoped proxy only',
                  official_installation_reference='https://arrow.apache.org/docs/python/install.html',
                  official_release_reference='https://pypi.org/project/pyarrow/25.0.1/',
                  source_dependency_evidence_command='exec-083c82f8-1c5d-466b-84bb-24f9f151383c',
                  smoke_script_sha256=sha256_file(Path(__file__)), temporary_payload_retained=False)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
