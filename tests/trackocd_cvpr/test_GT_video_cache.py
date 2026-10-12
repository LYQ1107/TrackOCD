import json
from pathlib import Path
import numpy as np
import pytest
from src.trackocd_cvpr.GT_video_cache import (video_plan, validate_arrays, observed_prefix,
                                            check_every_prefix, write_atomic_shard, read_verified_shard)


def fixture(tmp_path):
    observations = [[i, i * 3, -1, 2, 4, 5] for i in range(1, 19)]
    physical = tmp_path / 'geometry.json'
    physical.write_text(json.dumps({'tracks': [
        {'video_id': 91, 'opaque_track_id': {'must_not_decode': True}, 'observations': {'not_a_list': True}},
        {'video_id': 10, 'opaque_track_id': 3, 'observations': observations},
        {'video_id': 10, 'opaque_track_id': 7, 'observations': [[19, 57, 2, 3, 4, 5]]}]}))
    metadata = {'image_metadata': [dict(id=i, frame_index=i * 3, video_id=10, width=20, height=10,
                                     file_name='train/fixture.jpg') for i in range(1, 21)]}
    images, boxes, arrays = video_plan(physical, 10, metadata)
    arrays['visual'][:] = 0; arrays['visual'][:, 0] = 1
    return images, boxes, arrays


def test_all_observations_singleton_unclipped_geometry_and_empty_frame_clock(tmp_path):
    images, boxes, arrays = fixture(tmp_path)
    assert len(images) == 20 and len(boxes) == 19
    assert arrays['offsets'].tolist() == [0, 18, 19]  # no old p16 cap
    assert arrays['geometry'][0, 0] < 0 and np.all(arrays['quality'] == 1)
    validate_arrays(arrays)
    result = check_every_prefix(arrays)
    assert result['every_visible_prefix_checked'] == 19
    assert result['prefixes_with_actual_future_tail_mutation'] == 17
    view = observed_prefix(arrays, 0, 1)
    assert set(view.__dict__) == {'visual', 'boxes_normalized_xyxy', 'quality', 'elapsed_frames'}
    assert view.elapsed_frames.tolist() == [0] and view.visual.dtype == np.float32
    assert observed_prefix(arrays, 0, 18).visual.shape == (18, 768)
    with pytest.raises(ValueError): observed_prefix(arrays, 0, 19)


def test_atomic_reopen_hash_completion_preserve_and_tamper_rejection(tmp_path):
    _, _, arrays = fixture(tmp_path)
    record = write_atomic_shard(tmp_path, 0, arrays, {'fixture': True})
    assert record['empty_GT_frames'] == 1 and record['observations_uncapped']
    reopened, _ = read_verified_shard(tmp_path, 0)
    for key, value in arrays.items(): np.testing.assert_array_equal(value, reopened[key])
    with pytest.raises(ValueError): write_atomic_shard(tmp_path, 0, arrays, {})
    payload = tmp_path / record['payload']['filename']
    with payload.open('ab') as handle: handle.write(b'tampered')
    with pytest.raises(ValueError): read_verified_shard(tmp_path, 0)


def test_reject_zero_visual_or_misaligned_frame_clock(tmp_path):
    _, _, arrays = fixture(tmp_path)
    arrays['visual'][0] = 0
    with pytest.raises(ValueError): validate_arrays(arrays)
    arrays['visual'][0, 0] = 1
    arrays['clock_frame_index'][0] += 1
    with pytest.raises(ValueError): validate_arrays(arrays)


def test_no_cache_or_training_expansion_and_fixed_precision_resource_gates():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / 'configs/trackocd_cvpr/pilot_GT_video_cache_v1.json').read_text())
    assert cfg['pilot_videos'] == 2 and cfg['expected_observations'] == 418
    assert cfg['maximum_observations_per_track'] is None
    assert cfg['full_cache_or_optimizer_automatic_start'] is False
    assert cfg['batch_size'] == 4 and cfg['GPU_peak_bytes'] == 5 * 1024**3
    assert cfg['precision_max_absolute_error'] == 0.0005
    assert cfg['precision_min_cosine'] == 0.99999


def test_current_GPU_idle_guard_refuses_foreign_work_and_unknown_utilization(monkeypatch):
    from scripts.trackocd_cvpr import pilot_GT_video_cache as pilot
    cfg = {'GPU_uuid': 'GPU-fixture', 'GPU_idle_max_used_MiB': 64, 'GPU_idle_max_utilization_percent': 0}
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda cmd, **kw:
                        b'GPU-fixture, 15, 0\n' if 'uuid,memory.used,utilization.gpu' in cmd[1] else b'')
    assert pilot.require_idle_gpu(cfg)['chosen_GPU_compute_applications_before_launch'] == 0
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda cmd, **kw:
                        b'GPU-fixture, 15, 0\n' if 'uuid,memory.used,utilization.gpu' in cmd[1] else b'GPU-fixture\n')
    with pytest.raises(RuntimeError): pilot.require_idle_gpu(cfg)
    monkeypatch.setattr(pilot.subprocess, 'check_output', lambda cmd, **kw: b'GPU-fixture, 15, [N/A]\n')
    with pytest.raises(ValueError): pilot.require_idle_gpu(cfg)
