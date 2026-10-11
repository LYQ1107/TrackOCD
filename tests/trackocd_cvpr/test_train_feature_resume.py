import pytest
import numpy as np
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.train_feature_resume import assignments, recovered_feature


def test_only_reused_video_skipped_no_size_score_label_routing():
    videos = [{'video_id': i, 'score': 1 / (i + 1), 'role': 'ignored'} for i in range(7)]
    r = assignments(videos, 0, 2)
    assert r == {'0': [1, 3, 5], '1': [2, 4, 6]}
    assert set(r['0']).isdisjoint(r['1'])
    assert sorted(r['0'] + r['1']) == list(range(1, 7))
    with pytest.raises(ValueError):
        assignments(videos, 0, 5)
    with pytest.raises(ValueError):
        assignments(videos + [videos[1]], 0, 2)


def test_recovered_feature_explicit_failure_binding_no_copy_or_mismatched_protocol(tmp_path):
    original = {'common_config_sha256': 'common', 'maximum_observations': 16, 'batch_size': 8,
                'frame_root': 'frames', 'environment_python': 'python'}
    atomic_json(tmp_path / 'original.json', original)
    (tmp_path / 'raw').mkdir()
    path = tmp_path / 'raw/features.npz'
    visual = np.zeros((1, 768), np.float16)
    visual[0, 0] = 1
    np.savez_compressed(path, visual=visual, geometry=np.ones((1, 4), np.float32), quality=np.ones(1, np.float32),
                        frame_index=np.array([0]), image_id=np.array([1]), track_id=np.array([7]), offsets=np.array([0, 1]))
    recovered = {'status': 'RECOVERED_VALID_ALL_ID_FEATURE_INPUT_AFTER_FAILED_FINAL_JSON_NOT_NORMAL_WORKER_SUCCESS',
        'original_worker_returncode': 1, 'all_original_source_and_current_pixel_metadata_geometry_quality_L2_reader_verified': True,
        'physical_npz_sha256': 'physical', 'tracks': 1, 'observations': 1,
        'private_feature_payload': {'directory': 'raw', 'filename': 'features.npz', 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}}
    atomic_json(tmp_path / 'recovery.json', recovered)
    cfg = {**original, 'reuse': {'video_id': 47, 'original_interface_config': 'original.json',
        'config_sha256': sha256_file(tmp_path / 'original.json'), 'recovery_result': 'recovery.json',
        'recovery_result_sha256': sha256_file(tmp_path / 'recovery.json'), 'feature_sha256': sha256_file(path)}}
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    r = recovered_feature(tmp_path, cfg, {'npz_sha256': 'physical'})
    assert r['copied_or_reencoded'] is False and r['original_worker_failure_overwritten'] is False
    assert before == sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    with pytest.raises(ValueError):
        recovered_feature(tmp_path, {**cfg, 'batch_size': 4}, {'npz_sha256': 'physical'})
    with pytest.raises(ValueError):
        recovered_feature(tmp_path, cfg, {'npz_sha256': 'wrong_physical'})
