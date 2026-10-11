import copy
import json
import numpy as np
import pytest
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import seal_video
from src.trackocd_cvpr.train_physical_resume import validate_plan, explicit_reuse


def plan_fixture():
    video = {'video_id': 47, 'images': [{'image_id': 1, 'frame_index': 0,
        'image_path': 'train/video/0001.jpg', 'image_bytes': 10, 'image_sha256': 'a' * 64}]}
    return {'GT_or_category_fields': False, 'videos': [video]}


def test_plan_actual_gt_field_cadence_and_split_mutations():
    original = plan_fixture()
    cfg = {'videos': 1, 'canonical_images': 1}
    assert validate_plan(original, cfg) == original['videos']
    for field, value in [('category_id', 999), ('GT_purity', True), ('image_path', 'val/video/a.jpg')]:
        bad = copy.deepcopy(original)
        bad['videos'][0]['images'][0][field] = value
        with pytest.raises(ValueError):
            validate_plan(bad, cfg)
    bad = copy.deepcopy(original)
    bad['videos'][0]['images'].append(copy.deepcopy(bad['videos'][0]['images'][0]))
    with pytest.raises(ValueError):
        validate_plan(bad, {'videos': 1, 'canonical_images': 2})


def test_explicit_reuse_no_copy_and_changed_source_rejected(tmp_path):
    video = plan_fixture()['videos'][0]
    source_cfg = {'private_plan_sha256': 'p', 'checkpoint_sha256': 'c', 'native_model_config_sha256': 'n',
                  'frame_root': 'f', 'environment_python': 'e'}
    atomic_json(tmp_path / 'source.json', source_cfg)
    run = tmp_path / 'run'
    (run / 'shards').mkdir(parents=True)
    arrays = {'image_id': np.array([1]), 'frame_index': np.array([0]),
        'track_id': np.array([7]), 'boxes': np.array([[0., 0., 2., 2.]]), 'score': np.array([.5]),
        'frame_offsets': np.array([0, 1]), 'det_boxes': np.array([[0., 0., 2., 2.]]),
        'det_score': np.array([.5]), 'det_offsets': np.array([0, 1])}
    digest = sha256_file(tmp_path / 'source.json')
    record = seal_video(run, video, arrays, digest, 'test', {'frozen_state_unchanged': True})
    atomic_json(run / 'result.json', {'status': 'PASS_BOUNDED_TRAIN_PHYSICAL_SMOKE_NOT_MAIN_QUALIFICATION',
        'changed_future_prefix_equal': True, 'frozen_state_unchanged': True, 'shard': record})
    cfg = {**source_cfg, 'reuse': {'video_id': 47, 'directory': 'run', 'config_path': 'source.json',
        'config_sha256': digest, 'result_sha256': sha256_file(run / 'result.json'), 'npz_sha256': record['npz_sha256']}}
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    result = explicit_reuse(tmp_path, video, cfg)
    assert result['reused_without_copy_or_forward'] is True
    assert before == sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*'))
    altered = copy.deepcopy(cfg)
    altered['native_model_config_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        explicit_reuse(tmp_path, video, altered)
    atomic_json(run / 'result.json', {'status': 'changed'})
    with pytest.raises(ValueError):
        explicit_reuse(tmp_path, video, cfg)
