import copy
import hashlib
import json
from unittest.mock import patch
import pytest
from scripts.trackocd_cvpr.audit_train_metadata_containment import inspect_sanitized_plan, historical_sources, reject_sensitive_open


def synthetic_plan():
    return {'schema_version': 'trackocd.cvpr0.private-train-image-plan.v1',
            'GT_or_category_fields': False, 'cadence': 'synthetic', 'selection': 'synthetic',
            'videos': [{'video_id': 1, 'images': [{'image_id': 2, 'frame_index': 0,
                        'image_path': 'train/v/x.jpg', 'image_bytes': 1, 'image_sha256': 'a' * 64}]}]}


def test_synthetic_plan_containment_counts_only():
    result = inspect_sanitized_plan(synthetic_plan(), {'expected_videos': 1, 'expected_images': 1})
    assert result['videos'] == result['images'] == 1
    assert result['nested_GT_fields_in_sanitized_plan'] is False
    assert 'image_path' not in json.dumps(result)


@pytest.mark.parametrize('where', ('top', 'video', 'image'))
def test_gt_arrays_in_any_sealed_plan_level_are_rejected(where):
    plan = copy.deepcopy(synthetic_plan())
    obj = plan if where == 'top' else plan['videos'][0] if where == 'video' else plan['videos'][0]['images'][0]
    obj['neg_category_ids'] = [123]
    with pytest.raises(ValueError):
        inspect_sanitized_plan(plan, {'expected_videos': 1, 'expected_images': 1})


def test_source_byte_binding_is_not_loaded_runtime_proof(tmp_path):
    source = json.dumps({'private_plan_sha256': 'b' * 64}).encode()
    proof = {'preregistration_commit': 'c' * 40, 'source_sha256': {'config.json': hashlib.sha256(source).hexdigest()}}
    cfg = {'physical_registration_commit': 'c' * 40, 'physical_config': 'config.json', 'private_sanitized_plan_sha256': 'b' * 64}
    with patch('scripts.trackocd_cvpr.audit_train_metadata_containment.subprocess.check_output', return_value=source):
        rows = historical_sources(tmp_path, proof, cfg)
        assert rows[0]['recorded_source_file_binding_matches_git'] is True
        assert rows[0]['historical_loaded_runtime_module_identity_proven'] is False
        cfg['private_sanitized_plan_sha256'] = 'd' * 64
        with pytest.raises(ValueError, match='does not bind'):
            historical_sources(tmp_path, proof, cfg)


def test_sensitive_direct_opens_refused_but_sanitized_plan_permitted():
    for name in ('/synthetic/annotations/train.json', '/synthetic/frames/train/x.jpg',
                 '/synthetic/train_labels.parquet', '/synthetic/selection_plan.json',
                 '/synthetic/a.npy', '/synthetic/a.npz', '/synthetic/checkpoints/a.pth'):
        with pytest.raises(PermissionError):
            reject_sensitive_open('open', (name, 'r', 0))
    reject_sensitive_open('open', ('/synthetic/private_image_plan.json', 'r', 0))
    reject_sensitive_open('open', (17, 'r', 0))
    reject_sensitive_open('unrelated', ('/synthetic/a.npy',))
