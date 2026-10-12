import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.GT_video_cache import policy_dev_video_ids


def test_video_projection_skips_classes_fit_and_final_values_and_refuses_overlap(tmp_path):
    p = tmp_path / 'scope.json'
    p.write_text(json.dumps({'assignment': {'classes': {'must_not_decode': True}, 'video_ids': {
        'fit_known': {'must_not_decode': True}, 'policy_train_pseudo': [10],
        'development_pseudo': [20, 21], 'independent_final_pseudo': {'must_not_decode': True}}}}))
    assert policy_dev_video_ids(p, policy_count=1, dev_count=2) == {10: 'policy_train_pseudo', 20: 'development_pseudo', 21: 'development_pseudo'}
    with pytest.raises(ValueError): policy_dev_video_ids(p)
    p.write_text(json.dumps({'assignment': {'video_ids': {'policy_train_pseudo': [10], 'development_pseudo': [10, 21]}}}))
    with pytest.raises(ValueError): policy_dev_video_ids(p, policy_count=1, dev_count=2)


def test_fixed_policy_dev_count_and_no_final_or_automatic_fit():
    r = Path(__file__).resolve().parents[2]
    c = json.loads((r / 'configs/trackocd_cvpr/policy_dev_GT_video_cache_v1.json').read_text())
    assert c['policy_dev_cache_only'] and c['pilot_videos'] == 82 and c['reuse_first_video_shards'] == 0
    assert c['expected_tracks'] == 526 and c['expected_observations'] == 10021
    assert c['expected_metadata_frames'] == 2936 and c['maximum_observations_per_track'] is None
    assert not c['full_cache_or_optimizer_automatic_start']
    assert set(c['expected_role_totals']) == {'policy_train_pseudo', 'development_pseudo'}


def test_resource_retry_changes_namespace_only_and_preserves_terminal_old_attempt():
    r = Path(__file__).resolve().parents[2]
    c = json.loads((r / 'configs/trackocd_cvpr/policy_dev_GT_video_cache_v2_resource_retry.json').read_text())
    assert set(c) == {'schema_version', 'base_config', 'output_directory'}
    assert c['base_config'] == 'configs/trackocd_cvpr/policy_dev_GT_video_cache_v1.json'
    old = json.loads((r / 'outputs/trackocd_cvpr/cvpr2/policy_dev_GT_video_cache_v1_supervisor_attempt_1.json').read_text())
    assert not old['successful_owned_execution'] and old['owned_worker_exit_code'] == 1
