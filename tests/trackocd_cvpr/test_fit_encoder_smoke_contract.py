import json
import pytest
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import fit_video_ids
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_only_fit_video_values_projected_classes_and_other_scope_values_skipped(tmp_path):
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps({'assignment': {'classes': {'Novel': ['do not decode']},
        'video_ids': {'fit_known': list(range(257)), 'independent_final_pseudo': {'protected': 'not scalar'}}},
        'role_class_support': {'names': ['do not decode']}}))
    assert fit_video_ids(path) == set(range(257))


def test_scope_count_refuses_not_silently_select_smaller_pool(tmp_path):
    path = tmp_path / 'fixture.json'
    path.write_text(json.dumps({'assignment': {'video_ids': {'fit_known': [1]}}}))
    with pytest.raises(ValueError): fit_video_ids(path)


def test_real_config_CPU_smoke_not_training_cache_or_heldout_validation():
    cfg = json.loads((ROOT / 'configs/trackocd_cvpr/selected_frozen_encoder_smoke_v1.json').read_text())
    assert cfg['CPU_compute_workers'] == 1 and cfg['maximum_images_decoded'] == 2
    assert cfg['maximum_CPU_forward_calls'] == 3 and cfg['training_GPU_hours'] == 0
    assert cfg['rss_bytes_ceiling'] == 2147483648 and cfg['wall_seconds_ceiling'] == 180
    assert cfg['worker_script'] == 'scripts/trackocd_cvpr/smoke_selected_frozen_encoder.py'
