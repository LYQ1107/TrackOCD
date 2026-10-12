import json
import pytest
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import fit_video_ids, verified_encoder_sources
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


def test_present_sources_exact_and_missing_only_Git_skip_worktree(tmp_path):
    (tmp_path / 'backbone.py').write_bytes(b'fixture')
    def run(args, cwd):
        if args[1] == 'ls-files': return b'H backbone.py\nS setup.py\n'
        return b'fixture'
    hashes, missing, total = verified_encoder_sources(tmp_path, 'a' * 40, run)
    assert set(hashes) == {'backbone.py'} and missing == ['setup.py'] and total == 2
    def missing_unexpected(args, cwd):
        return b'H backbone.py\nH setup.py\n' if args[1] == 'ls-files' else b'fixture'
    with pytest.raises(ValueError, match='skip-worktree'):
        verified_encoder_sources(tmp_path, 'a' * 40, missing_unexpected)
    def changed(args, cwd):
        return b'H backbone.py\n' if args[1] == 'ls-files' else b'changed'
    with pytest.raises(ValueError, match='differ'):
        verified_encoder_sources(tmp_path, 'a' * 40, changed)
