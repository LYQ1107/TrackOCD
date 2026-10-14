import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.trackocd_cvpr import evaluate_M_development as original
from src.trackocd_cvpr.M_member_input_wall_recovery import (
    CONFIG, FAILED, TERMINAL, configuration)
from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_wall_recovery_is_new_namespace_and_changes_only_owned_wall_bound():
    args = SimpleNamespace(model='relative_quarantine', seed=1027)
    old = original.configuration(ROOT, args)
    new = configuration(ROOT, old, args)
    assert new['output_directory'].endswith(
        'M_GT_causal_dev_member_input_memo_wall_recovery_v1_relative_quarantine_s1027')
    assert new['wall_seconds_ceiling'] == 6000
    assert new['rss_bytes_ceiling'] == old['rss_bytes_ceiling'] == 3221225472
    for key, value in old.items():
        if key not in ('output_directory', 'output_prefix', 'wall_seconds_ceiling'):
            assert new[key] == value


@pytest.mark.parametrize('model,seed', [
    ('relative_weighted', 1027), ('pooled_weighted', 1027), ('relative_quarantine', 1028)])
def test_recorded_failure_cannot_be_used_for_another_trial(model, seed):
    args = SimpleNamespace(model=model, seed=seed)
    with pytest.raises(ValueError, match='fixed relative_quarantine/s1027'):
        configuration(ROOT, original.configuration(ROOT, args), args)


def test_original_failed_namespace_is_preserved_without_metric_reuse():
    assert sha256_file(ROOT / TERMINAL) == '5fa18163351e49df16b5e04102ed82500c2396cb7618b62d68a5122ffb9c359f'
    assert {p.name for p in (ROOT / FAILED).iterdir()} == {
        'launch.json', 'progress.json', 'owned_startup_observation_v1.json'}
    assert not (ROOT / FAILED / 'result.json').exists()
    manifest = json.loads((ROOT / CONFIG).read_text())
    assert manifest['partial_metric_reuse_or_old_namespace_overwrite'] is False
    assert manifest['scientific_acceptance_or_model_selection'] is False


def test_recovery_script_composes_original_full_probes_and_input_gate():
    source = (ROOT / 'scripts/trackocd_cvpr/evaluate_M_development_member_input_wall_recovery.py').read_text()
    assert 'original.worker(args, cfg, safety)' in source
    assert 'original.supervise(args, cfg, safety)' in source
    assert 'verify_requalified_member_input_shadow(ROOT)' in source
    assert 'actual full ID/reverse' not in source.lower()
    assert 'execution_profile.json' in source
