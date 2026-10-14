from pathlib import Path
from types import SimpleNamespace

from scripts.trackocd_cvpr import evaluate_M_development as original
from src.trackocd_cvpr.M_source_compatibility_M_recovery import configuration

ROOT = Path(__file__).resolve().parents[2]


def test_compatibility_replay_is_new_namespace_and_only_wall_bound_changes():
    args = SimpleNamespace(model='relative_quarantine', seed=1027)
    old = original.configuration(ROOT, args)
    new = configuration(ROOT, old, args)
    assert new['output_directory'].endswith(
        'M_GT_causal_dev_member_input_source_compatibility_v1_relative_quarantine_s1027')
    assert new['wall_seconds_ceiling'] == 6000
    assert new['rss_bytes_ceiling'] == old['rss_bytes_ceiling'] == 3221225472
    for key, value in old.items():
        if key not in ('output_directory', 'output_prefix', 'wall_seconds_ceiling'):
            assert new[key] == value


def test_compatibility_replay_script_patches_only_declared_cohort_gate():
    source = (ROOT / 'scripts/trackocd_cvpr/evaluate_M_development_source_compatibility.py').read_text()
    assert 'original.verified_cohort = verified_cohort' in source
    assert 'verify_source_compatibility_shadow(ROOT)' in source
    assert 'original.worker(args, cfg, safety)' in source
    assert 'compatibility_replay_is_not_formal_M_qualification' in source
