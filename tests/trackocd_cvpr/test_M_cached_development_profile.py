import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr.evaluate_M_development import configuration
from src.trackocd_cvpr.M_cached_development_profile import cached_configuration, verify_actual_shadow, CONFIG

ROOT = Path(__file__).resolve().parents[2]


def test_cached_profile_changes_only_namespace_not_dataset_methods_clock_thresholds_or_guards():
    args = SimpleNamespace(model='Mean_latest', seed=1027)
    original = configuration(ROOT, args); new = cached_configuration(ROOT, original, args)
    assert new['output_directory'].endswith('raw_cache_v1_Mean_latest')
    assert {k: v for k, v in original.items() if k not in ('output_prefix', 'output_directory')} == {k: v for k, v in new.items() if k not in ('output_prefix', 'output_directory')}


def test_no_real_replay_without_actual_shadow_fingerprint_even_if_source_tests_pass(tmp_path):
    p = tmp_path / CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({'shadow_result_sha256': None}))
    with pytest.raises(ValueError, match='pending'): verify_actual_shadow(tmp_path)


def test_cached_wrapper_uses_original_worker_supervision_and_actual_ID_future_checks():
    source = (ROOT / 'scripts/trackocd_cvpr/evaluate_M_development_cached.py').read_text()
    assert 'original.worker(args, cfg, safety)' in source and 'original.supervise(args, cfg, safety)' in source
    assert 'qualified_raw_cache_scope()' in source
    assert 'original_gate(args, cfg, safety)' in source and 'verify_actual_shadow(ROOT)' in source
