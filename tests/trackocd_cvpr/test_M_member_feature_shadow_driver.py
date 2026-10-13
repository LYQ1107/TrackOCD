import json
from pathlib import Path
import pytest
import scripts.trackocd_cvpr.shadow_M_member_feature_cache as driver

ROOT = Path(__file__).resolve().parents[2]


def test_registered_whole_policy_input_cache_source_requires_actual_old_all11_and_terminal():
    cfg = json.loads((ROOT / driver.CONFIG).read_text())
    proof = driver.configuration_gate(ROOT, cfg)
    assert proof['actual_entire_ALL11_weighted_M_support_fingerprint_shadow_checked']
    assert cfg['output_directory'].endswith('M_member_feature_cache_shadow_v1')
    assert not cfg['learned_output_memo_controller_refresh_skip_or_full_development_use']
    for p in driver.SOURCES:
        assert (ROOT / p).is_file(), p
    assert len(driver.SOURCES) == len(set(driver.SOURCES))


@pytest.mark.parametrize('change', [
    {'wall_seconds_ceiling': 6000}, {'training_seeds': [1027]}, {'order_seed': 1028},
    {'output_directory': 'outputs/old'}, {'learned_output_memo_controller_refresh_skip_or_full_development_use': True},
    {'qualified_all11_result_sha256': '0' * 64}, {'preserved_full_M_RAM_retry_terminal_sha256': '0' * 64},
])
def test_no_subset_guard_relaxation_or_learned_memo_can_be_qualified(change):
    cfg = json.loads((ROOT / driver.CONFIG).read_text())
    with pytest.raises(ValueError): driver.configuration_gate(ROOT, {**cfg, **change})


def test_registration_and_private_worker_are_unchanged_full_runtime_composition():
    source = (ROOT / driver.SCRIPT).read_text()
    assert 'original_gate(args, cfg, safety)' in source
    assert 'original_worker(args, cfg, safety)' in source
    assert 'len(records) != 11' in source
    assert 'model_outputs_cached' in source
    assert 'out /' in source and 'execution_profile.json' in source
