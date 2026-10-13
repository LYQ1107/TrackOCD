import json
from pathlib import Path
import pytest
from scripts.trackocd_cvpr.shadow_M_fingerprint_cache import CONFIG, verify_profile_configuration

ROOT = Path(__file__).resolve().parents[2]


def test_shadow_retains_entire_all11_protocol_and_guards_no_selected_cases():
    cfg = json.loads((ROOT / CONFIG).read_text())
    verify_profile_configuration(ROOT, cfg)
    for change in ({'wall_seconds_ceiling': 6000}, {'training_seeds': [1027]}, {'order_seed': 1028}):
        with pytest.raises(ValueError): verify_profile_configuration(ROOT, {**cfg, **change})


def test_wrapper_keeps_original_entire_old_new_comparison_and_requires_all11_scopes():
    source = (ROOT / 'scripts/trackocd_cvpr/shadow_M_fingerprint_cache.py').read_text()
    assert 'original_worker(args, cfg, safety)' in source and 'len(records) != 11' in source
    assert 'qualified_fingerprint_cache_scope()' in source
    assert '67108864' in source and 'execution_profile.json' in source
