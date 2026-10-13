from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr.evaluate_M_development import configuration
from src.trackocd_cvpr.M_all_model_fingerprint_development_profile import (
    CONFIG, PROFILE, all_model_configuration, verify_all_model_shadow, validate_all_model_records)
from tests.trackocd_cvpr.test_M_fingerprint_development_profile import synthetic_records

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('model,seed', [('relative_quarantine', 1027), ('relative_weighted', 1029), ('pooled_weighted', 1028)])
def test_full_frozen_protocol_unchanged_except_new_derived_profile_namespace(model, seed):
    args = SimpleNamespace(model=model, seed=seed)
    old = configuration(ROOT, args); new = all_model_configuration(ROOT, old, args)
    assert {k:v for k,v in old.items() if k not in ('output_directory','output_prefix')} == {
        k:v for k,v in new.items() if k not in ('output_directory','output_prefix')}


def test_pending_actual_fingerprints_refuse_without_opening_any_other_inputs(tmp_path):
    p = tmp_path / CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({'shadow_result_sha256':None}))
    with pytest.raises(ValueError, match='pending'): verify_all_model_shadow(tmp_path)


def test_synthetic_public_validator_requires_entire_all11_including_weighted_M_hits():
    r, t, e, ref = synthetic_records()
    e.update(profile=PROFILE, actual_all11_trace_digests_match_previous_profile=True,
        global_hashlib_numeric_hashes_models_data_thresholds_orders_history_and_guards_unchanged=True)
    validate_all_model_records(r, t, e, ref)
    for case in ('no_M_hits','partial_case','partial_scopes','mismatch','bad_exit','extra_bytes','nan_time','no_prior_match'):
        a,b,c = deepcopy(r),deepcopy(t),deepcopy(e)
        if case == 'no_M_hits': c['ordered_all11_scope_counters'][2]['fingerprint_hits'] = 0
        elif case == 'partial_case': a['cases'].pop('relative_weighted_s1028')
        elif case == 'partial_scopes': c['ordered_all11_scope_counters'].pop()
        elif case == 'mismatch': a['cases']['relative_quarantine_s1027']['private_trace_sha256'] = '0'*64
        elif case == 'bad_exit': b['owned_worker_exit_code'] = -9
        elif case == 'extra_bytes': c['ordered_all11_scope_counters'][8]['fingerprint_bytes'] = 33554433
        elif case == 'nan_time': b['wall_seconds_including_registration'] = float('nan')
        else: c['actual_all11_trace_digests_match_previous_profile'] = False
        with pytest.raises(ValueError): validate_all_model_records(a,b,c,ref)


def test_composition_preserves_original_full_worker_and_owned_supervisor_no_timer_changes():
    source = (ROOT / 'scripts/trackocd_cvpr/evaluate_M_development_all_model_fingerprints.py').read_text()
    assert 'original.worker(args, cfg, safety)' in source and 'original.supervise(args, cfg, safety)' in source
    assert 'original_gate(args, cfg, safety)' in source and 'verify_all_model_shadow(ROOT)' in source
    assert 'qualified_M_fingerprint_cache_scope()' in source and 'execution_profile.json' in source
