from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr.evaluate_M_development import configuration
from src.trackocd_cvpr.M_fingerprint_development_profile import (
    CONFIG, PROFILE, fingerprint_configuration, verify_fingerprint_shadow, validate_fingerprint_records)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('model,seed', [('Mean_latest', 1027), ('StateEMA_latest', 1027), ('relative_weighted', 1029)])
def test_new_profile_changes_namespace_only_not_whole_protocol_or_guards(model, seed):
    args = SimpleNamespace(model=model, seed=seed)
    original = configuration(ROOT, args); new = fingerprint_configuration(ROOT, original, args)
    for key in original:
        if key not in ('output_directory', 'output_prefix'): assert new[key] == original[key]
    assert 'fingerprint_cache_v1' in new['output_directory']


def test_pending_actual_profile_fingerprint_refuses_before_any_other_input(tmp_path):
    p = tmp_path / CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({'shadow_result_sha256': None}))
    with pytest.raises(ValueError, match='pending'): verify_fingerprint_shadow(tmp_path)


def synthetic_records():
    # Explicit synthetic profile counters for pure validator tests, NOT real
    # new-shadow evidence or an actual terminal qualification.
    out = ROOT / 'outputs/trackocd_cvpr/cvpr4/M_raw_cache_controller_shadow_v1'
    r = json.loads((out / 'result.json').read_text())
    t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
    reference = deepcopy(r['cases'])
    for row in r['cases'].values(): row['retained_derived_cache_bytes'] = 300
    e = {'profile': PROFILE, 'ordered_all11_scope_counters': [dict(Raw_statistics_bytes=100,
        fingerprint_bytes=200, fingerprint_hits=1, fingerprint_misses=1) for _ in range(11)],
        'global_hashlib_original_sources_data_models_thresholds_orders_and_guards_unchanged': True,
        'full72_development_speedup_purity_or_Goal_accepted': False}
    return r, t, e, reference


def test_pure_record_validator_requires_whole_cohort_profile_trace_and_normal_completion():
    r, t, e, ref = synthetic_records(); validate_fingerprint_records(r, t, e, ref)
    for kind in ('missing_case', 'different_trace', 'bad_exit', 'extra_bytes', 'no_actual_hit', 'missing_scope', 'nan_time', 'negative_RSS'):
        a, b, c = deepcopy(r), deepcopy(t), deepcopy(e)
        if kind == 'missing_case': a['cases'].pop('Mean_latest')
        elif kind == 'different_trace': a['cases']['Mean_latest']['private_trace_sha256'] = '0' * 64
        elif kind == 'bad_exit': b['owned_worker_exit_code'] = -9
        elif kind == 'extra_bytes': c['ordered_all11_scope_counters'][0]['fingerprint_bytes'] = 33554433
        elif kind == 'no_actual_hit': c['ordered_all11_scope_counters'][0]['fingerprint_hits'] = 0
        elif kind == 'missing_scope': c['ordered_all11_scope_counters'].pop()
        elif kind == 'nan_time': a['cases']['Mean_latest']['cache_seconds'] = float('nan')
        else: b['peak_sampled_owned_worker_RSS_bytes'] = -1
        with pytest.raises(ValueError): validate_fingerprint_records(a, b, c, ref)


def test_wrapper_keeps_original_full_worker_supervision_actual_ID_and_future_checks():
    source = (ROOT / 'scripts/trackocd_cvpr/evaluate_M_development_fingerprint_cached.py').read_text()
    assert 'original.worker(args, cfg, safety)' in source and 'original.supervise(args, cfg, safety)' in source
    assert 'original_gate(args, cfg, safety)' in source and 'verify_fingerprint_shadow(ROOT)' in source
    assert 'qualified_fingerprint_cache_scope()' in source and '67108864' in source
