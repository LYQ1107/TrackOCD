import json
from pathlib import Path
import pytest
from scripts.trackocd_cvpr import profile_M_policy as profile

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all11_fingerprint_proof_and_preserved_M_wall_failure_required():
    cfg = json.loads((ROOT / profile.CONFIG).read_text())
    assert profile.checked_profile_configuration(ROOT, cfg)['actual_entire_ALL11_fingerprint_shadow_profile_checked']


@pytest.mark.parametrize('field,value', [
    ('model', 'relative_weighted'), ('training_seed', 1028), ('order_seed', 1028),
    ('endpoint', 'BestDev'), ('optimizer_steps', 1), ('wall_seconds_ceiling', 6000),
    ('scope', [10, 53, 1268, 374, 1]), ('raw_profile_trace_callers_or_arguments_export', True),
    ('new_labels_pixels_dev_final_Val_Test_payloads', True),
    ('expected_private_full_trace_sha256', '0' * 64),
    ('output_directory', 'outputs/trackocd_cvpr/cvpr4/old_namespace')])
def test_no_selection_training_clock_drop_guard_relaxation_or_trace_substitution(field, value):
    cfg = json.loads((ROOT / profile.CONFIG).read_text()); cfg[field] = value
    with pytest.raises(ValueError): profile.checked_profile_configuration(ROOT, cfg)


def test_profiler_is_new_composed_own_worker_and_only_fixed_endpoint_is_deserialized():
    source = (ROOT / profile.SCRIPT).read_text()
    assert 'return runtime.main()' in source and 'runtime.verified_cohort(ROOT)' in source
    assert 'weights_only=True' in source and 'p != allowed' in source
    assert 'qualified_fingerprint_cache_scope()' in source
    assert 'trace_sha != TRACE_SHA' in source and 'profiler.disable()' in source
    assert 'dump_stats' not in source and 'optimizer.step' not in source
    assert 'fixed_endpoint_sha256' in source and 'model_state_unchanged' in source
