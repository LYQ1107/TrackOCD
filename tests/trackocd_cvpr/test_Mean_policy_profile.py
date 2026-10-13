import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.public_profile_summary import public_profile_summary
from scripts.trackocd_cvpr.profile_Mean_policy import checked_profile_configuration, CONFIG

ROOT = Path(__file__).resolve().parents[2]


def test_profile_exports_only_registered_function_counters_never_callers_private_strings(tmp_path):
    source = 'src/mean.py'; filename = str(tmp_path / source)
    stats = {(filename, 1, 'refresh'): (2, 3, .5, 2., {'private-route-or-GT': 999}),
             ('/private/semantic-name', 4, 'private-label'): (1, 1, 1., 1., {}),
             (filename, 2, '<private dynamic name>'): (1, 1, .1, .1, {})}
    out = public_profile_summary(stats, tmp_path, [source])
    assert len(out['top_registered_functions_by_overlapping_cumulative_time']) == 1
    assert out['all_profiled_calls'] == 5
    assert not out['raw_profiler_callers_arguments_locals_or_private_strings_exported']
    for sentinel in ('private-route-or-GT', '/private/semantic-name', 'private-label', '<private dynamic name>'):
        assert sentinel not in json.dumps(out)
    assert out['cumulative_times_overlap_not_additive_wall_time']


def test_nonfinite_stats_and_unbounded_export_refused(tmp_path):
    with pytest.raises(ValueError): public_profile_summary({('f', 1, 'x'): (1, 1, float('nan'), 1., {})}, tmp_path, [])
    with pytest.raises(ValueError): public_profile_summary({}, tmp_path, [], max_rows=51)


def test_actual_public_shadow_and_failure_required_before_policy_profile():
    cfg = json.loads((ROOT / CONFIG).read_text())
    assert checked_profile_configuration(ROOT, cfg)['actual_all11_policy_controller_shadow_checked']
    cfg['optimizer_steps'] = 1
    with pytest.raises(ValueError): checked_profile_configuration(ROOT, cfg)


def test_profiler_composes_original_persistent_supervisor_and_does_not_dump_raw_profile():
    source = (ROOT / 'scripts/trackocd_cvpr/profile_Mean_policy.py').read_text()
    assert 'return runtime.main()' in source and 'runtime.verified_cohort(ROOT)' in source
    assert 'dump_stats' not in source and 'torch.load' not in source
    assert 'qualified_raw_cache_scope()' in source
    assert 'trace_sha != cfg' in source and 'profiler.disable()' in source
