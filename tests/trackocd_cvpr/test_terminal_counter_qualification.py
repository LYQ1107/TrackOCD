from copy import deepcopy
import json
from pathlib import Path
import subprocess
import pytest
from scripts.trackocd_cvpr.qualify_policy_training_input import settled_after_counter_error
from src.trackocd_cvpr.C_terminal_counter_qualification import qualified_completed_C

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top1_conditioned_s1027'


def inputs():
    return (json.loads((OUT / 'result.json').read_text()),
            json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text()),
            json.loads((ROOT / 'configs/trackocd_cvpr/C_GT_causal_development_v1.json').read_text()))


def test_counter_missing_exit_boundary_reaps_owned_child_not_still_live_guard_bypass():
    class Finished:
        def wait(self, timeout): assert timeout == .1; return 0
    class StillLive:
        def wait(self, timeout): raise subprocess.TimeoutExpired('owned', timeout)
    assert settled_after_counter_error(Finished())
    assert not settled_after_counter_error(StillLive())


def test_actual_complete_closed_set_failure_counter_receipt_not_silently_rewritten():
    r, t, limits = inputs(); original = deepcopy(t)
    q = qualified_completed_C(r, t, limits)
    assert q['effective_completed_worker_qualification'] and not q['all_supervisor_memory_samples_certified']
    assert not q['original_failed_flag_rewritten_or_rerun_or_scientific_gain']
    assert t == original and not t['successful_owned_execution']
    for order in r['orders'].values():
        end = order['prefix_snapshots']['all_available']
        assert order['physical_actions'] == {'KNOWN': 8753}
        assert end['Known_micro_accuracy'] == pytest.approx(282 / 302)
        assert end['declared_pseudo_absorbed_as_Known_queries'] == 119
        assert end['certified_cross_video_reuse_micro_lower_bound'] == 0


@pytest.mark.parametrize('reason', ['RAM', 'wall_time', 'owned_output_size', None])
def test_real_other_guard_or_incomplete_outcome_not_excused_by_exit0(reason):
    r, t, limits = inputs(); t['guard_stop_reason'] = reason
    with pytest.raises(ValueError): qualified_completed_C(r, t, limits)


def test_nonzero_exit_memory_violation_missing_order_and_actual_invariance_failure_refused():
    for change in ('exit', 'RSS', 'order', 'future'):
        r, t, limits = inputs()
        if change == 'exit': t['owned_worker_exit_code'] = -9
        if change == 'RSS': r['resources']['peak_guarded_RSS_bytes'] = limits['rss_bytes_ceiling'] + 1
        if change == 'order': del r['orders']['1029']
        if change == 'future': r['actual_future_probe']['verified_same_prior_events'] = False
        with pytest.raises(ValueError): qualified_completed_C(r, t, limits)
