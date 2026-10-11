"""Executed synthetic runner checks; real retained-cache run stays separate."""
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.trackocd_cvpr.audit_atomic_legacy_control import fixed_policy, probe


@pytest.mark.parametrize('policy', (
    'always_new', 'existing_if_available_else_new', 'always_wait'))
def test_probe_compares_actual_routes_cues_actions_state_and_future(policy):
    visuals = np.zeros((6, 768), dtype=np.float16)
    for index in range(6):
        visuals[index, index] = 1
    reader = SimpleNamespace(
        rows=[{'physical_track_id': 91}, {'physical_track_id': 3}],
        data={'visual': visuals, 'geometry': np.tile([0.1, 0.2, 0.5, 0.8], (6, 1)),
              'quality': np.full(6, 0.7)})
    frames = {0: [(0, 0), (1, 3)], 4: [(0, 1), (1, 4)], 9: [(0, 2), (1, 5)]}
    cfg = {'max_tracks': 10, 'max_observations': 20, 'max_anonymous': 20}
    baseline = probe(reader, frames, cfg, policy, 'original', lambda: None)
    assert baseline == probe(reader, frames, cfg, policy, 'renumber_and_reverse', lambda: None)
    for mode in ('future_negation', 'future_append'):
        changed = probe(reader, frames, cfg, policy, mode, lambda: None)
        assert baseline['prior_output_state_sha256'] == changed['prior_output_state_sha256']
        assert baseline['full_output_state_sha256'] != changed['full_output_state_sha256']
        if mode == 'future_append':
            assert baseline['full_output_state_sha256'] == changed['retained_output_state_sha256_before_append']
    assert baseline['queries'] == baseline['atomic_groups'] == 6
    assert baseline['actions'] == ({'NEW': 6} if policy == 'always_new' else
                                   {'EXISTING': 5, 'NEW': 1} if policy == 'existing_if_available_else_new'
                                   else {'WAIT': 6})


def test_unregistered_policy_not_accepted():
    with pytest.raises(ValueError):
        fixed_policy('a_tuned_unknown_policy')
