import numpy as np
import pytest
import torch
from src.trackocd_core.persistent_policy import CandidateMemory
from src.trackocd_cvpr.fixed_history_shadow import fixed_history_replay


class AlwaysKnown(torch.nn.Module):
    def forward(self, cues):
        return torch.tensor([4., 0., 0., 0.])


CONSTANTS = {'wrong_known': 8, 'wrong_existing_pollution': 10, 'known_as_new': 4,
    'wrong_new_fragmentation': 3, 'wait_base': .5, 'wait_per_observation': .05}


def item(z, n):
    return {'embedding': np.array(z, np.float32), 'quality': np.ones(n, np.float32),
        'uncertainty': 0., 'maturity': float(n), 'elapsed': float(n), 'actual_prefix': n}


def run(future, labels=None):
    routes = [{'key': '7:999', 'video_id': 7, 'physical_track_id': 999,
               'frame_ids': [0, 1], 'observation_count': 2}]
    return fixed_history_replay(routes, [7], 1, {1: np.array([1., 0.]), 2: np.array([0., 1.])},
        {'7:999': item([0, 1], 1)}, {'7:999': future}, AlwaysKnown(),
        labels or {'7:999': {'category_id': 1, 'simulation_role': 'known'}}, CONSTANTS, 0.)


def test_actual_future_mutation_changes_shadow_not_original_decision():
    actual, rows, counts = run(item([1, 0], 2))
    changed, other, different = run(item([0, 1], 2))
    assert actual == changed and actual.events[0].known_category_id == 2
    assert rows[0]['actual_nominal_cost'] == 8
    assert rows[0]['future_shadow_nominal_cost_evaluator_only'] == 0
    assert counts['known_wrong_commit_later_zero_nominal_cost'] == 1
    assert different['known_wrong_commit_later_zero_nominal_cost'] == 0
    assert rows[0]['prior_visual_history_sha256'] == other[0]['prior_visual_history_sha256']


def test_real_role_category_mutation_only_changes_external_observer():
    actual, rows, _ = run(item([1, 0], 2))
    changed, different, _ = run(item([1, 0], 2), {'7:999': {'category_id': 17, 'simulation_role': 'pseudo_novel'}})
    assert actual == changed
    assert rows[0]['actual_logits'] == different[0]['actual_logits']
    assert rows[0]['prior_visual_history_sha256'] == different[0]['prior_visual_history_sha256']
    assert rows[0]['future_shadow_nominal_cost_evaluator_only'] != different[0]['future_shadow_nominal_cost_evaluator_only']


def test_shadow_write_is_actually_detected(monkeypatch):
    original = CandidateMemory.candidates
    def illegal(memory, embedding, uncertainty, maturity, quality, elapsed):
        if maturity > 1:
            memory.next_id += 1
        return original(memory, embedding, uncertainty, maturity, quality, elapsed)
    monkeypatch.setattr(CandidateMemory, 'candidates', illegal)
    with pytest.raises(AssertionError, match='mutated actual visual history'):
        run(item([1, 0], 2))


def test_short_singleton_has_no_fabricated_later_evidence():
    _, rows, counts = run(item([1, 0], 1))
    assert rows[0]['later_input_available'] is False
    assert rows[0]['future_shadow_action_evaluator_only'] is None
    assert counts['known_later_input_available'] == 0
