import numpy as np
import pytest
from src.trackocd_cvpr.memory_controls import CapacityMemory, prototype_subset, renumber_routes, controlled_replay
from src.trackocd_core.train_first_replay import causal_rows


def test_capacity_fifo_is_visual_only_and_tokens_not_recycled():
    memory = CapacityMemory({1: np.asarray([1., 0.])}, 1)
    first = memory.apply(2, 1, None, np.asarray([1., 0.]), [1.])
    second = memory.apply(2, 1, None, np.asarray([0., 1.]), [1.])
    assert first['token'] != second['token']
    assert first['token'] not in memory.anonymous
    assert memory.evictions == 1
    with pytest.raises(ValueError):
        memory.apply(1, 1, first['token'], np.asarray([1., 0.]), [1.])


def test_subset_uses_visual_hash_not_dictionary_insertion_or_class_id():
    p = {1: np.asarray([1., 0.]), 2: np.asarray([0., 1.])}
    q = {200: p[2], 100: p[1]}
    a = next(iter(prototype_subset(p, 1).values()))
    b = next(iter(prototype_subset(q, 1).values()))
    np.testing.assert_array_equal(a, b)


def test_real_old_equal_time_ID_dependency_is_not_hidden():
    rows = [{'key': str(i), 'video_id': 1, 'physical_track_id': i,
             'observation_count': 1, 'frame_ids': [10]} for i in (1, 2)]
    old = [r['key'] for r in causal_rows(rows, [1], 1)]
    changed = [r['key'] for r in causal_rows(renumber_routes(rows), [1], 1)]
    assert old != changed  # Counterexample, not a false end-to-end PASS claim.


def test_removed_prototype_keeps_original_evaluator_known_denominator():
    import torch
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_core.train_first_experiment import evaluate
    torch.manual_seed(1027)
    model = DecisionMLP().eval().requires_grad_(False)
    rows = [{'key': 'a', 'video_id': 1, 'physical_track_id': 1,
             'observation_count': 1, 'frame_ids': [1]}]
    bank = {'a': {'embedding': np.asarray([1., 0.], np.float32), 'uncertainty': 0.,
                  'maturity': 1., 'quality': np.ones(1, np.float32), 'elapsed': 0., 'actual_prefix': 1}}
    sealed, _ = controlled_replay(rows, [1], 1, {1: np.asarray([1., 0.])}, bank, model, 0., 16,
                                  known_ids=[1, 2])
    result = evaluate(sealed, rows, [{'key': 'a', 'category_id': 2, 'simulation_role': 'known'}])
    assert result['standard']['old_denominator'] == 1
    assert result['standard']['old_correct'] == 0
