from dataclasses import fields
import numpy as np
import pytest
from src.trackocd_cvpr.causal_observation_router import CurrentObservation, CurrentObservationRouter, visible_bytes


def observation(identity, axis=0, quality=0.7):
    visual = np.eye(1, 768, axis, dtype=np.float32)[0]
    return CurrentObservation(identity, visual, np.array([0.1, 0.2, 0.5, 0.8]), quality)


def actual_model_outputs(batch):
    # The callback receives ONLY PrefixView, not routing, GT or total length.
    return tuple((visible_bytes(v), int(v.weighted_mean().argmax())) for v in batch.model_views)


def test_actual_renumber_reorder_produces_same_current_model_inputs_and_outputs():
    a, b = CurrentObservationRouter(), CurrentObservationRouter()
    a1 = a.ingest(0, [observation('1', 2), observation('999', 0)])
    b1 = b.ingest(0, [observation('not-an-integer', 0), observation('000', 2)])
    assert actual_model_outputs(a1) == actual_model_outputs(b1)
    a2 = a.ingest(5, [observation('999', 1), observation('1', 3)])
    b2 = b.ingest(5, [observation('000', 3), observation('not-an-integer', 1)])
    assert actual_model_outputs(a2) == actual_model_outputs(b2)
    assert a2.visible_input_sha256 == b2.visible_input_sha256
    assert a2.routing_groups != b2.routing_groups


def test_real_future_append_mutation_does_not_change_retained_prior_inputs():
    a, b = CurrentObservationRouter(), CurrentObservationRouter()
    left = a.ingest(1, [observation('x', 2)])
    right = b.ingest(1, [observation('x', 2)])
    before = actual_model_outputs(left)
    a_future = a.ingest(2, [observation('x', 2)])
    b_future = b.ingest(2, [observation('x', 7), observation('new-future-track', 4)])
    assert actual_model_outputs(left) == before == actual_model_outputs(right)
    assert actual_model_outputs(a_future) != actual_model_outputs(b_future)


def test_exact_ties_form_atomic_group_not_id_ordered_sequential_births():
    a, b = CurrentObservationRouter(), CurrentObservationRouter()
    left = a.ingest(0, [observation('9'), observation('2')])
    right = b.ingest(0, [observation('z'), observation('a')])
    assert len(left.model_views) == len(right.model_views) == 1
    assert len(left.routing_groups[0]) == 2
    assert actual_model_outputs(left) == actual_model_outputs(right)
    # A subsequent different observation legitimately splits the visible group.
    split = a.ingest(1, [observation('2', 1), observation('9', 2)])
    assert len(split.model_views) == 2


def test_same_current_observation_different_visible_histories_do_not_coalesce():
    router = CurrentObservationRouter()
    router.ingest(0, [observation('a', 1), observation('b', 2)])
    batch = router.ingest(2, [observation('b', 0), observation('a', 0)])
    assert len(batch.model_views) == 2


def test_observation_alias_mutation_cannot_change_history_or_returned_view():
    router = CurrentObservationRouter()
    item = observation('a', 1)
    batch = router.ingest(0, [item])
    before = actual_model_outputs(batch)
    item.visual[:] = 100
    assert actual_model_outputs(batch) == before
    after = router.ingest(1, [observation('a', 1)])
    assert after.model_views[0].visual[0, 1] == 1
    with pytest.raises(ValueError):
        batch.model_views[0].visual[0, 0] = 5


@pytest.mark.parametrize('mutation', ('nonfinite', 'negative_quality', 'shape', 'duplicate'))
def test_invalid_current_input_transaction_refused(mutation):
    router = CurrentObservationRouter()
    item = observation('a')
    values = [item]
    if mutation == 'nonfinite':
        item.visual[0] = np.nan
    elif mutation == 'negative_quality':
        values = [observation('a', quality=-1)]
    elif mutation == 'shape':
        values = [CurrentObservation('a', np.zeros(767), np.zeros(4), 1)]
    else:
        values.append(observation('a'))
    with pytest.raises(ValueError):
        router.ingest(0, values)
    assert len(router.ingest(0, [observation('a')]).model_views) == 1


def test_empty_frames_chronology_and_observed_bounds_no_future_length_filter():
    router = CurrentObservationRouter(max_tracks=1, max_observations=2)
    assert not router.ingest(0, []).model_views
    router.ingest(1, [observation('a')])
    with pytest.raises(ValueError):
        router.ingest(1, [observation('a')])
    with pytest.raises(ValueError):
        router.ingest(2, [observation('b')])
    router.ingest(2, [observation('a')])
    with pytest.raises(ValueError):
        router.ingest(3, [observation('a')])


def test_GT_category_role_event_purity_fields_not_in_current_interface():
    assert {f.name for f in fields(CurrentObservation)} == {'routing_id', 'visual', 'box_normalized_xyxy', 'quality'}
    item = observation('a')
    with pytest.raises(AttributeError):
        object.__setattr__(item, 'category_id', 4)
    with pytest.raises(TypeError):
        CurrentObservation('a', item.visual, item.box_normalized_xyxy, item.quality, simulation_role='novel')
