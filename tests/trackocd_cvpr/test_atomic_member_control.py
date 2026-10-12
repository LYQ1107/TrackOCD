import numpy as np
import pytest
from src.trackocd_cvpr.causal_observation_router import CurrentObservation
from src.trackocd_cvpr.atomic_member_control import AtomicMemberControl


def obs(identity, axis=0):
    vector = np.zeros(768, np.float32)
    vector[axis] = 1
    return CurrentObservation(identity, vector, np.array([.1, .1, .5, .5]), .8)


def new_policy(features, context):
    values = np.full(len(features) + 2, -2.)
    values[-2] = 2.
    return values


def reuse_policy(features, context):
    if not len(features):
        return new_policy(features, context)
    values = np.full(len(features) + 2, -2.)
    values[0] = 2.
    return values


def state(control):
    bank, next_id = control.state_snapshot()
    return bank.state_evidence(), bank.commit_history(), next_id


def test_actual_control_updates_member_prefix_without_inflating_support_count():
    control = AtomicMemberControl({}, reuse_policy)
    assert control.ingest(0, [obs('x')]).events[0].token == 'S:0'
    assert control.ingest(1, [obs('x', 1)]).events[0].action == 'EXISTING'
    evidence = state(control)[0]['S:0']
    assert evidence.support_groups == 1
    assert np.allclose(evidence.prototype()[:2], np.ones(2) / np.sqrt(2))


def test_exact_alias_group_scores_once_then_broadcasts_and_splits_on_divergence():
    calls = []
    def policy(features, context):
        calls.append(features.tobytes())
        for array in (features, context):
            with pytest.raises(ValueError):
                array.flags.writeable = True
        return reuse_policy(features, context)
    control = AtomicMemberControl({}, policy)
    first = control.ingest(0, [obs('a'), obs('b')])
    assert len(calls) == len(first.events) == 1
    assert [event.token for _, event in first.routed_predictions()] == ['S:0', 'S:0']
    sealed = state(control)[1]
    control.ingest(1, [obs('a', 1), obs('b', 2)])
    assert state(control)[0]['S:0'].support_groups == 2
    assert state(control)[1][:len(sealed)] == sealed


def test_all_same_frame_queries_use_preframe_bank_not_partially_created_tokens():
    lengths = []
    def policy(features, context):
        lengths.append(len(features))
        return reuse_policy(features, context)
    control = AtomicMemberControl({}, policy)
    first = control.ingest(0, [obs('a'), obs('b', 1)])
    assert lengths == [0, 0]
    assert [event.action for event in first.events] == ['NEW', 'NEW']
    assert state(control)[2] == 2


def test_actual_ID_renaming_and_input_reorder_preserve_model_events_and_state_vectors():
    left, right = AtomicMemberControl({}, reuse_policy), AtomicMemberControl({}, reuse_policy)
    for frame, a, b in ((0, [obs('9'), obs('2')], [obs('alpha'), obs('beta')]),
                         (1, [obs('9', 1), obs('2', 2)], [obs('beta', 2), obs('alpha', 1)])):
        assert left.ingest(frame, a).events == right.ingest(frame, b).events
        assert state(left)[0] == state(right)[0] and state(left)[2] == state(right)[2]


def test_changed_future_reexecutes_control_but_cannot_rewrite_sealed_prefix():
    left, right = AtomicMemberControl({}, reuse_policy), AtomicMemberControl({}, reuse_policy)
    prior_l, prior_r = left.ingest(0, [obs('x')]), right.ingest(0, [obs('x')])
    assert prior_l == prior_r
    past = state(left)[1]
    left.ingest(1, [obs('x', 1)])
    right.ingest(1, [obs('x', 3), obs('future', 4)])
    assert prior_l == prior_r
    assert state(left)[1][:1] == state(right)[1][:1] == past
    assert state(left)[0] != state(right)[0]


@pytest.mark.parametrize('failure', ('exception', 'nan', 'shape'))
def test_later_scorer_failure_rolls_back_router_bank_birth_counter_and_retries(failure):
    calls = []
    def policy(features, context):
        calls.append(1)
        if len(calls) == 2:
            if failure == 'exception':
                raise RuntimeError('synthetic callback failure')
            return [np.nan, 1] if failure == 'nan' else [1]
        return new_policy(features, context)
    control = AtomicMemberControl({}, policy)
    before = state(control)
    with pytest.raises((ValueError, RuntimeError)):
        control.ingest(0, [obs('x'), obs('y', 1)])
    assert state(control) == before
    retry = control.ingest(0, [obs('x'), obs('y', 1)])
    assert [event.token for event in retry.events] == ['S:0', 'S:1']


def test_bank_commit_bound_refuses_whole_frame_without_birth_counter_drift():
    control = AtomicMemberControl({}, new_policy, bank_limits={'max_commits': 1})
    before = state(control)
    with pytest.raises(ValueError, match='bound'):
        control.ingest(0, [obs('x'), obs('alias')])
    assert state(control) == before
    assert control.ingest(0, [obs('x')]).events[0].token == 'S:0'


def test_cross_video_reused_route_handle_reuses_category_not_physical_member():
    control = AtomicMemberControl({}, reuse_policy)
    control.ingest(0, [obs('x')])
    control.start_next_video()
    assert control.ingest(0, [obs('x')]).events[0].action == 'EXISTING'
    assert state(control)[0]['S:0'].support_groups == 2


def test_indistinguishable_state_bindings_wait_instead_of_choosing_smallest_token():
    def policy(features, context):
        return reuse_policy(features, context)
    control = AtomicMemberControl({}, policy)
    control.ingest(0, [obs('a'), obs('b', 1)])
    before = state(control)
    event = control.ingest(1, [obs('query', 2)]).events[0]
    assert event.action == 'WAIT' and event.ambiguity_reason == 'INDISTINGUISHABLE_BINDINGS'
    assert state(control) == before


def test_known_numeric_binding_not_scorer_feature_and_can_be_renamed():
    prototype = obs('known').visual
    left, right = AtomicMemberControl({7: prototype}, reuse_policy), AtomicMemberControl({123456: prototype}, reuse_policy)
    a, b = left.ingest(0, [obs('x')]).events[0], right.ingest(0, [obs('x')]).events[0]
    assert (a.category_id, b.category_id) == (7, 123456)
    assert a.candidate_feature_bytes == b.candidate_feature_bytes and a.score_bytes == b.score_bytes
    assert b.candidate_bindings_for_external_loss[0][0].identifier == 123456
    assert state(left)[0] == state(right)[0] == {}


def test_no_GT_or_role_argument_and_detached_snapshot():
    control = AtomicMemberControl({}, new_policy)
    with pytest.raises(TypeError):
        control.ingest(0, [obs('x')], category_id=1)
    control.ingest(0, [obs('x')])
    bank, _ = control.state_snapshot()
    bank._members.clear()
    assert state(control)[0]['S:0'].support_groups == 1


def test_actual_model_callback_integrates_without_optimizer_or_real_input():
    import torch
    from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(57)
        model = CandidateScoreMLP().eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        def policy(features, context):
            with torch.no_grad():
                return model(torch.tensor(features.copy()), torch.tensor(context.copy())).numpy()
        control = AtomicMemberControl({7: obs('prototype').visual}, policy)
        control.ingest(0, [obs('a'), obs('b', 1)])
        result = control.ingest(1, [obs('a', 2), obs('b', 3)])
        assert len(result.events) == 2
        assert all(np.isfinite(np.frombuffer(event.score_bytes, dtype='<f4')).all() for event in result.events)
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_constructor_known_bound_is_not_silent_class_selection():
    with pytest.raises(ValueError, match='Known-binding bound'):
        AtomicMemberControl({7: obs('a').visual, 8: obs('b').visual}, new_policy, max_candidate_groups=1)
