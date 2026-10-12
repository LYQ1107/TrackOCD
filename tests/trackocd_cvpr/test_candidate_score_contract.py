import numpy as np
import pytest
import torch
from src.trackocd_core.features import prefix_view
from src.trackocd_cvpr.prefix_member_bank import StateEvidence
from src.trackocd_cvpr.candidate_score_contract import build_candidates, resolve_scores, CandidateScoreMLP


def vector(axis=0):
    result = np.zeros(768, np.float32)
    result[axis] = 1
    return result


def view():
    return prefix_view(vector()[None], np.array([[.1, .1, .5, .5]]), np.array([.8]), [0], 1)


def state(token, count=1, quality=.8, status='DEFINED'):
    total = vector().astype('<f8') * count if status == 'DEFINED' else np.zeros(768, '<f8')
    return StateEvidence(token, count, count, total.tobytes(), quality, 0., status)


def test_known_ID_renumber_and_insertion_permutation_preserve_actual_numeric_inputs():
    a = build_candidates(view(), {7: vector(), 999: vector(1)}, {})
    b = build_candidates(view(), {333: vector(1), 123456: vector()}, {})
    assert a.feature_bytes == b.feature_bytes and a.context_bytes == b.context_bytes
    assert resolve_scores(a, [3, 1, 0, -1])[:2] == ('KNOWN', 7)
    assert resolve_scores(b, [3, 1, 0, -1])[:2] == ('KNOWN', 123456)


def test_callback_arrays_have_only_numeric_features_and_cannot_be_made_writable():
    snapshot = build_candidates(view(), {123456: vector()}, {})
    features, context = snapshot.numeric_inputs()
    assert features.shape == (1, 11) and context.shape == (6,)
    assert 123456 not in features and 123456 not in context
    for array in (features, context):
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_identical_known_candidate_features_do_not_pick_smallest_ID():
    snapshot = build_candidates(view(), {991: vector(), 2: vector()}, {}, K=1)
    assert len(snapshot.bindings) == 1 and len(snapshot.bindings[0]) == 2
    assert resolve_scores(snapshot, [3, 0, -1]) == ('WAIT', None, 'INDISTINGUISHABLE_BINDINGS')


def test_known_and_anonymous_are_separate_families_with_no_token_feature():
    a = build_candidates(view(), {7: vector()}, {'S:1': state('S:1')}, K=1)
    b = build_candidates(view(), {123: vector()}, {'S:100000': state('S:100000')}, K=1)
    assert a.feature_bytes == b.feature_bytes and a.context_bytes == b.context_bytes
    assert resolve_scores(a, [0, 3, 1, -1])[:2] == ('EXISTING', 'S:1')
    assert resolve_scores(b, [0, 3, 1, -1])[:2] == ('EXISTING', 'S:100000')


def test_topK_boundary_ties_expand_feature_distinct_pool_or_refuse_bound():
    states = {'S:0': state('S:0', 1), 'S:1': state('S:1', 2)}
    snapshot = build_candidates(view(), {}, states, K=1)
    assert len(snapshot.bindings) == 2
    with pytest.raises(ValueError, match='Tie-expanded'):
        build_candidates(view(), {}, states, K=1, max_groups=1)


def test_zero_resultant_not_repaired_or_exposed_as_usable_candidate():
    states = {'S:0': state('S:0', status='ZERO_RESULTANT')}
    snapshot = build_candidates(view(), {}, states)
    assert len(snapshot.bindings) == 0
    assert resolve_scores(snapshot, [1, 0]) == ('NEW', None, None)
    assert states['S:0'].status == 'ZERO_RESULTANT'


def test_equal_action_maxima_use_explicit_ambiguity_wait_not_label_order():
    snapshot = build_candidates(view(), {3: vector()}, {})
    assert resolve_scores(snapshot, [1, 1, 0]) == ('WAIT', None, 'AMBIGUOUS_MAXIMA')


@pytest.mark.parametrize('scores', ([1], [1, np.nan, 0], [1, np.inf, 0]))
def test_score_shape_and_nonfinite_values_fail_closed(scores):
    with pytest.raises(ValueError):
        resolve_scores(build_candidates(view(), {7: vector()}, {}), scores)


def test_shared_708_parameter_scorer_actual_candidate_permutation_equivariance():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(52)
        model = CandidateScoreMLP()
        assert sum(p.numel() for p in model.parameters()) == 708
        with torch.no_grad():
            model.log_temperature.fill_(.3)
        x = torch.randn(5, 11)
        q = torch.randn(6)
        permutation = torch.tensor([4, 0, 2, 1, 3])
        left, right = model(x, q), model(x[permutation], q)
        assert torch.equal(left[:5][permutation], right[:5])
        assert torch.equal(left[5:], right[5:])
        assert model(torch.zeros(0, 11), q).shape == (2,)


def test_synthetic_backward_checks_gradient_without_fitting_or_changing_weights():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        model = CandidateScoreMLP()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        logits = model(torch.randn(3, 11), torch.randn(6))
        torch.nn.functional.cross_entropy(logits[None], torch.tensor([0])).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        with pytest.raises(TypeError):
            model(torch.zeros(0, 11), torch.zeros(6), category_id=7)


def test_finite_overflowing_prototype_not_silently_normalized_to_zero():
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError, match='nonzero common768D'):
            build_candidates(view(), {7: np.full(768, 1e30, np.float32)}, {})
