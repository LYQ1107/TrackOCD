import pytest
import torch
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
from src.trackocd_cvpr.differentiable_candidate_features import candidate_tensors, FEATURE_COUNT
from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
from tests.trackocd_cvpr.test_raw_preserving_evidence import tensors


def test_actual_joint_backward_reaches_evidence_and_scorer_without_optimizer():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(66)
        evidence, scorer = RawPreservingEvidence(), CandidateScoreMLP(feature_count=FEATURE_COUNT)
        before = {name: {key: value.clone() for key, value in model.state_dict().items()}
                  for name, model in (('R', evidence), ('C', scorer))}
        query = evidence(*tensors())
        raw = torch.randn(2, 768)
        adapted = torch.stack([evidence(raw[i:i+1], torch.tensor([[.1, .1, .5, .5]]),
                        torch.ones(1), torch.zeros(1, dtype=torch.int64)).embedding for i in range(2)])
        rows, context = candidate_tensors(query, raw, adapted, torch.tensor([[1, 0, 0, 0, 1], [0, .2, 2, .8, 1]]).float())
        assert rows.shape == (2, 14) and context.shape == (6,)
        assert sum(p.numel() for p in scorer.parameters()) == 804
        logits = scorer(rows, context)
        # Synthetic external target index, not a dataset or forward label.
        torch.nn.functional.cross_entropy(logits[None], torch.tensor([0])).backward()
        for name, model in (('R', evidence), ('C', scorer)):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
            assert all(torch.equal(before[name][key], value) for key, value in model.state_dict().items())


def test_tensor_features_and_scores_reexecute_candidate_permutation_equivariantly():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(67)
        query = RawPreservingEvidence()(*tensors())
        raw, adapted = torch.randn(3, 768), torch.randn(3, 768)
        stats = torch.tensor([[1, 0, 0, 0, 1], [0, .2, 2, .8, 1], [0, .4, 4, .6, 2]]).float()
        permutation = torch.tensor([2, 0, 1])
        rows, q = candidate_tensors(query, raw, adapted, stats)
        other, other_q = candidate_tensors(query, raw[permutation], adapted[permutation], stats[permutation])
        assert torch.equal(rows[permutation], other) and torch.equal(q, other_q)
        model = CandidateScoreMLP(feature_count=14)
        a, b = model(rows, q), model(other, other_q)
        assert torch.equal(a[:3][permutation], b[:3]) and torch.equal(a[3:], b[3:])


def test_empty_pool_and_semantic_ID_or_GT_purity_argument_are_not_accepted():
    query = RawPreservingEvidence()(*tensors())
    rows, context = candidate_tensors(query, torch.empty(0, 768), torch.empty(0, 768), torch.empty(0, 5))
    assert rows.shape == (0, 14) and context[-1] == -1
    with pytest.raises(TypeError):
        candidate_tensors(query, torch.empty(0, 768), torch.empty(0, 768), torch.empty(0, 5), category_id=1)
    with pytest.raises(ValueError):
        candidate_tensors(query, torch.ones(1, 768), torch.ones(1, 768), torch.tensor([[0, 1.2, 0, 0, 1]]))
