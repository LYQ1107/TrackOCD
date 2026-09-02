import torch

from src.iclr27_phase75e.pairwise_adapter import pairwise_torch_score


def test_assignment_is_candidate_permutation_invariant_and_gradients_finite():
    torch.manual_seed(7503)
    q = torch.nn.functional.normalize(torch.randn(4, 768), dim=-1).requires_grad_(True)
    c = torch.nn.functional.normalize(torch.randn(7, 768), dim=-1).requires_grad_(True)
    s1 = pairwise_torch_score(q, c)
    s2 = pairwise_torch_score(q, c[[3, 0, 6, 1, 5, 2, 4]])
    assert torch.allclose(s1, s2, atol=1e-6)
    s1.backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(c.grad).all()


def test_empty_or_bad_shapes_are_rejected():
    q = torch.zeros(2, 768)
    try:
        pairwise_torch_score(q, torch.zeros(2, 767))
    except ValueError:
        pass
    else:
        raise AssertionError("dimension mismatch must be rejected")
