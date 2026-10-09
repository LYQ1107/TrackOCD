import torch

from src.trackocd_v2.models.semantic_adapter import (
    SemanticAdapter,
    SemanticAdapterConfig,
    hard_negative_ranking_loss,
    multi_positive_contrastive_loss,
    prefix_consistency_loss,
)


def test_semantic_adapter_preserves_instance_branch_and_normalizes_semantic_branch():
    model = SemanticAdapter(SemanticAdapterConfig(input_dim=4, semantic_dim=3, hidden_dim=5))
    x = torch.randn(2, 4)
    output = model(x)
    assert output["z_instance"] is not x or torch.equal(output["z_instance"], x)
    assert torch.equal(output["z_instance"], x)
    assert output["z_semantic"].shape == (2, 3)
    assert torch.allclose(output["z_semantic"].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert model.metadata()["test_semantic_accessed"] is False


def test_train_losses_use_cross_track_positives_and_hard_negatives():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    labels = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([10, 11, 12, 13])
    contrastive = multi_positive_contrastive_loss(embeddings, labels, group_ids=groups)
    ranking = hard_negative_ranking_loss(embeddings, labels, group_ids=groups)
    assert torch.isfinite(contrastive)
    assert torch.isfinite(ranking)
    assert float(contrastive) >= 0.0
    assert float(ranking) == 0.0


def test_prefix_consistency_supports_causal_valid_mask():
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    second = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    value = prefix_consistency_loss(first, second, valid=torch.tensor([True, False]))
    assert float(value) == 0.0
