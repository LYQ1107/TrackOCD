import pytest
import torch
import json
from pathlib import Path

from src.trackocd_core.representation import CategoryEvidence, cross_video_category_loss
from src.trackocd_core.experiment_config import representation_config


def test_evidence_weights_are_causal_and_embedding_matches_truncated_forward():
    torch.manual_seed(1027)
    model = CategoryEvidence(True).eval()
    visual = torch.randn(2, 16, 768)
    quality = torch.rand(2, 16)
    elapsed = torch.arange(16).expand(2, -1).float() * 30
    full = model(visual, quality, elapsed)
    for p in (1, 2, 4, 8, 16):
        before = model(visual[:, :p], quality[:, :p], elapsed[:, :p])
        torch.testing.assert_close(before['weights'], full['weights'][:, :p], atol=1e-6, rtol=1e-6)
        changed = visual.clone()
        changed[:, p:] = 1e5
        after = model(changed[:, :p], quality[:, :p], elapsed[:, :p])
        torch.testing.assert_close(before['embedding'], after['embedding'])
        torch.testing.assert_close(before['embedding'].norm(dim=-1), torch.ones(2))
        assert (before['effective_maturity'] <= p + 1e-5).all()


def test_adapter_initialization_and_inputs_match_for_A1_A2():
    torch.manual_seed(1027)
    a1 = CategoryEvidence(False)
    torch.manual_seed(1027)
    a2 = CategoryEvidence(True)
    for name, value in a1.adapter.state_dict().items():
        torch.testing.assert_close(value, a2.adapter.state_dict()[name], atol=0, rtol=0)
    assert sum(p.numel() for p in a1.parameters()) == 525056
    assert sum(p.numel() for p in a2.parameters()) == 541889
    assert tuple(__import__('inspect').signature(a2.forward).parameters) == ('visual', 'quality', 'elapsed')


def test_cross_video_contrastive_has_no_same_individual_positive_or_self_shortcut():
    embedding = torch.eye(4, requires_grad=True)
    categories = torch.tensor([1, 1, 2, 2])
    videos = torch.tensor([10, 11, 12, 13])
    identities = torch.tensor([7, 7, 7, 7])  # local ID is not globally shared identity.
    loss = cross_video_category_loss(embedding, categories, videos, identities)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(embedding.grad).all()
    with pytest.raises(ValueError, match='cross-video'):
        cross_video_category_loss(embedding, categories, torch.tensor([1, 1, 2, 2]), identities)


def test_single_correction_cannot_change_data_seeds_budget_or_capacity(tmp_path):
    root = Path(__file__).resolve().parents[2]
    path = root / 'configs/trackocd_core/gt_representation_correction_r1.json'
    base = json.loads((root / 'configs/trackocd_core/gt_representation_pilot.json').read_text())
    merged = representation_config(root, path)
    for key in ('data_plan', 'semantic_adapter', 'training_seeds', 'steps_per_model_seed', 'batch_tracks',
                'optimizer', 'learning_rate', 'corruption', 'evaluation', 'maximum_training_wall_seconds'):
        assert merged[key] == base[key]
    assert merged['teacher_geometry_loss_weight'] == 5 and merged['root_cause_correction_rounds_used'] == 1
    bad = json.loads(path.read_text())
    bad['training_seeds'] = [123]
    modified = tmp_path / 'forbidden_override.json'
    modified.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Only one'):
        representation_config(root, modified)
