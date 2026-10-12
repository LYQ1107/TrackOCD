import numpy as np
import pytest
import torch
from src.trackocd_core.features import prefix_view
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence, raw_geometry_loss


def tensors():
    visual = torch.zeros(3, 768)
    visual[0, 0], visual[1, 1], visual[2, 2] = 50, 10, 2
    boxes = torch.tensor([[.1, .1, .5, .5], [.2, .1, .6, .5], [.3, .1, .7, .5]])
    return visual, boxes, torch.tensor([.3, .5, .7]), torch.tensor([0, 2, 5])


@pytest.mark.parametrize('temporal', (False, True))
def test_identity_start_preserves_actual_Raw_magnitude_weighted_geometry(temporal):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61)
        model = RawPreservingEvidence(temporal=temporal)
        visual, boxes, quality, elapsed = tensors()
        result = model(visual, boxes, quality, elapsed)
        assert torch.equal(result.raw, result.embedding)
        expected = (visual * (quality / quality.sum())[:, None]).sum(0)
        assert torch.equal(result.raw, expected / expected.norm())
        assert torch.allclose(result.visible_weights, quality / quality.sum())
        assert 0 <= result.angular_uncertainty <= 1


def test_modes_share_parameter_layout_not_declared_equal_function_class():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(62)
        temporal, pooled = RawPreservingEvidence(), RawPreservingEvidence(temporal=False)
        pooled.load_state_dict(temporal.state_dict())
        assert sum(p.numel() for p in temporal.parameters()) == sum(p.numel() for p in pooled.parameters()) == 50113
        assert torch.equal(temporal(*tensors()).embedding, pooled(*tensors()).embedding)
        with torch.no_grad():
            temporal.residual[-1].weight.fill_(.002)
            temporal.gates[-1].weight.fill_(.03)
        pooled.load_state_dict(temporal.state_dict())
        assert not torch.equal(temporal(*tensors()).embedding, pooled(*tensors()).embedding)


def test_real_changed_future_buffers_are_excluded_before_model_receives_prefix():
    with torch.random.fork_rng(devices=[]):
        model = RawPreservingEvidence()
        v, b, q, t = tensors()
        def input_from_future(value):
            full_v = np.concatenate((v.numpy(), np.full((2, 768), value, np.float32)))
            full_b = np.concatenate((b.numpy(), np.full((2, 4), np.nan, np.float32)))
            full_q = np.concatenate((q.numpy(), np.full(2, np.nan, np.float32)))
            prefix = prefix_view(full_v, full_b, full_q, [0, 2, 5, 8, 10], 3)
            return tuple(torch.tensor(array.copy()) for array in (prefix.visual, prefix.boxes_normalized_xyxy,
                                prefix.quality, prefix.elapsed_frames))
        left, right = model(*input_from_future(100)), model(*input_from_future(np.nan))
        assert torch.equal(left.embedding, right.embedding)
        assert torch.equal(left.visible_weights, right.visible_weights)


def test_quality_zero_fallback_no_future_count_or_category_argument():
    model = RawPreservingEvidence()
    v, b, q, t = tensors()
    result = model(v, b, torch.zeros_like(q), t)
    assert torch.allclose(result.visible_weights, torch.ones(3) / 3)
    with pytest.raises(TypeError):
        model(v, b, q, t, category_id=1)
    with pytest.raises(TypeError):
        model(v, b, q, t, future_total_length=100)


@pytest.mark.parametrize('field', ('time', 'box', 'quality', 'zero_visual', 'nonfinite'))
def test_malformed_current_prefix_fails_without_repair(field):
    v, b, q, t = tensors()
    if field == 'time':
        t = torch.tensor([1, 2, 5])
    elif field == 'box':
        b[0, 2] = 0
    elif field == 'quality':
        q[0] = -1
    elif field == 'zero_visual':
        v[0] = 0
    else:
        v[0, 1] = float('nan')
    with pytest.raises(ValueError):
        RawPreservingEvidence()(v, b, q, t)


def test_Raw_teacher_geometry_loss_detaches_teacher_not_student():
    with torch.random.fork_rng(devices=[]):
        raw = torch.randn(4, 768, requires_grad=True)
        adapted = torch.randn(4, 768, requires_grad=True)
        loss = raw_geometry_loss(raw, adapted)
        loss.backward()
        assert raw.grad is None and adapted.grad.abs().sum() > 0
        assert raw_geometry_loss(raw, raw).item() == 0
        with pytest.raises(ValueError):
            raw_geometry_loss(raw[:0], adapted[:0])


def test_extreme_finite_negative_gates_do_not_produce_zero_weight_nan():
    model = RawPreservingEvidence()
    with torch.no_grad():
        model.gates[-1].bias.fill_(-1000)
    output = model(*tensors())
    assert torch.isfinite(output.embedding).all()
    assert torch.isclose(output.visible_weights.sum(), torch.tensor(1.), rtol=0,
                         atol=2 * torch.finfo(torch.float32).eps)
    assert torch.all(output.visible_weights > 0)
    output.embedding.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_finite_values_with_overflowing_norm_rejected_not_normalized_to_zero():
    v, b, q, t = tensors()
    v[0] = 1e30
    with pytest.raises(ValueError, match='nonzero visible'):
        RawPreservingEvidence()(v, b, q, t)
