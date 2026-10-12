import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.fit_R_training import FitTrack, support_groups, current_R_fit_loss, bounded_fit_steps
from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence


def track(video, target, seed):
    rng = np.random.default_rng(seed)
    visual = rng.standard_normal((3, 768)).astype(np.float32)
    visual /= np.linalg.norm(visual, axis=1, keepdims=True)
    a = {'visual': visual.astype(np.float16), 'geometry': np.array([[0, 0, 1, 1]] * 3, np.float32),
         'quality': np.ones(3, np.float32), 'frame_index': np.array([1, 3, 5], np.int64),
         'offsets': np.array([0, 3], np.int64), 'track_id': np.array([seed], np.int64)}
    return FitTrack(video, a, 0, target)


def test_support_excludes_whole_query_video_masks_unknown_and_scope_deduplicates():
    a = track(10, 0, 1); alias = track(10, 0, 1); cross = track(11, 0, 1)
    other = track(12, 1, 2); unknown = track(99, None, 3)
    groups = support_groups([a, alias, cross, other, unknown], 999, 2)
    assert [len(g) for g in groups] == [2, 1]  # samevideo alias only, not crossvideo identical values
    groups = support_groups([a, alias, cross, other, unknown], 10, 2)
    assert [len(g) for g in groups] == [1, 1]
    shifted = track(10, 0, 1); shifted.arrays['frame_index'] += 100
    assert [len(g) for g in support_groups([a, alias, shifted, other], 999, 2)] == [2, 1]
    with pytest.raises(ValueError): support_groups([a, other], 10, 2)


def test_actual_fit_loss_support_query_gradients_and_no_unknown_loss_target():
    model = RawPreservingEvidence(); tracks = [track(10, 0, 1), track(11, 0, 2), track(12, 1, 3)]
    groups = support_groups(tracks, 10, 2)
    loss, _, support, query, proof = current_R_fit_loss(model, tracks[0].visible(2), groups, 0,
        temperature=0.1, geometry_weight=0.25, consistency_weight=0.1)
    loss.backward()
    assert support.grad.norm() > 0 and query.grad.norm() > 0 and model.residual[-1].weight.grad.norm() > 0
    assert proof['groups_reencoded'] == 2
    with pytest.raises(ValueError): current_R_fit_loss(model, tracks[0].visible(2), groups, None,
        temperature=0.1, geometry_weight=0.25, consistency_weight=0.1)


def test_two_synthetic_optimizer_steps_change_only_fresh_R_not_inputs_or_targets():
    model = RawPreservingEvidence()
    tracks = [track(v, c, v) for c, v in ((0, 10), (0, 11), (1, 12), (1, 13))] + [track(99, None, 99)]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    original = [t.arrays['visual'].copy() for t in tracks]
    cfg = dict(seed=1027, class_count=2, optimizer_steps=2, learning_rate=0.001, weight_decay=0.0001,
               temperature=0.1, geometry_weight=0.25, consistency_weight=0.1, gradient_clip=1.)
    _, curve, _ = bounded_fit_steps(model, tracks, cfg, lambda: None, lambda row: None)
    assert len(curve) == 2 and not torch.equal(before['residual.2.weight'], model.state_dict()['residual.2.weight'])
    for t, array in zip(tracks, original): np.testing.assert_array_equal(t.arrays['visual'], array)
    assert tracks[-1].loss_target is None


def test_config_remains_pending_realcache_and_executable_not_scientific_launch():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / 'configs/trackocd_cvpr/fit_R_optimizer_smoke_v1.json').read_text())
    assert cfg['optimizer_steps'] == 10 and cfg['class_count'] == 5 and cfg['execution_device'] == 'CPU'
    assert cfg['status'] == 'REGISTERED_OPTIMIZER_PLUMBING_SOURCE_CACHE_COMPLETION_REQUIRED'
    assert not cfg['full_scientific_fit_automatic_start']
    assert cfg['future_independent_scientific_seeds_not_automatic_launch'] == [1027, 1028, 1029]
