import pytest
from src.trackocd_cvpr.state_trajectories import ExternalStateAudit
from src.trackocd_cvpr.cross_video_forensics import CrossVideoTrajectories


def test_cross_video_reuse_counts_before_write_not_final_video_count():
    a, b = ExternalStateAudit(), CrossVideoTrajectories()
    rows = [(1, 1, 'NEW'), (1, 2, 'EXISTING'), (2, 2, 'EXISTING'), (2, 2, 'EXISTING')]
    for seq, (video, category, kind) in enumerate(rows):
        observed = a.observe(seq, video, {'kind': kind, 'token': 'S:0'}, 'S:0' if seq else None, category, 'known')
        b.consume({'sequence': seq, 'video_id': video, 'action': kind,
                   'category_id_evaluator_only': category, **observed})
    s = b.finish()[0]
    assert s['existing_reuses'] == 3 and s['video_count'] == 2
    assert s['cross_video_existing_reuses'] == 2
    assert s['cross_video_existing_reuses_after_bad_write'] == 2
    assert s['cross_video_existing_reuses_after_mixed'] == 2
    assert 'categories' not in s


def test_reject_shuffled_or_fabricated_purity():
    a, b = ExternalStateAudit(), CrossVideoTrajectories()
    observed = a.observe(0, 1, {'kind': 'NEW', 'token': 'S:0'}, None, 9, 'pseudo_novel')
    row = {'sequence': 0, 'video_id': 1, 'action': 'NEW', 'category_id_evaluator_only': 9, **observed}
    with pytest.raises(ValueError):
        b.consume({**row, 'sequence': 3})
    with pytest.raises(ValueError):
        CrossVideoTrajectories().consume({**row, 'selected_after': {**observed['selected_after'], 'purity': .5}})
