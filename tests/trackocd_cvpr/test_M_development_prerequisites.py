import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.M_development_cohort import cohort_sources, verified_cohort
from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP

ROOT = Path(__file__).resolve().parents[2]


def tiny_videos():
    videos = []
    for video, axis in ((1, 0), (2, 0), (3, 7)):
        visual = np.zeros((2, 768), np.float32); visual[:, axis] = 1
        arrays = {'track_id': np.array([17], np.int64), 'offsets': np.array([0, 2], np.int64),
            'frame_index': np.array([0, 1], np.int64), 'clock_frame_index': np.array([0, 1], np.int64),
            'visual': visual, 'geometry': np.array([[0., 0., 1., 1.]] * 2, np.float32), 'quality': np.ones(2, np.float32)}
        videos.append((video, arrays))
    return videos


def test_all9_missing_cohort_blocks_before_private_dev_no_partial_winner(tmp_path):
    assert len(cohort_sources()) == 27 and len(set(cohort_sources())) == 27
    # A synthetic absent cohort remains invalid after all real fits finish.
    # Preflight fails before even Git source checks or private endpoint reads.
    with pytest.raises(FileNotFoundError): verified_cohort(tmp_path)


@pytest.mark.parametrize('name', ['Mean_latest', 'StateEMA_latest', 'relative_quarantine', 'relative_weighted', 'pooled_weighted'])
def test_actual_small_causal_M_or_baseline_with_full_denominator_versions_and_ID_probe(name):
    videos = tiny_videos(); labels = {(1, '17'): ('declared_pseudoNovel', 20),
        (2, '17'): ('declared_pseudoNovel', 20), (3, '17'): ('Known', 10)}
    vector = np.zeros(768, np.float32); vector[7] = 1
    cfg = {'physical_tracks': 3, 'observations': 6, 'metadata_frames': 6, 'empty_frames': 0,
           'prefix_snapshots': [1, 2, 4, 8, 16], 'max_versions': 100,
           'raw_fixed_thresholds': {'tau_known': .65, 'tau_existing': .55}}
    model = None if name in ('Mean_latest', 'StateEMA_latest') else MemberReliabilityMLP(pooled_control=name == 'pooled_weighted').eval().requires_grad_(False)
    result, private = readonly_M_stream(videos, labels, {0: vector}, {0: 10}, cfg, name, 1027, lambda: None, model=model)
    renamed, new_private = readonly_M_stream(videos, labels, {0: vector}, {0: 10}, cfg, name, 1027, lambda: None,
                                            model=model, rename_and_reverse=True)
    assert result == renamed
    assert [r.events for _, r, _ in private] == [r.events for _, r, _ in new_private]
    assert result['observations'] == 6 and result['full_physical_bitset_scope'] == 3
    assert result['optimizer_steps'] == 0 and result['model_state_unchanged']
    assert not result['current_active_purity_official_CT_full_semantic_truth_Pred_or_self_exclusion_claim']


def test_metrics_manifest_is_not_Mdev_execution_or_full_CT_acceptance():
    c = json.loads((ROOT / 'configs/trackocd_cvpr/M_GT_causal_development_v1.json').read_text())
    assert c['status'].endswith('EXECUTION_DRIVER_PENDING')
    assert c['order_seeds'] == c['training_seeds'] == [1027, 1028, 1029]
    assert c['physical_tracks'] == 472 and c['observations'] == 8753 and c['optimizer_steps'] == 0
    assert not c['select_model_or_checkpoint'] and not c['execution_or_Model_Goal_acceptance_from_this_manifest']
