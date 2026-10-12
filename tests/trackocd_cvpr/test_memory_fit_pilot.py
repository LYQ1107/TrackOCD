import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.trackocd_cvpr.memory_fit_pilot import fit_memory_pilot
from scripts.trackocd_cvpr.pilot_memory_optimizer import CONFIG, validate_configuration

ROOT = Path(__file__).resolve().parents[2]


def synthetic_policy_clock():
    # Synthetic aliases exercise the real controller/loss/optimizer plumbing.
    # No private files, deployed purity claims or scientific fit counted here.
    videos = []; targets = {}
    extras = 895
    for video in range(10):
        tracks = 5 if video < 6 else 6
        clock = np.arange(38 if video < 4 else 37, dtype=np.int64)
        obs_track, obs_frame = [], []
        for frame in clock:
            if video == 0 and frame == 0: continue
            first = 1 if video == 0 else 0
            available = extras if frame == first or video == 9 else max(extras - 54, 0)
            active = 1 + min(available, tracks - 1); extras -= active - 1
            for track in range(active): obs_track.append(track); obs_frame.append(frame)
        visual = np.zeros((len(obs_track), 768), np.float32); visual[:, 0] = 1
        order = np.argsort(obs_track, kind='stable')
        arrays = {'track_id': np.arange(tracks, dtype=np.int64),
            'offsets': np.concatenate(([0], np.cumsum(np.bincount(obs_track, minlength=tracks)))),
            'frame_index': np.array(obs_frame, dtype=np.int64)[order],
            'clock_frame_index': clock, 'visual': visual,
            'geometry': np.tile(np.array([0, 0, 1, 1], np.float32), (len(visual), 1)),
            'quality': np.ones(len(visual), np.float32)}
        videos.append((video, arrays))
        targets.update({(video, track): ('Known', 0) for track in range(tracks)})
    return videos, targets


def test_synthetic_full_clock_optimizer_updates_not_real_dataset_or_science():
    cfg = json.loads((ROOT / CONFIG).read_text())
    videos, targets = synthetic_policy_clock()
    model, optimizer, summary, sealed, history, versions, commits = fit_memory_pilot(
        videos, SimpleNamespace(targets=targets), {}, cfg, lambda: None, lambda row, curve: None)
    assert summary['optimizer_steps'] == 10 and summary['trainable_parameters'] == 161
    assert summary['model_parameter_sha256_before'] != summary['model_parameter_sha256_after']
    assert summary['complete_rollout']['physical_observations'] == 1268
    assert summary['complete_rollout']['metadata_frames'] == 374 and len(sealed) == 374
    assert len(history) == len(commits) and all(v.commit_watermark <= len(commits) for v in versions)
    assert not summary['GT_driven_memory_repair_or_deleted_wrong_history']
    assert not summary['strict_historical_purity_or_CT_or_Novel_gain_measured']


@pytest.mark.parametrize('field,value', [('optimizer_steps', 200), ('seed', 1029),
    ('execution_device', 'CUDA'), ('dev_final_Val_Test_new_nonKnown_labels_or_pixels', True)])
def test_pilot_cannot_silently_expand_execution_or_semantic_scope(field, value):
    cfg = json.loads((ROOT / CONFIG).read_text()); cfg[field] = value
    with pytest.raises(ValueError):
        validate_configuration(cfg, {'status': 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'})
