import json
from pathlib import Path
import numpy as np
import pytest
from src.trackocd_core.train_first_split import known_tracks, split_tracks, resource_statistics
from src.trackocd_core.features import prefix_view

ROOT = Path(__file__).resolve().parents[2]


def test_short_known_tracks_retained_and_true_novel_never_enter_rows():
    gt = {'images':[{'id':1,'video_id':3,'frame_index':1,'file_name':'train/x.jpg'}],
          'annotations':[{'video_id':3,'track_id':4,'image_id':1,'category_id':7,'bbox':[0,0,2,2]},
                         {'video_id':3,'track_id':5,'image_id':1,'category_id':8,'bbox':[0,0,2,2]}]}
    rows = known_tracks(gt,{7})
    assert len(rows)==1 and rows[0]['total_observations']==1
    assert resource_statistics(rows,{7,9})['missing_train_known_ids']==[9]
    assert resource_statistics(rows,{7})['no_minimum_p16_filter']
    assert len(rows[0]['observations'])==1


def test_actual_train_first_manifest_is_disjoint_larger_than_old_pilot_and_no_gt_repair():
    path=ROOT/'outputs/trackocd_core/TRAIN_ONLY_CATEGORY_SPLIT_AUDIT.json'
    audit=json.loads(path.read_text()); split=audit['split']
    assert audit['statistics']['known_tracks']==2196 and audit['statistics']['known_categories_supported']==48
    assert split['partitions']['representation_fit']['tracks']==1305
    assert len(split['categories']['representation_fit'])==15
    categories=[set(v) for v in split['categories'].values()]
    assert all(not a&b for i,a in enumerate(categories) for b in categories[i+1:])
    videos=[set(v['videos']) for v in split['partitions'].values()]
    assert all(not a&b for i,a in enumerate(videos) for b in videos[i+1:])
    assert len(split['prototype_supported_known_ids'])==48 and len(split['prototype_missing_known_ids'])==30
    ep=json.loads((ROOT/'outputs/trackocd_core/TRAIN_ONLY_EPISODE_MANIFEST.json').read_text())
    assert 'predicted actions only' in ep['inference_state']
    assert not audit['val_or_test_access'] and not audit['training_or_inference_started']
    scope=json.loads((ROOT/'configs/trackocd_core/SCOPE_AMENDMENT_TRAIN_FIRST.json').read_text())
    assert scope['historical_M1_status']=='BLOCKED_FRONTEND_QUALITY'
    assert scope['training_waits_for_strong_M1'] is False
    assert scope['frontend']['strong_frontend_qualified'] is False


def test_short_prefix_does_not_synthesize_p16_or_read_future():
    visual=np.ones((2,768),dtype=np.float32); geometry=np.ones((2,4));quality=np.ones(2)
    one=prefix_view(visual,geometry,quality,[1,2],1)
    visual[1]=np.nan
    assert np.isfinite(one.weighted_mean()).all()
    with pytest.raises(ValueError): prefix_view(visual,geometry,quality,[1,2],2)


def test_split_rejects_non_known_or_no_valid_negatives():
    row={'category_id':8,'video_id':1,'total_observations':1,'observations':[{}]}
    with pytest.raises(ValueError,match='Train Known'): split_tracks([row],{7},{})


def test_actual_main_feature_cache_complete_frozen_and_short_prefixes_causal():
    from src.trackocd_core.train_first_cache import TrainFirstCache
    from src.trackocd_v2.io import sha256_file
    root=ROOT/'outputs/trackocd_core/features/train_first_v1';cache=TrainFirstCache(root)
    m=cache.manifest
    assert len(cache.rows)==m['tracks']==2166 and m['observations']==24628
    assert m['DINO_checkpoint_sha256']=='0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73'
    assert m['config_sha256']==sha256_file(ROOT/'configs/trackocd_core/training_split.json')
    assert not m['optimizer_used'] and not m['val_or_test_access'] and not m['historical_outputs_overwritten']
    assert all(w['frozen_encoder'] and not w['optimizer_used'] for w in m['workers'])
    assert m['reused_exact_protocol_observations']==sum(w['reused_observations'] for w in m['workers'])>0
    supervisor=json.loads((root/'supervisor.json').read_text())
    assert supervisor['error'] is None and supervisor['owned_child_returncodes']==[0]
    for row in cache.rows:
        for cap in (1,2,4,8,16):
            v=cache.get_prefix(row['key'],cap)
            assert len(v.visual)==min(cap,row['observation_count']) and np.isfinite(v.weighted_mean()).all()
