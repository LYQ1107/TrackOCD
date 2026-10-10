import numpy as np
from src.trackocd_core.limited_masa_features import select_all_prefixes


def test_all_predicted_ids_short_and_low_score_kept_no_future_pick():
    video={'images':[{'frame_index':i} for i in range(4)]}
    a={'frame_offsets':np.array([0,2,4,5,7]),'track_id':np.array([9,2,9,2,9,9,77]),
       'score':np.array([.01,.2,.5,.6,.9,1.,.001])}
    rows,count=select_all_prefixes(video,a,2)
    assert [r['physical_track_id'] for r in rows]==[2,9,77] and count==5
    assert [o['source_offset'] for o in rows[1]['observations']]==[0,2]
    assert len(rows[-1]['observations'])==1
    # A future high score cannot replace the first low-score observation.
    b={**a,'score':np.array([.01,.2,.5,.6,.99,.999,.001])}
    assert select_all_prefixes(video,b,2)==(rows,count)


def test_actual_first_four_frozen_videos_keep_every_id_and_full_prefix_count():
    import json
    from pathlib import Path
    from src.trackocd_core.physical_qualification import completed_video
    root=Path(__file__).resolve().parents[2];cfg=json.loads((root/'configs/trackocd_core/limited_masa_features.json').read_text())
    plan=json.loads((root/cfg['private_metadata_plan']).read_text())
    assert [v['video_id'] for v in plan['videos'][:4]]==[4,20,22,23]
    for v in plan['videos'][:4]:
        done=completed_video(root/cfg['prediction_run'],v,cfg['prediction_config_sha256'])
        with np.load(root/cfg['prediction_run']/'shards'/done['npz_filename'],allow_pickle=False) as a:
            rows,count=select_all_prefixes(v,a)
            ids,n=np.unique(a['track_id'],return_counts=True)
            assert {r['physical_track_id'] for r in rows}==set(map(int,ids))
            assert count==int(np.minimum(n,16).sum())
            assert all(len(r['observations'])>=1 for r in rows)


def test_feature_source_enforces_freeze_barrier_resume_and_no_physical_rerun():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];source=(root/'scripts/trackocd_core/extract_limited_masa_features.py').read_text()
    assert 'FROZEN_TRAIN_ONLY_MODELS_AND_OPERATING_POINTS' in source
    assert '/TAO-Amodal/annotations/' in source and '/frames/test/' in source
    assert "'/train_labels.parquet'" in source and 'completed_feature' in source
    assert 'PASS_FROZEN_FEATURE_BASELINE_POLICY_EVALUATOR_INTEGRATION' in source
    assert 'start_new_session=True' in source and 'for p in children:' in source
    assert 'from src.trackocd_core.masa_native' not in source


def test_actual_four_video_features_all_ids_quality_payload_and_tiny_reuse():
    import json
    from pathlib import Path
    from src.trackocd_core.limited_masa_features import completed_feature,LimitedMasaVideoCache
    from src.trackocd_v2.io import sha256_file
    root=Path(__file__).resolve().parents[2];out=root/'outputs/trackocd_core/features/masa_limited_prefix_v1'
    manifest=json.loads((out/'integration_manifest.json').read_text())
    assert manifest['videos']==4 and manifest['tracks']==892 and manifest['observations']==5160
    assert manifest['reused_smoke_observations']==8
    assert not manifest['GT_or_role_input'] and not manifest['detector_tracker_inference'] and not manifest['optimizer_used']
    assert not manifest['TAO_Test_access'] and manifest['limited_not_strong_frontend']
    for r in manifest['records']:
        assert completed_feature(out/'shards',r['video_id'],manifest['config_sha256'],r['input_npz_sha256'])==r
        c=LimitedMasaVideoCache(out/'shards'/r['npz_filename'],r['video_id'])
        assert len(c.rows)==r['tracks'] and sum(a['observation_count'] for a in c.rows)==r['observations']
        for row in c.rows:
            for p in (1,2,4,8,16):
                v=c.get_prefix(row['key'],p);assert len(v.visual)==min(p,row['observation_count'])
                assert np.isfinite(v.visual).all() and np.all((v.quality>=0)&(v.quality<=1))
        c.close()
    cfg=root/'configs/trackocd_core/limited_masa_features.json';assert manifest['config_sha256']==sha256_file(cfg)


def test_semantic_integration_source_seals_before_evaluator_join_and_no_tuning():
    from pathlib import Path
    root=Path(__file__).resolve().parents[2];s=(root/'scripts/trackocd_core/integrate_limited_masa.py').read_text()
    assert s.index('sealed,runtime=replay(')<s.index("gt=json.loads(gt_path.read_text())")
    assert "'prototype_coverage':{'available':48,'inherited_known':78,'missing':30" in s
    assert 'known_ids=known_ids' in s and "'Val_tuning':False" in s
    assert "matches={TrackKey(r['video_id'],str(r['physical_track_id'])):None for r in routes}" in s
