import json
from pathlib import Path
import pytest
from src.trackocd_core.evaluation import Target, TrackKey
from src.trackocd_core.evaluation.persistent import fixed_cross_video_targets
from src.trackocd_core.evaluation.physical_support import cross_video_support


def row(video, local, category=10, count=2, role='novel'):
    return {'video_id':video,'gt_local_id':local,'category_id':category,'role':role,'reliable_predicted_observations':count}


def test_missing_earlier_source_does_not_erase_fixed_opportunity_or_fake_support():
    rows = [row(1,7,count=0),row(2,7),row(3,7)]
    r = cross_video_support(rows,[1,2,3])
    assert r['fixed_gt_reuse_opportunities']==2 and r['reliable_current_opportunities']==2
    assert r['prefix_support']['1']['both_earlier_and_current_have_at_least_p']==1
    assert r['current_reliable_but_no_earlier_reliable_same_category']==1
    assert r['prefix_support']['4']['fixed_gt_opportunity_denominator']==2
    assert r['prefix_support']['4']['both_earlier_and_current_have_at_least_p']==0


def test_same_video_two_individuals_are_not_cross_video_sources_to_each_other():
    rows = [row(1,1,count=0),row(2,1),row(2,2)]
    r = cross_video_support(rows,[1,2])
    assert r['fixed_gt_reuse_opportunities']==2
    assert r['prefix_support']['1']['both_earlier_and_current_have_at_least_p']==0
    backwards=cross_video_support(rows,[2,1])
    assert backwards['fixed_gt_reuse_opportunities']==1
    assert backwards['missing_current_physical_opportunities']==1


def test_uncovered_categories_and_short_tracks_remain_in_denominator_and_not_semantic_score():
    rows = [row(1,1),row(2,1,count=1),row(1,2,11,count=0),row(2,2,11,count=0),row(3,3,12),row(2,99,category=1,role='known')]
    r = cross_video_support(rows,[1,2,3])
    assert r['gt_categories_with_cross_video_reuse']==2
    assert r['fixed_gt_reuse_opportunities']==2
    assert r['prefix_support']['1']['fraction_of_all_fixed_gt_opportunities']==.5
    assert r['prefix_support']['2']['both_earlier_and_current_have_at_least_p']==0
    assert r['semantic_decisions_or_correct_reuse_computed'] is False
    assert r['short_tracks_removed_from_denominator'] is False


def test_one_video_zero_denominator_is_null_and_duplicate_namespace_rejected():
    assert cross_video_support([row(1,1)],[1])['prefix_support']['1']['fraction_of_all_fixed_gt_opportunities'] is None
    with pytest.raises(ValueError): cross_video_support([row(1,1),row(1,1)],[1])
    with pytest.raises(ValueError): fixed_cross_video_targets([Target(TrackKey(2,'1'),10,'novel')],[1])
    with pytest.raises(ValueError): fixed_cross_video_targets([], [1,1])
    with pytest.raises(ValueError): cross_video_support([row(1,1)],[1],prefixes=[16])


def test_registered_only_fixed_input_counts_no_model_or_primary_pass():
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'configs/trackocd_core/physical_cross_video_support.json').read_text())
    assert c['videos']==988 and c['known_gt_tracks']==4413 and c['novel_gt_tracks']==819
    assert c['registered_orders']==['main','seed1027','seed1028','seed1029']
    assert c['prefixes']==[1,2,4,8,16] and c['limits']['cpu_workers']==1
    for k in ('model_inference_or_training','test_access','new_images_weights_or_feature_cache','primary_freeze_permitted','semantic_scientific_pass_permitted'):
        assert c[k] is False


def test_actual_support_receipt_keeps_all_orders_prefixes_and_same_fixed_denominators():
    root=Path(__file__).resolve().parents[2]
    r=json.loads((root/'outputs/trackocd_core/audit/physical_cross_video_support.json').read_text())
    assert r['status']=='COMPLETE_POSTHOC_PHYSICAL_SUPPORT_NOT_SEMANTIC_SUCCESS'
    assert r['videos']==988 and r['images']==36375
    front=r['frontends']; assert set(front)=={'MASA_NATIVE','PANDAS_BT_FG0'}
    assert front['MASA_NATIVE']['coverage']['novel']=={'gt_tracks':819,'reliably_observed':189}
    assert front['PANDAS_BT_FG0']['coverage']['novel']=={'gt_tracks':819,'reliably_observed':30}
    for order,denominator in [('main',527),('seed1027',529),('seed1028',547),('seed1029',531)]:
        for name in front:
            row=front[name]['orders'][order]
            assert row['fixed_gt_reuse_opportunities']==denominator
            assert row['gt_categories_with_cross_video_reuse']==98
            assert set(row['prefix_support'])=={'1','2','4','8','16'}
            counts=[row['prefix_support'][str(p)]['both_earlier_and_current_have_at_least_p'] for p in (1,2,4,8,16)]
            assert counts==sorted(counts,reverse=True)
            assert row['reliable_current_opportunities']+row['missing_current_physical_opportunities']==denominator
            assert row['semantic_decisions_or_correct_reuse_computed'] is False
    assert front['MASA_NATIVE']['orders']['main']['prefix_support']['1']['both_earlier_and_current_have_at_least_p']==87
    assert front['MASA_NATIVE']['orders']['main']['prefix_support']['16']['both_earlier_and_current_have_at_least_p']==69
    assert front['PANDAS_BT_FG0']['orders']['main']['prefix_support']['1']['both_earlier_and_current_have_at_least_p']==6
    for k in ('model_inference_or_training','test_access','primary_freeze_permitted','semantic_scientific_pass_permitted','gt_or_geometry_feedback_to_model','short_tracks_removed_from_denominator'):
        assert r[k] is False


def test_support_denominator_matches_independent_original_formula_not_only_shared_helper():
    rows=[row(1,1),row(1,2),row(2,1,count=0),row(3,1),row(2,2,category=11),row(3,2,category=11),row(1,4,role='known')]
    for order in ([1,2,3],[3,2,1],[2,1,3],[2,3,1]):
        ranks={v:i for i,v in enumerate(order)}
        novel=[r for r in rows if r['role']=='novel']
        expected=sum(any(q['category_id']==r['category_id'] and ranks[q['video_id']]<ranks[r['video_id']] for q in novel) for r in novel)
        assert cross_video_support(rows,order)['fixed_gt_reuse_opportunities']==expected
