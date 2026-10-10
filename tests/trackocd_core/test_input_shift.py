import ast
import json
from pathlib import Path
from src.trackocd_v2.io import sha256_file
from scripts.trackocd_core.report_input_shift import length_profile


def test_shift_lengths_are_true_clipped_prefixes_and_keep_singletons():
    r=length_profile([1,2,4,16])
    assert r['tracks']==4 and r['singletons']==1 and r['observations']==23 and r['mean_available_observations']==5.75
    assert r['total_observations_at_cap']=={'1':4,'2':7,'4':11,'8':15,'16':23}


def test_shift_source_is_postseal_descriptive_not_model_or_label_repair():
    root=Path(__file__).resolve().parents[2];s=(root/'scripts/trackocd_core/report_input_shift.py').read_text();ast.parse(s)
    assert s.index('seal=json.loads')<s.index('gt=json.loads(gt_path.read_text())')
    assert 'posthoc_description_after_first_Val_metrics_not_preregistered_causal_hypothesis' in s
    assert "'models_or_operating_points_modified':False" in s and "'GT_to_live_inference':False" in s
    assert '256*2**20' in s and 'remaining_RAM_reservation_bytes' in s
    assert 'a[\'visual\']' not in s and 'optimizer' not in s


def test_actual_shift_receipt_is_bound_to_sealed_existing_assets():
    root=Path(__file__).resolve().parents[2];out=root/'outputs/trackocd_core'
    r=json.loads((out/'INPUT_SHIFT_DIAGNOSTIC_RESULT.json').read_text())
    assert r['status']=='COMPLETE_POSTHOC_DESCRIPTIVE_INPUT_SHIFT_NOT_CAUSAL_INTERVENTION'
    for field,path in [('source_sha256',root/'scripts/trackocd_core/report_input_shift.py'),
                       ('model_freeze_sha256',out/'core_training/train_first_v1/model_freeze.json'),
                       ('Native_feature_manifest_sha256',out/'features/masa_limited_prefix_v1/full_manifest.json'),
                       ('Train_feature_manifest_sha256',out/'features/train_first_v1/manifest.json'),
                       ('existing_GT_join_sha256',out/'audit/physical_cross_video_support_private/evaluator_only_geometry_join.json')]:
        assert r[field]==sha256_file(path)
    plan=json.loads((out/'core_training/train_first_v1/limited_masa_evaluation/plan.json').read_text())
    row=next(x for x in plan['logical_cases'] if (x['method'],x['order'],x['prefix'])==('B0_frame_snapshot_vote','main',1))
    seal=json.loads((out/f"core_training/train_first_v1/limited_masa_evaluation/cases/{row['execution_id']}/sealed_complete.json").read_text())
    assert r['sealed_reference_sha256']==seal['ledger']['sha256'] and seal['physical_tracks']==304561
    assert r['resources']['max_observed_RSS_bytes']<=256*2**20
    assert r['resources']['CPU_workers']==1 and not r['resources']['GPU_used']
    assert not any(r[k] for k in ('models_or_operating_points_modified','GT_to_live_inference','Val_training_or_tuning','TAO_Test_access'))


def test_actual_coverage_cohorts_and_true_prefix_profiles_reconcile():
    root=Path(__file__).resolve().parents[2];r=json.loads((root/'outputs/trackocd_core/INPUT_SHIFT_DIAGNOSTIC_RESULT.json').read_text())
    assert r['GT_counts_including_any_other_roles']=={'known':4413,'novel':819}
    assert r['Train_selected']['tracks']==2166 and r['Train_selected']['observations']==24628
    groups=r['Native_by_evaluator_reliable_match']
    assert {k:v['tracks'] for k,v in groups.items()}=={'known':1400,'novel':189,'other_matched':0,'unmatched_unknown':302972}
    for field in ('tracks','observations','singletons','at_least16'):
        assert sum(v.get(field,0) for v in groups.values())==r['Native_all'][field]
    for cap in ('1','2','4','8','16'):
        assert sum(v.get('total_observations_at_cap',{}).get(cap,0) for v in groups.values())==r['Native_all']['total_observations_at_cap'][cap]
    table={(x['prototype_group'],x['reliable_physical_match']):x['GT_tracks'] for x in r['Known_GT_prototype_and_physical_coverage']}
    assert table=={('fit15',True):1307,('fit15',False):2793,('prototype_only33',True):87,('prototype_only33',False):202,('no_prototype30',True):6,('no_prototype30',False):18}
    assert sum(table.values())==4413 and sum(n for (_,matched),n in table.items() if matched)==1400
    assert r['posthoc_description_after_first_Val_metrics_not_preregistered_causal_hypothesis']
    assert r['unmatched_not_assumed_background_or_proven_pollution']
