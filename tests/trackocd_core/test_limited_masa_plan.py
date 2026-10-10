import json
from pathlib import Path
from src.trackocd_core.limited_masa_plan import make_plan


def test_all_logical_cases_and_only_exact_operating_point_aliases():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/trackocd_core/limited_masa_evaluation.json').read_text())
    freeze=json.loads((root/'outputs/trackocd_core/core_training/train_first_v1/model_freeze.json').read_text())
    policy=json.loads((root/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    p=make_plan(cfg,freeze,policy,list(range(988)))
    assert len(p['logical_cases'])==2040 and len(p['jobs'])==840 and len(p['groups'])==18
    assert {r['method'] for r in p['logical_cases']}==set(cfg['representation_methods']+cfg['policy_methods'])
    for r in p['logical_cases']:
        j=p['jobs'][r['execution_id']]
        assert j['prefix']==r['prefix'] and j['order']==r['order'] and j['seed']==r['seed']
    for s in cfg['seeds']:
        a=[r for r in p['logical_cases'] if r['method']=='D2_RISK_AWARE' and r['seed']==s and r['coverage_target']==1.]
        b=[r for r in p['logical_cases'] if r['method']=='WITHOUT_TEMPORAL' and r['seed']==s and r['coverage_target']==1.]
        assert [r['execution_id'] for r in a]==[r['execution_id'] for r in b]


def test_full_inference_source_has_gt_barrier_exact_proof_atomic_resume_own_children():
    root=Path(__file__).resolve().parents[2];s=(root/'scripts/trackocd_core/run_limited_masa_inference.py').read_text()
    assert "'evaluator_only_geometry_join' in p" in s and "'physical_cross_video_support' in p" in s
    assert "'/TAO-Amodal/annotations/' in p" in s and "'/frames/test/' in p" in s
    assert "'PASS_EXACT_DECISION_EQUIVALENCE'" in s and "assignment['plan_sha256']" in s
    assert "if existing is not None:completed.append(job['id']);continue" in s
    assert 'start_new_session=True' in s and 'for p in children:' in s
    assert 'new_optimization' in s and 'from src.trackocd_core.masa_native' not in s


def test_fixed_pairs_cover_all_policy_operating_points_without_outcome_selection():
    from src.trackocd_core.limited_masa_pairs import registered_pairs
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/trackocd_core/limited_masa_evaluation.json').read_text())
    f=json.loads((root/'outputs/trackocd_core/core_training/train_first_v1/model_freeze.json').read_text())
    p=json.loads((root/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    plan=make_plan(cfg,f,p,list(range(988)));pairs=list(registered_pairs(plan['logical_cases'],cfg))
    assert len(pairs)==780
    policy=[r for r in pairs if r[0]=='D1_SIMPLE_MLP']
    assert len(policy)==180 and {r[5] for r in policy}=={.5,.75,1.}


def test_evaluator_gt_only_after_seal_primary_full_denoms_and_canonicalCSV_schema():
    import csv,io
    root=Path(__file__).resolve().parents[2];s=(root/'scripts/trackocd_core/evaluate_limited_masa.py').read_text()
    assert s.index('sealed=read_sealed(')<s.index("gt=json.loads(gt_path.read_text())")
    assert '(4413,819)' in s and '(1400,189)' in s and "cfg['fixed_full_GT']['reuse_opportunities'][job['order']]" in s
    assert 'same_global_mapping' not in s or 'no_second' in s
    assert 'bootstrap' in s and "if target==1." in s and "inference_path=out/'full_inference_manifest.json'" in s
    import ast
    ast.parse(s)
    meta=set(('scope','stage','method','backend','seed','order','prefix','coverage_target'))
    order=set(next(csv.reader(io.StringIO((root/'outputs/trackocd_core/PER_ORDER_RESULTS.csv').read_text()))))
    expected=meta|set(('old_acc','new_acc','h_score','all_acc','correct_commit_ct','false_merge_rate','false_split_new_rate','wrong_known_assignment_rate','wait_unresolved_rate','effective_commit_coverage'))
    assert order==expected
    error=set(next(csv.reader(io.StringIO((root/'outputs/trackocd_core/ERROR_BREAKDOWN.csv').read_text()))))
    outcomes=set(('pure_correct_reuse','false_split_new','wrong_known_assignment','contaminated_token_reuse','unverified_token_reuse','wrong_category_merge','same_video_only_unsupported_existing','missed_predicted_opportunity','wait','wait_with_no_decision_record'))
    assert error==meta|outcomes|set(('wrong_known_all_novel','novel_gt_denominator','memory_contamination_writes','contaminated_states'))


def test_joint_resource_reservation_is_only_live_owned_process_headroom(tmp_path):
    import os
    from scripts.trackocd_core.run_limited_masa_inference import remaining_RAM_reservation,counterpart_reservation
    class Child:
        pid=os.getpid()
        def poll(self):return None
    peak=16*2**30
    remaining=remaining_RAM_reservation([Child()],peak)
    assert 0<=remaining<peak
    class Exited(Child):
        def poll(self):return 0
    assert remaining_RAM_reservation([Exited()],peak)==0
    assert counterpart_reservation(tmp_path,'missing.json')==0
    (tmp_path/'state.json').write_text(json.dumps({'remaining_RAM_reservation_bytes':123}))
    assert counterpart_reservation(tmp_path,'state.json')==123


def test_actual_full_features_validated_without_dropping_short_or_unmatched():
    from src.trackocd_v2.io import sha256_file
    root=Path(__file__).resolve().parents[2]
    d=json.loads((root/'outputs/trackocd_core/LIMITED_MASA_FULL_FEATURE_VALIDATION.json').read_text())
    assert d['status']=='PASS_ALL_988_ATOMIC_FEATURE_SHARDS_AND_TRUE_FIRST_PREFIXES'
    assert (d['videos'],d['tracks'],d['observations'],d['singletons_retained'],d['reused_smoke_observations'])==(988,304561,1294110,114380,8)
    assert not d['Val_GT_read'] and not d['TAO_Test_access']
    assert d['validation_source_sha256']==sha256_file(root/'scripts/trackocd_core/validate_full_limited_masa_features.py')
    assert d['full_feature_manifest_sha256']==sha256_file(root/'outputs/trackocd_core/features/masa_limited_prefix_v1/full_manifest.json')
