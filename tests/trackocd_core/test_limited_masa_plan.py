import json
from pathlib import Path
from src.trackocd_core.limited_masa_plan import make_plan


def test_all_logical_cases_and_only_exact_operating_point_aliases():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/trackocd_core/limited_masa_evaluation.json').read_text())
    freeze=json.loads((root/'outputs/trackocd_core/core_training/train_first_v1/model_freeze.json').read_text())
    policy=json.loads((root/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    p=make_plan(cfg,freeze,policy,list(range(988)))
    assert len(p['logical_cases'])==2040 and len(p['jobs'])==840 and len(p['groups'])==16
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
