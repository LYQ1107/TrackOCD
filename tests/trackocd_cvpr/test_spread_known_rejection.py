from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from src.trackocd_cvpr.spread_known_calendar import uniform_ranks,freeze_calendar
from src.trackocd_cvpr.paired_spread_known_rejection_training import paired_candidate_fit
from src.trackocd_cvpr.paired_known_rejection_training import paired_candidate_fit as original
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
from tests.trackocd_cvpr.test_policy_training_data import arrays


def data(rows=None):
    return ([(1,arrays() if rows is None else rows)],
        SimpleNamespace(targets={(1,12):('Known',0),(1,99):('policy_pseudo',0)}),
        {0:np.eye(1,768,0,dtype=np.float32)[0]})


def fit(inputs,k,coefficient=0.,plan=None):
    plan=freeze_calendar(*inputs[:2],1027,1,k,lambda:None) if plan is None else plan
    return paired_candidate_fit(*inputs,1027,episodes=1,updates_per_episode=k,
        checkpoint_steps=[k],guard=lambda:None,progress=lambda r,c:None,checkpoint=lambda *args:None,
        repair_coefficient=coefficient,frozen_calendar=plan)


def test_fixed_uniform_rank_formula_no_padding_or_search():
    ranks=uniform_ranks(333,50)
    assert len(set(ranks))==50 and ranks[0]==3 and ranks[-1]==329
    assert uniform_ranks(50,50)==tuple(range(50))
    with pytest.raises(ValueError): uniform_ranks(49,50)
    with pytest.raises(ValueError): uniform_ranks(True,1)


def test_plan_does_not_select_by_pseudo_targets_or_anonymous_history():
    inputs=data(); a=freeze_calendar(*inputs[:2],1027,1,1,lambda:None)
    inputs[1].targets[(1,99)]=('no_retained_semantic_truth',None)
    b=freeze_calendar(*inputs[:2],1027,1,1,lambda:None)
    assert a==b and a['eligible_Known_frame_counts']==[2]
    assert a['selected_frames']==(((0,0,5),),)


def test_selecting_all_anchors_zero_intervention_preserves_original_numeric_history():
    inputs=data(); models,_,summary,private=fit(inputs,2)
    old,_,oldsummary,oldprivate=original(*inputs,1027,episodes=1,updates_per_episode=2,
        checkpoint_steps=[2],guard=lambda:None,progress=lambda r,c:None,checkpoint=lambda *args:None,
        repair_coefficient=0.)
    for arm in summary['arms']:
        assert digest(models[arm])==digest(old[arm]) and repr(private[arm])==repr(oldprivate[arm])
        assert [r['loss'] for r in summary['loss_curves'][arm]]==[r['loss'] for r in oldsummary['loss_curves'][arm]]


def test_fixed_plan_future_probe_keeps_prior_predictions_not_reselected_training():
    inputs=data(); plan=freeze_calendar(*inputs[:2],1027,1,1,lambda:None)
    _,_,_,first=fit(inputs,1,plan=plan)
    changed=arrays(); changed['visual'][1]=0; changed['visual'][1,7]=1
    _,_,_,second=fit(data(changed),1,plan=plan)
    for arm in first: assert repr(first[arm][0][1][:3])==repr(second[arm][0][1][:3])


def test_repair_extra_gradient_on_fixed_spread_selected_synthetic_cooccurring_frame():
    rows=arrays()
    rows['offsets']=np.array([0,2,4],np.int64)
    rows['visual']=np.concatenate((rows['visual'],rows['visual'][2:3]),axis=0)
    rows['geometry']=np.tile([.1,.2,.5,.8],(4,1)).astype(np.float32)
    rows['quality']=np.ones(4,np.float32)
    rows['frame_index']=np.array([1,5,1,5],np.int64)
    _,_,summary,_=fit(data(rows),1,coefficient=.25)
    a=summary['loss_curves']['baseline'][0]; b=summary['loss_curves']['Known_set_rejection'][0]
    assert a['shared_physical_calendar_sha256']==b['shared_physical_calendar_sha256']
    assert a['rejection_parameter_gradient_norm']==0 and b['rejection_parameter_gradient_norm']>0
    assert summary['endpoint_parameter_sha256']['baseline']!=summary['endpoint_parameter_sha256']['Known_set_rejection']


def test_new_driver_scope_and_qualified_dependency_closure():
    import scripts.trackocd_cvpr.pilot_spread_known_rejection as driver
    root=Path(__file__).resolve().parents[2]
    for path in driver.SOURCES: assert (root/path).is_file(),path
    assert driver.OUT!=__import__('scripts.trackocd_cvpr.pilot_paired_known_rejection',fromlist=['OUT']).OUT
    source=(root/driver.SCRIPT).read_text()
    assert 'frozen_calendar=frozen' in source and 'private_frozen_calendar.json' in source
    assert 'old_pilot_result_sha256' in source and 'opportunity_qualification_sha256' in source
