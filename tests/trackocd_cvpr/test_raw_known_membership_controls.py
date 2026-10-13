import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.raw_known_membership_control import (
    NamedKnownLogit,calibrated_raw_scores,collect_full_policy_samples,objective_and_gradient,
    full_policy_fit,calendar_matched_fit,group_label,membership_diagnostics,INITIAL,RIDGE)
from src.trackocd_cvpr.raw_known_membership_protocol import source_window,CONFIG
from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
from src.trackocd_cvpr.C_development_rollout import raw_fixed_scores

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    n = 50; visual = np.zeros((3*n,768),np.float16)
    visual[:n,0] = 1; visual[n:2*n,1] = 1; visual[2*n:,2] = 1
    arrays = {'track_id':np.array([10,20,30],np.int64),'offsets':np.array([0,n,2*n,3*n],np.int64),
        'visual':visual,'geometry':np.tile([.1,.1,.5,.5],(3*n,1)).astype(np.float32),
        'quality':np.ones(3*n,np.float32),'frame_index':np.tile(np.arange(1,n+1),3).astype(np.int64),
        'clock_frame_index':np.arange(n+1,dtype=np.int64)}
    projection = SimpleNamespace(targets={(1,10):('Known',0),(1,20):('policy_pseudo',0)})
    known = {0:np.eye(1,768,0,dtype=np.float32)[0]}
    return [(1,arrays)],projection,known


def fit(inputs,plan=None):
    plan = freeze_calendar(*inputs[:2],1027,1,50,lambda:None) if plan is None else plan
    return calendar_matched_fit(*inputs,1027,frozen_calendar=plan,episodes=1,updates_per_episode=50,
        checkpoint_steps=[50],guard=lambda:None,progress=lambda row:None,checkpoint=lambda *args:None)


def test_initial_monotone_map_exactly_preserves_legacy_raw_including_float32_threshold_and_ties():
    model = NamedKnownLogit().eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == 2
    values = [-1.,.1,.55,np.nextafter(np.float32(.65),np.float32(-1)),np.float32(.65),
              np.nextafter(np.float32(.65),np.float32(1)),.7,1.]
    for bk in values:
        for be in values:
            features = np.zeros((2,11),np.float32); features[:,0] = [bk,be]; features[0,1] = 1
            assert np.array_equal(calibrated_raw_scores(features,np.zeros(6),model=model),
                                  raw_fixed_scores(features,np.zeros(6),tau_known=.65,tau_existing=.55))
    with pytest.raises(ValueError,match='Frozen'):
        calibrated_raw_scores(features,np.zeros(6),model=NamedKnownLogit())
    with pytest.raises(ValueError,match='Finite'):
        model(float('nan'))


def test_full_collection_retains_unknown_physical_denominators_but_never_negative_BG_or_novel_truth():
    inputs = fixture(); private,proof = collect_full_policy_samples(*inputs,lambda:None)
    assert private.shape == (100,3) and np.isclose(private[:,2].sum(),1)
    assert proof['physical_observations'] == 150 and proof['metadata_frames'] == 51 and proof['empty_frames'] == 1
    assert proof['Known_groups'] == proof['policy_pseudo_groups'] == proof['masked_groups'] == 50
    assert not proof['unknown_as_negative_BG_Novel_or_individual_anonymous_target']
    projection = SimpleNamespace(targets={(1,'opaque-a'):('Known',0),(1,'opaque-b'):('Known',1)})
    assert group_label(projection,1,('opaque-a',)) == ('Known',0)
    assert group_label(projection,1,('opaque-a','opaque-b')) is None
    assert group_label(projection,1,('unknown',)) is None


def test_scalar_analytic_gradient_matches_autograd_and_finite_differences_not_opaque_solver_claim():
    private,_ = collect_full_policy_samples(*fixture(),lambda:None); theta = np.array([1.2,-.7])
    loss,gradient = objective_and_gradient(theta,private)
    p = torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    x,y,w = torch.tensor(private.T,dtype=torch.float64)
    objective = (w*torch.nn.functional.binary_cross_entropy_with_logits(p[0]*x+p[1],y,reduction='none')).sum()
    objective = objective + .5*RIDGE*((p-torch.tensor(INITIAL))**2).sum(); objective.backward()
    assert loss == pytest.approx(float(objective.detach()),abs=1e-14)
    assert np.allclose(gradient,p.grad.numpy(),rtol=0,atol=1e-14)
    eps = 1e-6
    for i in range(2):
        delta = np.eye(2)[i]*eps
        assert gradient[i] == pytest.approx((objective_and_gradient(theta+delta,private)[0]-objective_and_gradient(theta-delta,private)[0])/(2*eps),abs=1e-9)


def test_full_policy_solver_is_one_deterministic_fit_records_actual_success_and_no_fake_seeds():
    private,_ = collect_full_policy_samples(*fixture(),lambda:None)
    model,r = full_policy_fit(private,iteration_budget=50,function_budget=250,guard=lambda:None)
    assert r['solver_success'] and r['iterations'] <= 50 and r['function_evaluations'] <= 250
    assert r['final_regularized_class_macro_NLL'] < objective_and_gradient(INITIAL,private)[0]
    assert model.affine[0] >= 0 and not r['training_seeds_or_calendar_matched_to_C']
    assert not r['calibration_deployment_posterior_foreground_or_Goal_accepted']
    with pytest.raises(ValueError,match='fixed pilot/formal'):
        full_policy_fit(private,iteration_budget=51,function_budget=250,guard=lambda:None)


def test_fixed_proper_score_diagnostics_conditional_on_lawful_roles_not_full_GT_calibration():
    private,proof = collect_full_policy_samples(*fixture(),lambda:None)
    model = NamedKnownLogit().eval().requires_grad_(False); report = membership_diagnostics(model,private)
    assert report['lawful_groups'] == 100 and proof['physical_observations'] == 150
    assert report['Known_groups'] == report['policy_pseudo_groups'] == 50
    assert len(report['bins']) == 10 and sum(b['lawful_group_count'] for b in report['bins']) == 100
    p = torch.sigmoid(model(torch.tensor(private[:,0]))).detach().numpy()
    assert report['micro_Brier'] == pytest.approx(float(((p-private[:,1])**2).mean()))
    assert report['micro_NLL'] == pytest.approx(float((-private[:,1]*np.log(p)-(1-private[:,1])*np.log1p(-p)).mean()))
    assert not report['calibration_quality_proved_or_model_selected_from_diagnostics']
    assert report['declared_role_conditioning_and_class_weighting_not_full_deployment_prior']


def test_calendar_control_uses_all_real_50_steps_full_clock_and_both_roles_not_unknown_negatives():
    inputs = fixture(); _,_,summary,_ = fit(inputs)
    assert summary['optimizer_steps'] == 50 and len(summary['loss_curve']) == 50
    assert summary['episodes'][0]['physical_observations'] == 150
    assert summary['episodes'][0]['masked_groups'] == 50
    assert all(r['lawful_group_counts'] == {'Known':1,'policy_pseudo':1} for r in summary['loss_curve'])
    assert summary['same_frames_not_same_target_information_capacity_or_optimizer_function_class']
    assert not summary['anonymous_action_targets_GT_memory_edits_or_unknown_negative_added']


def test_frozen_plan_future_change_does_not_reselect_calendar_or_change_prior_numeric_logits():
    inputs = fixture(); plan = freeze_calendar(*inputs[:2],1027,1,50,lambda:None)
    _,_,first,a = fit(inputs,plan)
    changed = fixture(); changed[0][0][1]['visual'][49] = 0; changed[0][0][1]['visual'][49,7] = 1
    _,_,second,b = fit(changed,plan)
    assert a[:-1] == b[:-1]
    assert first['actual_shared_calendar_sha256'] == second['actual_shared_calendar_sha256']


def test_actual_full_C_matrix_source_prerequisite_grants_no_private_load_or_calibration_quality():
    cfg = source_window(ROOT)
    assert set(cfg['controls']) == {'calendar_matched','full_policy'}
    assert not cfg['automatic_private_load_training_development_or_expansion']
    assert not cfg['Existing_gate_fitted_or_full_posterior_foreground_purity_calibration_claimed']


@pytest.mark.parametrize('field,value',[
    ('unknown_as_negative_BG_Novel_or_GT_history_edit',True),
    ('dev_final_Val_Test_new_nonKnown_or_original_pixels_used_in_fit',True),
    ('actual_calibration_quality_or_Goal_accepted_from_source',True),
    ('old_weights_or_pilot_warmstart',True),
])
def test_scope_probability_and_authority_expansion_refuses_before_any_private_data(tmp_path,field,value):
    cfg = json.loads((ROOT/CONFIG).read_text()); cfg[field] = value
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='Explicit bounded'):
        source_window(tmp_path)
