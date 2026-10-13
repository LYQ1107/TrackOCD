"""Synthetic pair qualification only; no real labels/assets/weights loaded."""
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
from src.trackocd_cvpr.paired_known_rejection_training import paired_candidate_fit
from src.trackocd_cvpr.candidate_scientific_training import scientific_candidate_fit
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
from tests.trackocd_cvpr.test_policy_training_data import arrays
from src.trackocd_cvpr.paired_known_rejection_protocol import source_window, CONFIG


def inputs(rows=None):
    projection = SimpleNamespace(targets={(1,12):('Known',0),(1,99):('policy_pseudo',0)})
    known = {0:np.eye(1,768,0,dtype=np.float32)[0]}
    return [(1,arrays() if rows is None else rows)],projection,known


def pair(coefficient=.25, rows=None, checkpoints=None):
    checkpoint_rows = [] if checkpoints is None else checkpoints
    def save(arm,model,optimizer,step,*other): checkpoint_rows.append((arm,step,digest(model)))
    return paired_candidate_fit(*inputs(rows),1027,episodes=2,updates_per_episode=1,
        checkpoint_steps=[1,2],guard=lambda:None,progress=lambda r,c:None,checkpoint=save,
        repair_coefficient=coefficient)


def test_zero_zero_synthetic_pair_matches_original_single_arm_weights_and_private_histories():
    window = {'variants':{'Top8_conditioned':{'K_per_family':8}},'training':{
        'learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,'episodes':2,
        'updates_per_episode':1,'optimizer_steps':2,'checkpoint_steps':[1,2]}}
    old,_,old_summary,old_private = scientific_candidate_fit(*inputs(),window,'Top8_conditioned',1027,
        lambda:None,lambda r,c:None,lambda *args:None)
    models,_,summary,private = pair(0.)
    for name in summary['arms']:
        assert digest(models[name]) == digest(old)
        assert repr(private[name]) == repr(old_private)
        assert [r['loss'] for r in summary['loss_curves'][name]] == [r['loss'] for r in old_summary['loss_curve']]


def test_rejection_is_additive_legal_set_loss_shared_calendar_not_same_targets_or_calibration():
    saved = []; models,_,summary,private = pair(checkpoints=saved)
    assert summary['initialization_sha256']['baseline'] == summary['initialization_sha256']['Known_set_rejection']
    assert summary['endpoint_parameter_sha256']['baseline'] != summary['endpoint_parameter_sha256']['Known_set_rejection']
    assert summary['optimizer_steps_each'] == 2 and summary['trainable_parameters_each'] == 708
    assert len(saved) == 4 and len(private['baseline']) == len(private['Known_set_rejection']) == 2
    for left,right in zip(summary['loss_curves']['baseline'],summary['loss_curves']['Known_set_rejection']):
        assert left['shared_physical_calendar_sha256'] == right['shared_physical_calendar_sha256']
        assert left['rejection_coefficient'] == left['rejection_parameter_gradient_norm'] == 0
        assert right['rejection_coefficient'] == .25 and right['rejection_parameter_gradient_norm'] > 0
        assert right['loss'] == pytest.approx(right['base_loss']+.25*right['rejection_macro_loss'])
    for arm in summary['arms']:
        assert all(r['metadata_frames'] == 4 and r['physical_observations'] == 3 for r in summary['episodes'][arm])
    assert summary['same_update_frames_not_identical_endogenous_candidate_or_loss_opportunities']
    assert not summary['GT_repair_unknown_as_background_or_individual_pseudo_action_target_added']
    assert not summary['calibration_development_Pred_joint_or_Goal_accepted']


def test_future_change_does_not_rewrite_first_episode_prior_predictions():
    _,_,_,first = pair(); changed = arrays(); changed['visual'][1] = 0; changed['visual'][1,7] = 1
    _,_,_,second = pair(rows=changed)
    for name in first:
        assert repr(first[name][0][1][:3]) == repr(second[name][0][1][:3])


def test_unknown_only_query_cannot_create_weight_decay_steps_or_invented_target():
    videos,projection,known = inputs(); projection.targets = {}
    with pytest.raises(ValueError,match='Whole shared update budget unmet'):
        paired_candidate_fit(videos,projection,known,1027,episodes=1,updates_per_episode=1,
            checkpoint_steps=[1],guard=lambda:None,progress=lambda r,c:None,checkpoint=lambda *args:None)


def test_pseudo_only_frame_cannot_make_calendar_depend_on_repaired_history_or_extra_loss():
    videos,projection,known = inputs()
    projection.targets = {(1,12):('policy_pseudo',0),(1,99):('policy_pseudo',1)}
    with pytest.raises(ValueError,match='Whole shared update budget unmet'):
        paired_candidate_fit(videos,projection,known,1027,episodes=1,updates_per_episode=1,
            checkpoint_steps=[1],guard=lambda:None,progress=lambda r,c:None,checkpoint=lambda *args:None)


def test_frozen_window_is_not_training_authority_or_a_selected_scientific_repair():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root/'configs/trackocd_cvpr/C_paired_known_set_rejection_v1.json').read_text())
    assert cfg['repair_coefficient'] == .25 and cfg['training_seeds'] == [1027,1028,1029]
    assert cfg['training']['optimizer_steps_each'] == 200 and cfg['pilot']['optimizer_steps_each'] == 50
    assert cfg['Raw_calibration_and_coverage_latency_matched_controls_still_required']
    assert not cfg['automatic_training_launch'] and not cfg['formal_expansion_before_successful_pilot']
    assert not cfg['private_input_reads_training_development_or_Goal_acceptance_from_source']


def test_public_source_window_requires_real_complete_old_C_matrix_and_audit():
    root = Path(__file__).resolve().parents[2]
    assert source_window(root)['repair_coefficient'] == .25


@pytest.mark.parametrize('field,value', [('repair_coefficient',1.),('training_seeds',[1027]),
    ('formal_expansion_before_successful_pilot',True),('calibrated_Raw_thresholds_or_posterior_claimed',True),
    ('wall_seconds_ceiling',6000),('automatic_training_launch',True)])
def test_source_window_cannot_relax_frozen_scientific_or_safety_scope(tmp_path,field,value):
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root/CONFIG).read_text()); cfg[field] = value
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): source_window(tmp_path)
