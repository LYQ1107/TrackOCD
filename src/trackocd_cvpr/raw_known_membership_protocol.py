"""Public source-only controls gate; grants no private asset access."""
import json
from src.trackocd_v2.io import sha256_file
from .spread_C_development_complete_matrix import verified_matrix,aggregate_matrix

CONFIG = 'configs/trackocd_cvpr/raw_known_membership_controls_v1.json'
MATRIX = 'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_all6_full_dev_matrix_v1.json'
MATRIX_SHA = 'f78a108a3a2d00dcf02c1b87b3af7f171e1a549e1fb7f0ae12c808b63719e5ad'


def source_window(root):
    cfg = json.loads((root/CONFIG).read_text())
    required = {'status':'SOURCE_ONLY_TWO_FIXED_RAW_KNOWN_MEMBERSHIP_CONTROLS_ACTUAL_PILOT_REQUIRED',
        'implementation':'src/trackocd_cvpr/raw_known_membership_control.py',
        'data_config':'configs/trackocd_cvpr/candidate_optimizer_pilot_v1.json',
        'authority':'configs/trackocd_cvpr/formal_research_scope_approval_v1.json',
        'completed_C_matrix':MATRIX,'completed_C_matrix_sha256':MATRIX_SHA,
        'target':'Membership in fixed fit5 named Known set: eligible Known=1, existing Known-derived policy2=0; mask unknown, outside-role and conflicting aliases. Not foreground/BG/novel category or individual anonymous-action truth',
        'model':'Two float64 parameters: a>=0,b; logit=a*current Raw max NamedKnown cosine+b; initial a=1,b=-0.65; sigmoid is a task score, not a proved deployment posterior',
        'Raw_prior':'Complete frozen Raw fit5/1158 supervised supports/257fit videos; no policy/dev/final prototype or selected R',
        'decision_rule':'Inherited Existing cosine>=0.55 and>=bestNamedKnown first; else bestKnown if fitted logit>=0; else NEW. Tie ambiguity WAIT via unchanged common resolver',
        'data_scope':{'policy_videos':10,'physical_tracks':54,'observations':1268,'metadata_frames':374,
                      'empty_frames':1,'Known_targets':34,'policy_pseudo_targets':10,'masked_tracks':10},
        'pilot':{'seed':1027,'episodes':1,'calendar_updates':50,'calendar_checkpoints':[50],
                 'full_policy_maxiter':50,'full_policy_maxfun':250,
                 'require_both_lawful_roles_actual_gradients_and_own_checkpoint_reopen':True,'formal_warmstart':False},
        'execution_device':'CPU','CPU_compute_workers':1,'rss_bytes_ceiling':3221225472,
        'wall_seconds_ceiling':3000,'output_bytes_ceiling':16777216,'maximum_actual_pilot_attempts':1}
    false_fields = ('Existing_gate_fitted_or_full_posterior_foreground_purity_calibration_claimed',
        'class_macro_conditional_task_is_deployment_prior','actual_calibration_quality_or_Goal_accepted_from_source',
        'coefficient_seed_threshold_optimizer_step_search_or_dev_selection',
        'dev_final_Val_Test_new_nonKnown_or_original_pixels_used_in_fit',
        'unknown_as_negative_BG_Novel_or_GT_history_edit','old_weights_or_pilot_warmstart',
        'automatic_private_load_training_development_or_expansion')
    if any(cfg.get(k)!=v for k,v in required.items()) or any(cfg.get(k) is not False for k in false_fields):
        raise ValueError('Explicit bounded source-only controls; no role/search/calibration or permission expansion')
    expected_controls = {
        'calendar_matched':{'seeds':[1027,1028,1029],'episodes':4,'updates_per_episode':50,
            'optimizer_steps':200,'checkpoint_steps':[20,50,100,200],'optimizer':'AdamW',
            'learning_rate':.001,'weight_decay':.0001,'gradient_clip':1.,
            'slope_projection':'a>=0 after each optimizer update',
            'objective':'Class-macro binary membership BCE; same physical Known-rank update calendar as C, NOT identical targets, gradient exposure, capacity or function class'},
        'full_policy':{'independent_seed_runs':False,'optimizer':'L-BFGS-B','maxiter':200,'maxfun':1000,
            'maxls':20,'maxcor':10,'ftol':1e-12,'gtol':1e-8,'slope_bound':'a>=0','ridge_to_initial':.0001,
            'objective':'One deterministic full lawful policy scalar table/class-macro BCE plus fixed anchored ridge; more exposure/different optimizer from C, not calendar-matched'}}
    if cfg['controls'] != expected_controls or (root/MATRIX).is_symlink() or sha256_file(root/MATRIX)!=MATRIX_SHA:
        raise ValueError('Fixed two controls and genuine complete ALL6 prerequisite, no inherited partial win')
    report = json.loads((root/MATRIX).read_text()); actual = aggregate_matrix(verified_matrix(root))
    if {k:report[k] for k in actual} != actual:
        raise ValueError('Actual full paired evidence before new calibration source window')
    return cfg
