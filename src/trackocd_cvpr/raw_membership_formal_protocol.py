"""Public prerequisite/configuration for four explicit fresh formal controls."""
import json
from .raw_known_membership_protocol import source_window
from .raw_membership_pilot_qualification import verify_actual_pilot

CONFIG = 'configs/trackocd_cvpr/raw_known_membership_formal_execution_v1.json'
PREFIX = 'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_formal_v1'
EXPECTED = {
    'status':'REGISTERED_FRESH_RAW_MEMBERSHIP_FORMAL_CONTROLS_ACTUAL_PILOT_REQUIRED',
    'protocol':'configs/trackocd_cvpr/raw_known_membership_controls_v1.json',
    'calendar_seeds':[1027,1028,1029],'calendar_episodes':4,'calendar_updates_per_episode':50,
    'calendar_optimizer_steps':200,'calendar_checkpoint_steps':[20,50,100,200],
    'full_policy_independent_seed_runs':False,'full_policy_maxiter':200,'full_policy_maxfun':1000,
    'output_prefix':PREFIX,'controls':['calendar_matched','full_policy'],
    'one_explicit_control_seed_per_invocation':True,'maximum_actual_fit_attempts_per_control_seed':1,
    'old_or_pilot_weight_warmstart':False,'automatic_development_or_next_fit':False,
    'dev_selected_model_checkpoint_threshold_or_control':False,'calibration_quality_or_Goal_accepted_from_fit':False,
    'execution_device':'CPU','CPU_compute_workers':1,'rss_bytes_ceiling':3221225472,
    'wall_seconds_ceiling':3000,'output_bytes_ceiling':16777216}


def configuration(root,control,seed):
    cfg = json.loads((root/CONFIG).read_text())
    if (cfg!=EXPECTED or control not in EXPECTED['controls']
            or control=='calendar_matched' and (type(seed) is not int or seed not in (1027,1028,1029))
            or control=='full_policy' and seed is not None):
        raise ValueError('Exact four fresh controls; fullpolicy is ONE deterministic fit, not fake seeds')
    source_window(root); verify_actual_pilot(root)  # No private checkpoint or GT read.
    name = f'{PREFIX}_{control}'+(f'_s{seed}' if seed is not None else '')
    return {**cfg,'control':control,'seed':seed,'output_directory':name}
