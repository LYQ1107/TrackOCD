"""New full-development source protocol; only public qualification here."""
import json
from .spread_C_complete_fit_cohort import verified_cohort

CONFIG = 'configs/trackocd_cvpr/C_spread_known_full_development_v1.json'
PREFIX = 'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_full_causal_dev_v1'
EXPECTED = {
    'status':'REGISTERED_READONLY_FULL_CAUSAL_DEV_ALL6_SPREAD_FIXED200_ENDPOINTS_REQUIRED',
    'base_config':'configs/trackocd_cvpr/C_GT_causal_development_v1.json',
    'trained_arms':['baseline', 'Known_set_rejection'], 'training_seeds':[1027, 1028, 1029],
    'checkpoint_step':200, 'learned_numeric_interface':'Top8_conditioned', 'order_seeds':[1027, 1028, 1029],
    'physical_tracks':472, 'metadata_frames':2562, 'empty_frames':116, 'observations':8753,
    'prefix_snapshots':[1, 2, 4, 8, 16], 'execution_device':'CPU', 'CPU_compute_workers':1,
    'rss_bytes_ceiling':3221225472, 'wall_seconds_ceiling':3000, 'output_bytes_ceiling':16777216,
    'output_prefix':PREFIX, 'optimizer_steps':0, 'select_model_or_checkpoint':False,
    'one_explicit_arm_seed_per_worker':True, 'require_all6_private_endpoint_fingerprints_before_dev_input':True,
    'require_nonvacuous_ID_reverse_and_future_prefix_probes':True,
    'parallel_owned_research_worker_or_automatic_next_trial':False,
    'maximum_actual_evaluation_attempts_per_arm_seed':1,
    'calibrated_Raw_or_coverage_latency_match_claimed':False,
    'full_semantic_truth_Hscore_CT_joint_Pred_or_Goal_accepted':False,
    'final_Val_Test_or_original_pixels_or_new_nonKnown_labels':False,
}


def configuration(root, arm, seed):
    cfg = json.loads((root/CONFIG).read_text())
    if cfg != EXPECTED or arm not in EXPECTED['trained_arms'] or type(seed) is not int or seed not in EXPECTED['training_seeds']:
        raise ValueError('Exact all6 full-clock read-only protocol, no subset/selection/extra steps')
    verified_cohort(root)  # No private endpoint or dev inputs opened by configuration.
    base = json.loads((root/cfg['base_config']).read_text())
    for key in ('order_seeds', 'physical_tracks', 'metadata_frames', 'empty_frames', 'observations',
                'prefix_snapshots', 'execution_device', 'CPU_compute_workers', 'rss_bytes_ceiling',
                'wall_seconds_ceiling', 'output_bytes_ceiling', 'optimizer_steps', 'select_model_or_checkpoint'):
        if base[key] != cfg[key]:
            raise ValueError('Inherited qualified full causal clock/metric and guard protocol unchanged')
    if base['K_per_family']['Top8_conditioned'] != 8:
        raise ValueError('Both trained arms have unchanged actual Top8 numeric interface')
    return {**base, **cfg, 'arm':arm, 'seed':seed, 'output_directory':f'{PREFIX}_{arm}_s{seed}'}
