"""Public all-four fit qualification before any full development private input."""
import json
from .raw_membership_formal_cohort import verified_cohort, TRIALS

CONFIG = 'configs/trackocd_cvpr/raw_known_membership_full_development_v1.json'
FIT_RECEIPT = 'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_all4_formal_fit_receipt_v1.json'
PREFIX = 'outputs/trackocd_cvpr/cvpr3/raw_NamedKnown_membership_full_causal_dev_v1'
EXPECTED = {
    'status': 'REGISTERED_READONLY_FULL_DEV_ALL4_FORMAL_RAW_MEMBERSHIP_ENDPOINTS_REQUIRED',
    'base_config': 'configs/trackocd_cvpr/C_GT_causal_development_v1.json',
    'controls': ['calendar_matched', 'full_policy'], 'calendar_seeds': [1027, 1028, 1029],
    'full_policy_independent_seed_runs': False, 'calendar_checkpoint_step': 200,
    'order_seeds': [1027, 1028, 1029], 'physical_tracks': 472, 'observations': 8753,
    'metadata_frames': 2562, 'empty_frames': 116, 'prefix_snapshots': [1, 2, 4, 8, 16],
    'execution_device': 'CPU', 'CPU_compute_workers': 1, 'rss_bytes_ceiling': 3221225472,
    'wall_seconds_ceiling': 3000, 'output_bytes_ceiling': 16777216, 'output_prefix': PREFIX,
    'optimizer_steps': 0, 'select_model_or_checkpoint': False,
    'fixed_existing_gate': 0.55, 'fixed_membership_logit_gate': 0.0,
    'conditional_reliability_bins': 10, 'all4_actual_fits_required_before_dev': True,
    'all13_private_snapshot_fingerprints_required_before_dev': True,
    'full3orders_and_nonvacuous_ID_reverse_future_probes_required': True,
    'one_explicit_control_seed_per_worker': True, 'maximum_attempts_per_control_seed': 1,
    'parallel_research_worker_or_auto_next_control': False,
    'semantic_denominator': 'Existing Known302/declared-pseudo119; outside2/unknown49 physically retained, not negatives. Conflicting aliases masked. Group-observation and track-prefix conditional reliability separately reported.',
    'full_semantic_truth_Hscore_CT_or_deployment_calibration_or_Goal_claimed': False,
    'foreground_existing_gate_or_fair_latency_coverage_matching_claimed': False,
    'final_Val_Test_original_pixels_new_nonKnown_or_Novel_vocabulary': False,
    'dev_selected_control_seed_threshold_bin_or_checkpoint': False,
}


def configuration(root, control, seed):
    cfg = json.loads((root/CONFIG).read_text())
    if cfg != EXPECTED or (control, seed) not in TRIALS or (seed is not None and type(seed) is not int):
        raise ValueError('Exact ALL4 full development; no subset, selection, artificial fullpolicy seeds')
    proof = verified_cohort(root)
    public = json.loads((root/FIT_RECEIPT).read_text())
    if any(public[k] != v for k, v in proof.items()):
        raise ValueError('Actual public ALL4 aggregate matches original complete formal fits')
    base = json.loads((root/cfg['base_config']).read_text())
    for key in ('order_seeds', 'physical_tracks', 'observations', 'metadata_frames', 'empty_frames',
                'prefix_snapshots', 'execution_device', 'CPU_compute_workers', 'rss_bytes_ceiling',
                'wall_seconds_ceiling', 'output_bytes_ceiling', 'optimizer_steps', 'select_model_or_checkpoint'):
        if base[key] != cfg[key]:
            raise ValueError('Original full causal clock/metric/guard unchanged')
    if base['K_per_family']['Top8_conditioned'] != 8 or base['raw_fixed_thresholds'] != {'tau_known': .65, 'tau_existing': .55}:
        raise ValueError('Same Top8 numeric kernel, original Raw and Existing gates')
    name = f'{PREFIX}_{control}' + (f'_s{seed}' if seed is not None else '')
    return {**base, **cfg, 'control': control, 'calendar_seed': seed, 'output_directory': name}
