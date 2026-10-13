"""Public formal source gates; no private input access or automatic launch."""
import json
from .paired_known_rejection_protocol import source_window
from .spread_C_pilot_qualification import verify_informative_pilot

CONFIG = 'configs/trackocd_cvpr/C_spread_known_formal_fixed200_v1.json'
PREFIX = 'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_Fixed200_v1'
EXPECTED = {
    'status': 'REGISTERED_FRESH_SPREAD_PAIRED_FIXED200_FULL_POLICY_NO_DEV_AUTO_START',
    'training_seeds': [1027, 1028, 1029], 'arms': ['baseline', 'Known_set_rejection'],
    'episodes': 4, 'updates_per_episode_each': 50, 'optimizer_steps_each': 200,
    'checkpoint_steps': [20, 50, 100, 200], 'repair_coefficient': .25,
    'calendar_rule': 'MIDPOINT_UNIFORM_KNOWN_ELIGIBLE_RANKS_NO_PSEUDO_OR_PREDICTION_HISTORY_SELECTOR',
    'output_prefix': PREFIX, 'checkpoint_selection': 'Fixed200, no dev/final/Val/Test selection',
    'pilot_or_historical_checkpoint_warmstart': False, 'full_development_automatic_start': False,
    'formal6fits_development_calibration_or_Goal_accepted': False,
    'execution_device': 'CPU', 'CPU_compute_workers': 1, 'rss_bytes_ceiling': 3221225472,
    'wall_seconds_ceiling': 3000, 'output_bytes_ceiling': 16777216,
    'one_explicit_seed_pair_per_invocation': True, 'parallel_owned_research_training': False,
    'maximum_training_attempts_per_seed': 1,
    'new_nonKnown_labels_unknown_as_background_or_GT_history_repair': False,
}


def configuration(root, seed):
    cfg = json.loads((root / CONFIG).read_text())
    if cfg != EXPECTED or type(seed) is not int or seed not in EXPECTED['training_seeds']:
        raise ValueError('Exact fresh three-seed paired200 protocol; no search or pilot warmstart')
    source_window(root)
    verify_informative_pilot(root)  # Public fingerprints and actual nonzero gradient, not weights.
    return {**cfg, 'seed': seed, 'output_directory': f'{PREFIX}_s{seed}'}
