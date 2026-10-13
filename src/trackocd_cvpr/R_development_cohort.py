"""All12 real fixed endpoints required before any full-cohort dev GT read."""
import json
from src.trackocd_v2.io import sha256_file

VARIANTS = ('temporal_CE', 'timeless_CE', 'temporal_Raw_teacher', 'timeless_Raw_teacher')
SEEDS = (1027, 1028, 1029)


def cohort_sources():
    return tuple(path for variant in VARIANTS for seed in SEEDS for path in (
        f'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_{variant}_s{seed}/launch.json',
        f'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_{variant}_s{seed}/result.json',
        f'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_{variant}_s{seed}_supervisor_attempt_1.json'))


def verified_cohort(root):
    results = {}; starts = {}; footprints = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            out = root / f'outputs/trackocd_cvpr/cvpr2/R_GT_window_v1_{variant}_s{seed}'
            r = json.loads((out / 'result.json').read_text())
            t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
            if (r['variant'] != variant or r['seed'] != seed or r['completed_optimizer_steps'] != 200
                    or r['status'] != 'COMPLETED_FIXED_R_FIT_TRIAL_NOT_DEV_VALIDATED_OR_GOAL_ACCEPTED'
                    or r['checkpoint_selection'] != 'Fixed200 endpoint' or not r['endpoint']['reopen_exact']
                    or r['old_fitted_adapter_initialization'] or r['dev_final_Val_Test_targets_or_pixels_used']
                    or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0
                    or t['guard_stop_reason'] is not None or t['last_persisted_completed_optimizer_steps'] != 200
                    or len(r['loss_curve']) != 200 or r['resources']['CUDA_initialized']):
                raise ValueError('All12 genuinely trained fixed endpoints/owned terminals, no selection')
            launch = json.loads((out / 'launch.json').read_text())
            for path in ('configs/trackocd_cvpr/R_GT_scientific_window_v1.json', 'src/trackocd_cvpr/fit_R_training.py',
                         'src/trackocd_cvpr/batched_evidence.py', 'src/trackocd_cvpr/raw_preserving_evidence.py',
                         'src/trackocd_cvpr/fit_loss_supervision.py', 'src/trackocd_cvpr/exact_support_reduction.py'):
                if launch['source_sha256'][path] != sha256_file(root / path):
                    raise ValueError('Same actual frozen R window/kernel/loss/support math before dev labels')
            checkpoint = out / 'private_step200_checkpoint.pt'
            if (checkpoint.is_symlink() or checkpoint.stat().st_size != r['endpoint']['bytes']
                    or sha256_file(checkpoint) != r['endpoint']['sha256']):
                raise ValueError('Actual own private endpoint size/SHA before GT projection')
            start = r['model_parameter_sha256_before']
            footprint = [(x['support_members'], x['support_observations']) for x in r['loss_curve']]
            if seed in starts and (start != starts[seed] or footprint != footprints[seed]):
                raise ValueError('Perseed matched fresh initialization/query-support calendar required')
            starts[seed], footprints[seed] = start, footprint
            results[(variant, seed)] = (checkpoint, r)
    if len(set(starts.values())) != 3:
        raise ValueError('Independent training seeds, not reused fits')
    return results
