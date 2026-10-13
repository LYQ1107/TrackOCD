"""Read-only whole M cohort endpoint gate; never select from partial fits."""
import hashlib
import json
import subprocess
from src.trackocd_v2.io import sha256_file
from .memory_transport_recovery import cohort_directory, verify_original_preinput_failure

VARIANTS = ('relative_quarantine', 'relative_weighted', 'pooled_weighted')
SEEDS = (1027, 1028, 1029)


def cohort_sources():
    return tuple(path for v in VARIANTS for s in SEEDS for path in (
        cohort_directory(v, s) + '/launch.json',
        cohort_directory(v, s) + '/result.json',
        cohort_directory(v, s) + '_supervisor_attempt_1.json'))


def verified_cohort(root):
    results, initial = {}, {}
    # Establish the complete public cohort before opening any private endpoint.
    # Missing later trials must not qualify a selected successful subset.
    for path in cohort_sources():
        candidate = root / path
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.is_symlink():
            raise ValueError('Public cohort records must be regular local files')
    verify_original_preinput_failure(root)
    # Original M1 used recursive copying; later copy-only optimization is
    # declared, not falsely identical runtime source. Numeric kernels below
    # must still match; actual M1/M2 summary/history parity is separately saved.
    copy_paths = ('src/trackocd_cvpr/prefix_member_bank.py', 'src/trackocd_cvpr/versioned_reliability_memory.py')
    old_commit = 'e9c894ed0b3ef70b3fc1bc9651893de7a38b3030'
    optimized_commit = '88b5c56a814f2be9eaf7f760bcf172f7d248f3bd'
    allowed = {p: {hashlib.sha256(subprocess.check_output(['git', 'show', c + ':' + p], cwd=root)).hexdigest()
                   for c in (old_commit, optimized_commit)} for p in copy_paths}
    if any(sha256_file(root / p) not in allowed[p] for p in copy_paths):
        raise ValueError('Current runtime must be one exact pinned declared copy implementation')
    for variant in VARIANTS:
        for seed in SEEDS:
            out = root / cohort_directory(variant, seed)
            r = json.loads((out / 'result.json').read_text())
            t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
            if (r['variant'] != variant or r['seed'] != seed or r['summary']['optimizer_steps'] != 200
                    or r['summary']['trainable_parameters'] != 161
                    or r['status'] != 'COMPLETED_FIXED_M_TRIAL_NOT_DEV_PRED_JOINT_OR_GOAL_ACCEPTANCE'
                    or r['checkpoint_selection'] != 'Fixed200 endpoint' or not r['endpoint']['reopen_exact']
                    or r['old_R_C_M_or_pilot_warmstart'] or r['dev_final_Val_Test_or_new_nonKnown_labels_used']
                    or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0
                    or t['guard_stop_reason'] is not None or t['last_persisted_completed_optimizer_steps'] != 200):
                raise ValueError('All9 actual fresh fixed M endpoints and normal owned terminals required')
            if len(r['summary']['episodes']) != 4 or any((e['videos'], e['metadata_frames'], e['physical_observations'],
                e['empty_frames'], e['optimizer_steps']) != (10, 374, 1268, 1, 50) for e in r['summary']['episodes']):
                raise ValueError('All complete episodes, not a selected successful subset')
            launch = json.loads((out / 'launch.json').read_text())
            for p in ('configs/trackocd_cvpr/M_GT_scientific_window_v1.json', 'src/trackocd_cvpr/memory_scientific_training.py',
                      'src/trackocd_cvpr/member_reliability.py', 'src/trackocd_cvpr/reliability_training_loss.py',
                      'src/trackocd_cvpr/reliability_member_control.py'):
                if launch['source_sha256'][p] != sha256_file(root / p): raise ValueError('Same actual M window/kernel/loss/features/control')
            for p in copy_paths:
                if launch['source_sha256'][p] not in allowed[p]: raise ValueError('Only declared equivalent copy implementations')
            path = out / 'private_step200_checkpoint.pt'
            if path.is_symlink() or path.stat().st_size != r['endpoint']['bytes'] or sha256_file(path) != r['endpoint']['sha256']:
                raise ValueError('Actual fixed private endpoint fingerprint before GT projection')
            before = r['summary']['model_parameter_sha256_before']
            if seed in initial and initial[seed] != before: raise ValueError('Matched fresh initialization per seed')
            initial[seed] = before; results[(variant, seed)] = (path, r)
    if len(set(initial.values())) != 3: raise ValueError('Three genuinely independent training seeds')
    return results
