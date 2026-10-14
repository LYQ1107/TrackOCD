"""Explicit endpoint source-compatibility gate; never a provenance bypass.

The nine fixed M endpoints were trained with the pre-3583d6a source.  The
current source differs only in the complete-support input bounds.  This gate
accepts that single declared receipt difference for a deployment-trace audit,
while retaining every endpoint/result/terminal/weight check.  It does not
claim current-source training provenance or qualify full M development.
"""
import hashlib
import json
import subprocess

from src.trackocd_v2.io import sha256_file
from .memory_transport_recovery import cohort_directory, verify_original_preinput_failure
from .M_development_cohort import VARIANTS, SEEDS, cohort_sources

CURRENT_SOURCES = {
    'scripts/trackocd_cvpr/pilot_memory_optimizer.py': '94737daa6eeff781cba52f8a39744b1e48103849b39ef63610f8cdf9962d63e6',
    'scripts/trackocd_cvpr/qualify_policy_training_input.py': '5df3982d129654a6b55de1e51b65c5212890cac30b0b5210707b1659c45470ef',
    'src/trackocd_cvpr/member_reliability.py': '09dc1d566357063dd2f9fb23001c41b0bb161eab64d2cb89c02a6d349952f0f2',
    'src/trackocd_cvpr/prefix_member_bank.py': 'a454f7192bacd5d4762d468b46a08afd6b2f94d27ea755e5b1ccd56b81a03e71',
    'src/trackocd_cvpr/versioned_reliability_memory.py': 'ec593d0cff68db835a86806b12018d2dd7fd02aa017fcbc57c024de0eaf35e3c',
    'src/trackocd_cvpr/memory_transport_recovery.py': '603128b8bb1652cc47d89574a721d7ab89501640f1a70f089971026e82c164a8',
}
# The table is expanded below with exact historical receipt hashes.  Keeping
# it code-local makes an unknown receipt mismatch fail closed before caches.
ALLOWED_RECEIPT_SOURCES = {
    'scripts/trackocd_cvpr/pilot_memory_optimizer.py': {
        '6210712f0e2262117e62423fd463792f08ae3948a27bdaa16e0eb63e4acf0dca',
        '0cd6cd3b5fda786ff49c24c927444511ce1bf40511b42da16613b09e862399bf',
        '614e3bcae1caeb71fada4ae1f71ccee628a630eec5caa2c337ec323d5bea4b4b',
    },
    'scripts/trackocd_cvpr/qualify_policy_training_input.py': {
        'a79eeb87a384b3b4d80785d6977a041953126569e528e060efc45d9d35d49a73',
    },
    'src/trackocd_cvpr/member_reliability.py': {
        '1ec246809fe85c25209dd175953889339e55ca412a3b767347f5d89d0ac652b8',
    },
    'src/trackocd_cvpr/prefix_member_bank.py': {
        'f2d40b4062ac3e13950654ad467a6ebb7f8dbbf0aeedecc14f2eba7240f007d3',
    },
    'src/trackocd_cvpr/versioned_reliability_memory.py': {
        'd4dc6f94698a93dc578733b3b048c6ebf692aa9abbcb701e788bae40eed57e4a',
    },
    'src/trackocd_cvpr/memory_transport_recovery.py': {
        'e8b7c204000a5423637a60d6c39ebc4984d2297fcde4caf5e8f903907632287c',
    },
}


def verified_cohort(root):
    results, initial = {}, {}
    for path in cohort_sources():
        candidate = root / path
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.is_symlink():
            raise ValueError('Public cohort records must be regular local files')
    verify_original_preinput_failure(root)
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
            result = json.loads((out / 'result.json').read_text())
            terminal = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
            if (variant, seed) == ('relative_weighted', 1029):
                from .M_terminal_replica_recovery import verify_replica_result
                if result.get('execution_transport_replica') != verify_replica_result(root, result):
                    raise ValueError('Exact terminal completion replica required')
            if (result['variant'] != variant or result['seed'] != seed
                    or result['summary']['optimizer_steps'] != 200
                    or result['summary']['trainable_parameters'] != 161
                    or result['status'] != 'COMPLETED_FIXED_M_TRIAL_NOT_DEV_PRED_JOINT_OR_GOAL_ACCEPTANCE'
                    or result['checkpoint_selection'] != 'Fixed200 endpoint'
                    or not result['endpoint']['reopen_exact']
                    or result['old_R_C_M_or_pilot_warmstart']
                    or result['dev_final_Val_Test_or_new_nonKnown_labels_used']
                    or not terminal['successful_owned_execution']
                    or terminal['owned_worker_exit_code'] != 0
                    or terminal['guard_stop_reason'] is not None
                    or terminal['last_persisted_completed_optimizer_steps'] != 200):
                raise ValueError('All9 actual fixed M endpoint/result/terminal checks required')
            if len(result['summary']['episodes']) != 4 or any(
                    (episode['videos'], episode['metadata_frames'], episode['physical_observations'],
                     episode['empty_frames'], episode['optimizer_steps']) != (10, 374, 1268, 1, 50)
                    for episode in result['summary']['episodes']):
                raise ValueError('Complete four-episode endpoint scope required')
            launch = json.loads((out / 'launch.json').read_text())
            mismatches = []
            for path, expected in launch['source_sha256'].items():
                current = sha256_file(root / path)
                if current != expected:
                    mismatches.append((path, expected, current))
            for path, expected, current in mismatches:
                if (path not in ALLOWED_RECEIPT_SOURCES
                        or expected not in ALLOWED_RECEIPT_SOURCES[path]
                        or current != CURRENT_SOURCES[path]):
                    raise ValueError('Unknown endpoint receipt source mismatch: ' + path)
            if any(sha256_file(root / path) != digest for path, digest in CURRENT_SOURCES.items()):
                raise ValueError('Current source compatibility table does not match worktree')
            endpoint = out / 'private_step200_checkpoint.pt'
            if endpoint.is_symlink() or endpoint.stat().st_size != result['endpoint']['bytes'] or sha256_file(endpoint) != result['endpoint']['sha256']:
                raise ValueError('Actual fixed endpoint fingerprint required')
            initial_seed = result['summary']['model_parameter_sha256_before']
            if seed in initial and initial[seed] != initial_seed:
                raise ValueError('Matched fresh initialization per seed')
            initial[seed] = initial_seed
            results[(variant, seed)] = (endpoint, result)
    if len(set(initial.values())) != 3:
        raise ValueError('Three independent endpoint seeds required')
    return results
