"""Current-source full-M endpoint adapter.

Public cohort receipts are verified before opening the nine local fixed endpoint
checkpoints.  This adapter never substitutes the historical endpoint namespace.
"""
import json

from src.trackocd_v2.io import sha256_file
from .current_source_endpoint_cohort import configuration as public_configuration
from .current_source_endpoint_cohort import audit_cohort

CONFIG = 'configs/trackocd_cvpr/M_current_source_endpoint_cohort_audit_v1.json'


def cohort_sources(root):
    manifest, _, _ = public_configuration(root)
    paths = [CONFIG, 'outputs/trackocd_cvpr/cvpr4/M_current_source_endpoint_cohort_audit_v1/result.json']
    for case in manifest['cases']:
        namespace = case['output_directory']
        paths.extend((namespace + '/launch.json', namespace + '/progress.json',
                      namespace + '/result.json', namespace + '_supervisor_attempt_1.json'))
    return tuple(dict.fromkeys(paths))


def verified_cohort(root):
    public = audit_cohort(root)
    if (public['status'] != 'COHORT_AUDIT_PASS_CURRENT_SOURCE_ENDPOINT_PROVENANCE_ONLY'
            or public['case_count'] != 9
            or not public['all_current_source_hashes_match_registered_closure']):
        raise ValueError('Current-source public endpoint cohort audit is not a pass')
    manifest, _, _ = public_configuration(root)
    results = {}
    initial = {}
    for case in manifest['cases']:
        variant, seed, namespace = case['variant'], case['seed'], case['output_directory']
        result_path = root / (namespace + '/result.json')
        result = json.loads(result_path.read_text())
        checkpoint = root / (namespace + '/private_step200_checkpoint.pt')
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if (result.get('variant') != variant or result.get('seed') != seed
                or result['summary']['optimizer_steps'] != 200
                or result['summary']['trainable_parameters'] != 161
                or result['status'] != 'COMPLETED_FIXED_M_TRIAL_NOT_DEV_PRED_JOINT_OR_GOAL_ACCEPTANCE'
                or result['old_R_C_M_or_pilot_warmstart']
                or result['dev_final_Val_Test_or_new_nonKnown_labels_used']):
            raise ValueError('Current-source endpoint identity/protocol failed: ' + namespace)
        if len(result['summary']['episodes']) != 4 or any(
                (episode['videos'], episode['metadata_frames'], episode['physical_observations'],
                 episode['empty_frames'], episode['optimizer_steps']) != (10, 374, 1268, 1, 50)
                for episode in result['summary']['episodes']):
            raise ValueError('Current-source endpoint does not cover all four fixed episodes: ' + namespace)
        endpoint = result['endpoint']
        if checkpoint.stat().st_size != endpoint['bytes'] or sha256_file(checkpoint) != endpoint['sha256']:
            raise ValueError('Private endpoint fingerprint mismatch: ' + namespace)
        before = result['summary']['model_parameter_sha256_before']
        if seed in initial and initial[seed] != before:
            raise ValueError('Matched seed initialization differs across current-source variants')
        initial[seed] = before
        results[(variant, seed)] = (checkpoint, result)
    if set(results) != {(v, s) for v in ('relative_quarantine', 'relative_weighted', 'pooled_weighted') for s in (1027, 1028, 1029)}:
        raise ValueError('Exactly nine current-source endpoints are required')
    if len(set(initial.values())) != 3:
        raise ValueError('Three genuinely independent current-source training seeds required')
    return results
