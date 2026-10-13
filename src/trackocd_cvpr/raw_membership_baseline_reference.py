"""Historical PUBLIC Raw outcome reference, not a new private evaluation.

Exact original bytes/source commit, unchanged numeric kernels and actual clock
can support comparisons. Neither byte checks nor aggregate counts establish
deployment calibration, independence, fullsemantic purity or matched latency.
"""
import ast
import hashlib
import json
import subprocess
from src.trackocd_v2.io import sha256_file

DIRECTORY = 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Raw_Top8'
COMMIT = '88b5c56a814f2be9eaf7f760bcf172f7d248f3bd'
HASHES = {
    DIRECTORY+'/launch.json':'f9384670505e088c7bae018584bf956fb2e8731286b229f5d8af64f2968b742e',
    DIRECTORY+'/result.json':'e6fae9929d29358da18005fe52833c96de3a8abe61f50d8160665fcba9d99320',
    DIRECTORY+'_supervisor_attempt_1.json':'15d3a51b13b300c9c3e3ad94d42aac90f45fc96dc20f0af24bf9c9b091cde544',
}
CHANGED_SUPERVISION = {'scripts/trackocd_cvpr/evaluate_C_development.py',
                       'scripts/trackocd_cvpr/qualify_policy_training_input.py'}


def function_ast(source, name):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1: raise ValueError('Unique historical/current numerical worker body')
    return ast.dump(nodes[0], include_attributes=False)


def verified_raw_reference(root):
    for p, h in HASHES.items():
        if not (root/p).is_file() or (root/p).is_symlink() or sha256_file(root/p) != h:
            raise ValueError('Exact original public Raw triplet, not rewritten or selected result')
    launch = json.loads((root/(DIRECTORY+'/launch.json')).read_text())
    r = json.loads((root/(DIRECTORY+'/result.json')).read_text())
    t = json.loads((root/(DIRECTORY+'_supervisor_attempt_1.json')).read_text())
    if (r['status'] != 'MEASURED_CAUSAL_GT_C_DEVELOPMENT_NOT_M_PRED_FULL_TRUTH_OR_GOAL_ACCEPTANCE'
            or r['model'] != 'Raw_Top8' or r['training_seed'] is not None
            or r['optimizer_steps'] != 0 or r['model_or_checkpoint_selection']
            or r['final_Val_Test_or_new_nonKnown_semantic_values_read']
            or r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
            or not r['actual_full_ID_rename_reverse_stream_equal']
            or not r['actual_future_probe']['verified_same_prior_events']
            or r['actual_future_probe']['future_observations_mutated'] <= 0
            or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours'] != 0
            or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0
            or t['guard_stop_reason'] is not None or t['training_GPU_hours'] != 0
            or t['peak_sampled_worker_threads'] != 1 or t['peak_sampled_owned_worker_RSS_bytes'] > 3221225472
            or t['wall_seconds_including_imports_and_network'] > 3000
            or launch['live_remote_registration_proof'] != r['live_remote_registration_proof']
            or r['live_remote_registration_proof']['sha'] != COMMIT
            or set(r['orders']) != {'1027', '1028', '1029'}):
        raise ValueError('Genuine complete bounded zero-fit Raw reference/full probes, not fake seeds')
    required = {'src/trackocd_cvpr/C_development_rollout.py',
        'src/trackocd_cvpr/policy_training_data.py', 'src/trackocd_cvpr/atomic_member_control.py',
        'src/trackocd_cvpr/candidate_score_contract.py', 'src/trackocd_cvpr/causal_category_audit.py',
        'src/trackocd_cvpr/causal_observation_router.py', 'src/trackocd_cvpr/prefix_member_bank.py',
        'src/trackocd_cvpr/R_development_evaluation.py', 'src/trackocd_cvpr/fit_R_training.py',
        'src/trackocd_cvpr/raw_preserving_evidence.py', 'configs/trackocd_cvpr/C_GT_causal_development_v1.json',
        'configs/trackocd_cvpr/frozen_global_data_GT_control_v1.json', *CHANGED_SUPERVISION}
    if not required <= set(launch['source_sha256']):
        raise ValueError('Actual original full numeric/input/clock/guard source lineage')
    historical = {}; changed = []
    for p, h in launch['source_sha256'].items():
        original = subprocess.check_output(['git', 'show', COMMIT+':'+p], cwd=root)
        if hashlib.sha256(original).hexdigest() != h:
            raise ValueError('Recorded historical source binds exact original Git bytes')
        if (root/p).is_symlink() or sha256_file(root/p) != h:
            changed.append(p); historical[p] = original
    if set(changed) != CHANGED_SUPERVISION:
        raise ValueError('Only documented nonnumerical registration/supervision changes; no metric/kernel drift')
    driver = 'scripts/trackocd_cvpr/evaluate_C_development.py'
    if function_ast(historical[driver], 'worker') != function_ast((root/driver).read_bytes(), 'worker'):
        raise ValueError('Original/current full private-input/scoring/outcome/probe worker body must be identical')
    if r['development_scope']['physical_label_strata'] != {'Known':302, 'declared_pseudoNovel':119,
            'outside_eligible_Known':2, 'no_retained_semantic_truth':49}:
        raise ValueError('Complete existing target and unknown reference scope')
    for order, measured in r['orders'].items():
        if (measured['order_seed'] != int(order) or measured['observations'] != 8753
                or (measured['metadata_frames'], measured['empty_frames']) != (2562, 116)
                or measured['optimizer_steps'] != 0 or not measured['model_state_unchanged']
                or sum(measured['physical_actions'].values()) != 8753
                or set(measured['prefix_snapshots']) != {'1', '2', '4', '8', '16', 'all_available'}):
            raise ValueError('Every original complete annotated physical clock/order/prefix')
        for s in measured['prefix_snapshots'].values():
            if (s['physical_tracks'], s['Known_queries'], s['declared_pseudo_queries']) != (472, 302, 119):
                raise ValueError('Full physical/eligible semantic reference denominators')
    return {'status':'VERIFIED_HISTORICAL_PUBLIC_RAW_FULL_CAUSAL_REFERENCE_NOT_NEW_GT_OR_LATENCY_MATCH',
        'original_source_commit':COMMIT, 'original_triplet_sha256':HASHES,
        'original_current_numeric_worker_AST_equal':True,
        'all_other_actual_source_hashes_unchanged':True,
        'private_GT_weights_or_score_tables_loaded':False,
        'order_replays_are_independent_training_seeds':False,
        'calibration_fair_latency_coverage_winner_or_Goal_accepted':False}
