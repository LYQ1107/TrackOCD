"""Public original full-development trial qualification and ALL6-only summary.

No private labels/checkpoints, fitting, live registration bypass, calibration
or final model selection. A single qualified result is never the full matrix.
"""
import json
from src.trackocd_v2.io import sha256_file
from .spread_C_development_protocol import PREFIX, EXPECTED
from .spread_C_complete_fit_cohort import verified_cohort

ARMS = ('baseline', 'Known_set_rejection')
SEEDS = (1027, 1028, 1029)
PREFIXES = ('1', '2', '4', '8', '16', 'all_available')


def directory(arm, seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Explicit frozen arm and training seed')
    return f'{PREFIX}_{arm}_s{seed}'


def trial_sources(arm, seed):
    out = directory(arm, seed)
    return (out+'/launch.json', out+'/result.json', out+'_supervisor_attempt_1.json')


def verified_trial(root, arm, seed):
    from scripts.trackocd_cvpr.evaluate_spread_C_development import SOURCES
    paths = [root/p for p in trial_sources(arm, seed)]
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Complete original full-dev result/launch/normal terminal required')
    launch, r, t = (json.loads(p.read_text()) for p in paths)
    expected = next(x for x in verified_cohort(root)['trials'] if x['arm'] == arm and x['seed'] == seed)
    if (r['status'] != 'MEASURED_FULL_CAUSAL_SPREAD_PAIRED_C_DEV_NOT_CALIBRATED_JOINT_PRED_FINAL_OR_GOAL'
            or r['arm'] != arm or r['training_seed'] != seed or not r['all6_fixed200_endpoint_fingerprints_checked']
            or r['evaluated_endpoint'] != expected['endpoint']
            or r['evaluated_model_parameter_sha256'] != expected['endpoint_parameter_sha256']
            or r['optimizer_steps'] != 0 or r['model_or_checkpoint_selection']
            or r['final_Val_Test_or_new_nonKnown_semantic_values_read']
            or r['calibrated_Raw_or_coverage_latency_match_claimed']
            or r['full_semantic_truth_Hscore_CT_joint_Pred_or_Goal_accepted']
            or not r['actual_full_ID_rename_reverse_stream_equal']
            or not r['actual_future_probe']['verified_same_prior_events']
            or r['actual_future_probe']['future_observations_mutated'] < 1
            or set(r['orders']) != set(map(str, SEEDS))
            or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0
            or t['guard_stop_reason'] is not None or not t['complete_result_present']
            or t['CPU_compute_workers'] != 1 or t['peak_sampled_worker_threads'] > 2
            or t['peak_identified_native_driver_helpers'] > 1 or t['training_GPU_hours'] != 0
            or r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours'] != 0
            or r['resources']['peak_guarded_RSS_bytes'] > EXPECTED['rss_bytes_ceiling']
            or t['peak_sampled_owned_worker_RSS_bytes'] > EXPECTED['rss_bytes_ceiling']
            or r['resources']['wall_seconds_excluding_network'] > EXPECTED['wall_seconds_ceiling']
            or t['wall_seconds_including_worker_imports_network'] > EXPECTED['wall_seconds_ceiling']
            or launch['arm'] != arm or launch['seed'] != seed or launch['endpoint'] != expected['endpoint']
            or launch['preregistration_commit'] != r['live_remote_registration_proof']['sha']
            or launch['live_remote_registration_proof'] != r['live_remote_registration_proof']):
        raise ValueError('Actual complete causal frozen endpoint/full probes and normal bounded terminal')
    if set(launch['source_sha256']) != set(SOURCES):
        raise ValueError('Every original actual evaluation dependency required')
    for p, expected_hash in launch['source_sha256'].items():
        if (root/p).is_symlink() or sha256_file(root/p) != expected_hash:
            raise ValueError('Qualified causal numeric/evaluation source changed')
    for order, row in r['orders'].items():
        if ((row['observations'], row['metadata_frames'], row['empty_frames']) != (8753, 2562, 116)
                or row['order_seed'] != int(order) or not row['model_state_unchanged']
                or row['optimizer_steps'] != 0 or not row['GT_oracle_control_not_Pred']
                or row['unknown_declared_background_or_Novel']
                or set(row['prefix_snapshots']) != set(PREFIXES)
                or sum(row['physical_actions'].values()) != 8753):
            raise ValueError('Every complete physical dev clock, fixed model and declared order')
        for end in row['prefix_snapshots'].values():
            if (end['physical_tracks'], end['Known_queries'], end['declared_pseudo_queries']) != (472, 302, 119):
                raise ValueError('Fixed complete physical and semantic denominator, no unknown drop')
    return r


def verified_matrix(root):
    table = {(a, s):verified_trial(root, a, s) for a in ARMS for s in SEEDS}
    reference = table[(ARMS[0], SEEDS[0])]
    for r in table.values():
        for order in map(str, SEEDS):
            if (r['orders'][order]['private_video_ordinal_order_sha256'] !=
                    reference['orders'][order]['private_video_ordinal_order_sha256']):
                raise ValueError('Same actual frozen video ordinal permutation per order across ALL6')
    return table


def aggregate_matrix(table):
    if set(table) != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError('ALL6 actual trial matrix only; no partial winner or selected seed/order')
    fields = ('Known_micro_accuracy', 'Known_macro_accuracy', 'all_physical_commit_track_coverage',
        'declared_pseudo_absorbed_as_Known_queries', 'declared_pseudo_anonymous_commit_queries',
        'certified_cross_video_reuse_micro_lower_bound', 'certified_cross_video_reuse_macro_lower_bound')
    means = {}
    for arm in ARMS:
        seeds = {}
        for seed in SEEDS:
            r = table[(arm, seed)]
            seeds[str(seed)] = {p:{key:sum(r['orders'][str(o)]['prefix_snapshots'][p][key] for o in SEEDS)/3
                                  for key in fields} for p in PREFIXES}
        means[arm] = {'per_training_seed_order_mean':seeds,
            'three_training_seed_mean':{p:{key:sum(seeds[str(s)][p][key] for s in SEEDS)/3
                                           for key in fields} for p in PREFIXES}}
    return {'arms':means, 'selected_model_seed_order_or_prefix':False,
        'order_replays_are_independent_training_seeds':False,
        'calibrated_Raw_coverage_latency_Pred_joint_final_or_Goal_accepted':False}
