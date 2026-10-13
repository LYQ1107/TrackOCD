"""Actual six-fit public cohort checks; private fingerprint checks opt-in only.

This is not a live registration/asset permission gate. A later evaluator must
register these sources plus every cohort receipt, verify live remote/authority,
and then explicitly check private endpoint bytes before any dev input loads.
"""
import json
from src.trackocd_v2.io import sha256_file
from .spread_C_formal_protocol import PREFIX, EXPECTED

SEEDS = (1027, 1028, 1029)


def cohort_sources():
    return tuple(p for seed in SEEDS for p in (
        f'{PREFIX}_s{seed}/launch.json', f'{PREFIX}_s{seed}/result.json',
        f'{PREFIX}_s{seed}_supervisor_attempt_1.json'))


def verified_cohort(root, *, check_private_endpoints=False):
    from scripts.trackocd_cvpr.fit_spread_known_fixed200 import SOURCES
    rows = []; initializations = {}; calendars = {}; qualified_sources = None
    for seed in SEEDS:
        out = root/f'{PREFIX}_s{seed}'
        paths = (out/'launch.json', out/'result.json', out.with_name(out.name+'_supervisor_attempt_1.json'))
        if any(not p.is_file() or p.is_symlink() for p in paths):
            raise ValueError('All three normal complete seed pairs required, not partial metrics')
        launch, result, terminal = (json.loads(p.read_text()) for p in paths)
        s = result['summary']; checkpoints = result['private_checkpoints']
        if (result['status'] != 'COMPLETED_FRESH_SPREAD_PAIRED_FIXED200_NOT_DEV_CALIBRATION_OR_GOAL_ACCEPTANCE'
                or result['training_seed'] != seed or result['checkpoint_selection'] != EXPECTED['checkpoint_selection']
                or result['pilot_or_historical_checkpoint_warmstart'] or result['dev_final_Val_Test_inputs_used']
                or result['raw_private_history_or_GT_exported']
                or result['formal6fits_development_calibration_or_Goal_accepted']
                or result['resources']['CUDA_initialized'] or result['resources']['training_GPU_hours'] != 0
                or not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0
                or terminal['guard_stop_reason'] is not None or not terminal['complete_result_present']
                or terminal['CPU_compute_workers'] != 1 or terminal['training_GPU_hours'] != 0
                or terminal['peak_sampled_worker_threads'] > 2 or terminal['peak_identified_native_driver_helpers'] > 1
                or terminal['peak_sampled_owned_worker_RSS_bytes'] > EXPECTED['rss_bytes_ceiling']
                or terminal['wall_seconds_including_worker_imports_network'] > EXPECTED['wall_seconds_ceiling']
                or s['arms'] != EXPECTED['arms'] or s['optimizer_steps_each'] != 200
                or s['trainable_parameters_each'] != 708 or len(set(s['initialization_sha256'].values())) != 1
                or s['offline_eligible_Known_frames_each'] != [333]*4
                or s['offline_Known_uniform_rank_plan_rule'] != EXPECTED['calendar_rule']
                or not s['frozen_plan_used_before_updates_not_model_input']
                or s['GT_repair_unknown_as_background_or_individual_pseudo_action_target_added']
                or s['calibration_development_Pred_joint_or_Goal_accepted']
                or launch['preregistration_commit'] != result['live_remote_registration_proof']['sha']
                or launch['live_remote_registration_proof'] != result['live_remote_registration_proof']
                or len(checkpoints) != 8 or not all(c['reopen_exact'] for c in checkpoints)):
            raise ValueError('Actual fresh complete bounded paired200 cohort, no dev or scientific acceptance')
        if set(launch['source_sha256']) != set(SOURCES):
            raise ValueError('Complete actual qualified training source closure')
        if qualified_sources is None:
            qualified_sources = launch['source_sha256']
        if launch['source_sha256'] != qualified_sources:
            raise ValueError('All seed pairs use same qualified numerical/source path')
        initializations[seed] = next(iter(s['initialization_sha256'].values()))
        calendars[seed] = s['actual_shared_physical_calendar_sha256']
        for arm in s['arms']:
            curves = s['loss_curves'][arm]; episodes = s['episodes'][arm]
            if (s['endpoint_parameter_sha256'][arm] == s['initialization_sha256'][arm]
                    or [r['step'] for r in curves] != list(range(1, 201))
                    or len(episodes) != 4
                    or any((e['episode'], e['optimizer_steps'], e['videos'], e['metadata_frames'],
                            e['physical_observations'], e['empty_frames']) != (i, 50, 10, 374, 1268, 1)
                           for i, e in enumerate(episodes))
                    or any(r['rejection_coefficient'] != (.25 if arm == 'Known_set_rejection' else 0.)
                           or r['episode'] != (r['step']-1)//50
                           or r['episode_optimizer_step'] != (r['step']-1)%50+1 for r in curves)
                    or [r['shared_physical_calendar_sha256'] for r in curves] !=
                       [r['shared_physical_calendar_sha256'] for r in s['loss_curves']['baseline']]
                    or curves[-1]['shared_physical_calendar_sha256'] != calendars[seed]
                    or [c['step'] for c in checkpoints if c['arm'] == arm] != EXPECTED['checkpoint_steps']):
                raise ValueError('Every real step, full clocks, matched calendar and fixed snapshots')
            endpoint = next(c for c in checkpoints if c['arm'] == arm and c['step'] == 200)
            if check_private_endpoints:
                p = out/f'private_{arm}_step200.pt'
                if not p.is_file() or p.is_symlink() or p.stat().st_size != endpoint['bytes'] or sha256_file(p) != endpoint['sha256']:
                    raise ValueError('Actual frozen private Fixed200 endpoint fingerprint')
            rows.append({'seed':seed, 'arm':arm, 'result_sha256':sha256_file(paths[1]),
                'launch_sha256':sha256_file(paths[0]), 'terminal_sha256':sha256_file(paths[2]),
                'endpoint':endpoint, 'endpoint_parameter_sha256':s['endpoint_parameter_sha256'][arm],
                'actual_extra_gradient_steps':sum(r['rejection_parameter_gradient_norm'] > 0 for r in curves),
                'full_episode_actions':[e['physical_actions'] for e in episodes]})
    if len(set(initializations.values())) != 3 or len(set(calendars.values())) != 3:
        raise ValueError('Three distinct actual initializations and training calendars')
    for p, expected in qualified_sources.items():
        if (root/p).is_symlink() or sha256_file(root/p) != expected:
            raise ValueError('Qualified actual training dependencies changed')
    return {'all6_actual_fresh_fixed200_fits':True, 'same_seed_initialization_and_calendar_verified':True,
        'three_distinct_actual_seeds_verified':True, 'trials':rows,
        'private_endpoint_hashes_checked':check_private_endpoints,
        'development_calibration_Pred_joint_or_Goal_accepted':False}
