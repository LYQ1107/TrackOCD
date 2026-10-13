"""Narrow PUBLIC readback qualification, never rewrite a failed supervisor.

Only complete C eval/exit0 with terminal-counter-only observation exception,
under unchanged resource limits. Not a generic guard bypass, rerun, scientific
win or proof that every supervisor sample existed. Requires caller hashes and
fresh registered-source verification before creating a separate receipt.
"""
def qualified_completed_C(result, terminal, limits):
    if (result['status'] != 'MEASURED_CAUSAL_GT_C_DEVELOPMENT_NOT_M_PRED_FULL_TRUTH_OR_GOAL_ACCEPTANCE'
            or terminal['owned_worker_exit_code'] != 0
            or terminal['guard_stop_reason'] != 'owned_memory_counter_unavailable'
            or terminal['successful_owned_execution'] is not False
            or not terminal['only_new_owned_Popen_worker_managed']
            or result['optimizer_steps'] != 0 or result['model_or_checkpoint_selection']
            or result['final_Val_Test_or_new_nonKnown_semantic_values_read']
            or not result['actual_full_ID_rename_reverse_stream_equal']
            or not result['actual_future_probe']['verified_same_prior_events']
            or result['actual_future_probe']['future_observations_mutated'] < 1
            or set(result['orders']) != {'1027', '1028', '1029'}):
        raise ValueError('Only full completed exit0 C evaluation/counter-only exception')
    if (result['resources']['CUDA_initialized']
            or result['resources']['peak_guarded_RSS_bytes'] > limits['rss_bytes_ceiling']
            or terminal['peak_sampled_owned_worker_RSS_bytes'] > limits['rss_bytes_ceiling']
            or result['resources']['wall_seconds_excluding_network'] > limits['wall_seconds_ceiling']
            or terminal['wall_seconds_including_imports_and_network'] > limits['wall_seconds_ceiling']):
        raise ValueError('Real memory/time/CPU limits cannot be excused by exit0')
    for order in result['orders'].values():
        if (order['observations'], order['metadata_frames'], order['empty_frames']) != (8753, 2562, 116):
            raise ValueError('No incomplete/favorable scope')
        if not order['model_state_unchanged'] or order['optimizer_steps'] != 0:
            raise ValueError('Strict frozen evaluator')
        if set(order['prefix_snapshots']) != {'1', '2', '4', '8', '16', 'all_available'}:
            raise ValueError('All frozen prefix outcomes')
        for p in order['prefix_snapshots'].values():
            if (p['physical_tracks'], p['Known_queries'], p['declared_pseudo_queries']) != (472, 302, 119):
                raise ValueError('Complete fixed physical/query denominators')
    return {'status': 'PUBLIC_READBACK_QUALIFIED_COMPLETE_C_WORKER_WITH_COUNTER_OBSERVATION_EXCEPTION',
        'effective_completed_worker_qualification': True,
        'original_supervisor_successful_owned_execution': False,
        'original_guard_stop_reason': terminal['guard_stop_reason'],
        'own_Popen_wait_exit_code': terminal['owned_worker_exit_code'],
        'complete_worker_result_source_scope_frozen_state_and_actual_invariance_checked': True,
        'all_supervisor_memory_samples_certified': False,
        'original_failed_flag_rewritten_or_rerun_or_scientific_gain': False}
