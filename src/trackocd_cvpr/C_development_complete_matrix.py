"""PUBLIC-only full original C development matrix; no selected seed/order.

Old explicit counter-observation exception and one pre-input namespace recovery
remain visible. No GT/cache/weight reads and no fitting. Public recorded source
and worker invariants establish the scope of this aggregate, not Pred/full CT.
"""
import json
from src.trackocd_v2.io import sha256_file
from .C_terminal_counter_qualification import qualified_completed_C

VARIANTS = ('Top1_conditioned', 'Top8_conditioned', 'Top8_pool_context')
SEEDS = (1027, 1028, 1029)
PREFIXES = ('1', '2', '4', '8', '16', 'all_available')
REQUALIFIED = 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top1_conditioned_s1027_terminal_requalification_v1.json'
FAILED_REGISTRATION = 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Top8_pool_context_s1028_supervisor_attempt_1.json'
EXTRA_SOURCES = (REQUALIFIED, FAILED_REGISTRATION, 'configs/trackocd_cvpr/C_registration_transport_retry1.json')


def directory(variant, seed):
    prefix = 'C_GT_causal_dev_registration_retry1' if (variant, seed) == ('Top8_pool_context', 1028) else 'C_GT_causal_dev_v1'
    return f'outputs/trackocd_cvpr/cvpr3/{prefix}_{variant}_s{seed}'


def cohort_sources():
    return tuple(p for v in VARIANTS for s in SEEDS for p in (
        directory(v, s) + '/launch.json', directory(v, s) + '/result.json', directory(v, s) + '_supervisor_attempt_1.json'))


def verified_matrix(root):
    # Establish ALL nine public trials before any aggregation; no partial win.
    for path in (*cohort_sources(), *EXTRA_SOURCES):
        p = root / path
        if not p.is_file(): raise FileNotFoundError(p)
        if p.is_symlink(): raise ValueError('No redirected cohort public files')
    cfg = json.loads((root / 'configs/trackocd_cvpr/C_GT_causal_development_v1.json').read_text())
    if sha256_file(root / FAILED_REGISTRATION) != 'e6315ce9eb14c5b583c8ecfe9d3e46a6a7eb8abb13e72e342e1c78a9292a987c':
        raise ValueError('Original failed registration receipt preserved')
    if sha256_file(root / REQUALIFIED) != '889df238859053535dce995a9ea428e32af227e25ed02d4ac2f394ab049d5280':
        raise ValueError('Exact executed separate first-C qualification, no blanket exception')
    table = {}
    core = ('configs/trackocd_cvpr/C_GT_causal_development_v1.json',
        'src/trackocd_cvpr/C_development_rollout.py', 'src/trackocd_cvpr/candidate_score_contract.py',
        'src/trackocd_cvpr/candidate_training_loss.py', 'src/trackocd_cvpr/candidate_scientific_training.py',
        'src/trackocd_cvpr/prefix_member_bank.py', 'src/trackocd_cvpr/causal_category_audit.py')
    for variant in VARIANTS:
        for seed in SEEDS:
            out = root / directory(variant, seed)
            r = json.loads((out / 'result.json').read_text())
            t = json.loads(out.with_name(out.name + '_supervisor_attempt_1.json').read_text())
            launch = json.loads((out / 'launch.json').read_text())
            for path in core:
                if launch['source_sha256'][path] != sha256_file(root / path):
                    raise ValueError('Identical original numeric protocol/kernel/loss; declared transport source may differ')
            if (r['status'] != 'MEASURED_CAUSAL_GT_C_DEVELOPMENT_NOT_M_PRED_FULL_TRUTH_OR_GOAL_ACCEPTANCE'
                    or r['model'] != variant or r['training_seed'] != seed or r['optimizer_steps'] != 0
                    or r['model_or_checkpoint_selection'] or r['final_Val_Test_or_new_nonKnown_semantic_values_read']
                    or r['full_semantic_truth_Hscore_CT_M_Pred_or_Goal_accepted']
                    or not r['actual_full_ID_rename_reverse_stream_equal']
                    or not r['actual_future_probe']['verified_same_prior_events']
                    or r['actual_future_probe']['future_observations_mutated'] < 1
                    or set(r['orders']) != set(map(str, SEEDS))):
                raise ValueError('All original complete causal frozen outcomes, no selected subset')
            if (variant, seed) == ('Top1_conditioned', 1027):
                q = json.loads((root / REQUALIFIED).read_text())
                if (sha256_file(out / 'result.json') != q['result_sha256']
                        or sha256_file(out.with_name(out.name + '_supervisor_attempt_1.json')) != q['original_terminal_sha256']
                        or not q['executed_source_fingerprints_checked_against_original_commit']
                        or q['original_supervisor_successful_owned_execution']
                        or not q['effective_completed_worker_qualification']):
                    raise ValueError('Actual immutable first-C completion exception binding')
                qualified_completed_C(r, t, cfg)
            elif not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0 or t['guard_stop_reason'] is not None:
                raise ValueError('Normal actual owned completion required for all other trials')
            if (r['resources']['CUDA_initialized'] or r['resources']['training_GPU_hours'] != 0
                    or r['resources']['peak_guarded_RSS_bytes'] > cfg['rss_bytes_ceiling']
                    or t['peak_sampled_owned_worker_RSS_bytes'] > cfg['rss_bytes_ceiling']
                    or r['resources']['wall_seconds_excluding_network'] > cfg['wall_seconds_ceiling']
                    or t['wall_seconds_including_imports_and_network'] > cfg['wall_seconds_ceiling']):
                raise ValueError('Original actual CPU/resource bounds, no completion guard rescue')
            for row in r['orders'].values():
                if ((row['observations'], row['metadata_frames'], row['empty_frames']) != (8753, 2562, 116)
                        or not row['model_state_unchanged'] or row['optimizer_steps'] != 0
                        or row['unknown_declared_background_or_Novel']
                        or set(row['prefix_snapshots']) != set(PREFIXES)
                        or sum(row['physical_actions'].values()) != 8753):
                    raise ValueError('Complete physical clock and unchanged fixed endpoint')
                for end in row['prefix_snapshots'].values():
                    if (end['physical_tracks'], end['Known_queries'], end['declared_pseudo_queries']) != (472, 302, 119):
                        raise ValueError('Complete fixed physical/query denominator, unknown not removed')
            table[(variant, seed)] = r
    return table


def aggregate_matrix(table):
    if set(table) != {(v, s) for v in VARIANTS for s in SEEDS}:
        raise ValueError('All9 matrix only; no seed/order/prefix winner')
    fields = ('Known_micro_accuracy', 'Known_macro_accuracy', 'all_physical_commit_track_coverage',
        'declared_pseudo_absorbed_as_Known_queries', 'declared_pseudo_anonymous_commit_queries',
        'certified_cross_video_reuse_micro_lower_bound', 'certified_cross_video_reuse_macro_lower_bound')
    means = {}
    for variant in VARIANTS:
        per_seed = {}
        for seed in SEEDS:
            r = table[(variant, seed)]
            per_seed[str(seed)] = {p: {key: sum(r['orders'][str(o)]['prefix_snapshots'][p][key] for o in SEEDS) / 3
                for key in fields} for p in PREFIXES}
        means[variant] = {'per_training_seed_order_mean': per_seed,
            'three_training_seed_mean': {p: {key: sum(per_seed[str(s)][p][key] for s in SEEDS) / 3
                for key in fields} for p in PREFIXES}}
    return {'models': means, 'selected_model_seed_order_or_prefix': False,
        'order_replays_are_independent_training_seeds': False,
        'official_CT_full_semantic_truth_Pred_or_Goal_accepted': False,
        'all_original_failed_receipts_and_declared_completion_exceptions_preserved': True}
