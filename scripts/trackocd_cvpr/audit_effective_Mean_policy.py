#!/usr/bin/env python3
"""External effective-support audit of complete original Mean policy pilot.

Own-process observer only, original model/worker/probes/guards remain. Requires
every prediction trace/outcome to match original measured pilot before acceptance.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import pilot_self_excluded_Mean_policy as original
from src.trackocd_cvpr import self_excluded_policy_rollout as rollout
from src.trackocd_cvpr.effective_mean_observer import observed_mean_constructors
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/audit_effective_Mean_policy.py'
CONFIG = 'configs/trackocd_cvpr/effective_Mean_policy_audit_v1.json'
RECOVERY = 'configs/trackocd_cvpr/effective_Mean_policy_registration_retry1.json'
PILOT = 'outputs/trackocd_cvpr/cvpr4/self_excluded_Mean_policy_pilot_v1'
SOURCES = (SCRIPT, CONFIG, RECOVERY,
    'outputs/trackocd_cvpr/cvpr4/effective_Mean_policy_audit_v1_supervisor_attempt_1.json',
    'src/trackocd_cvpr/effective_mean_observer.py',
    'src/trackocd_cvpr/effective_excluded_mean_audit.py', 'configs/trackocd_cvpr/source_effective_excluded_mean_audit_v1.json',
    PILOT + '/launch.json', PILOT + '/result.json', PILOT + '_supervisor_attempt_1.json')


def configuration(root, profile):
    recovery = profile.get('registration_recovery')
    output = profile['output_directory']
    if recovery is not None:
        expected = '768141eee7da3cff9a43e9d9a8964ad97a1b31b629ce21a5c29d7866041d118f'
        failed = root / 'outputs/trackocd_cvpr/cvpr4/effective_Mean_policy_audit_v1'
        terminal = failed.with_name(failed.name + '_supervisor_attempt_1.json')
        if (recovery != {'original_terminal_sha256': expected, 'source_transport_only_no_private_inputs': True}
                or output != 'outputs/trackocd_cvpr/cvpr4/effective_Mean_policy_audit_registration_retry1'
                or failed.exists() or sha256_file(terminal) != expected):
            raise ValueError('Exact pre-input registration failure/new namespace only')
        t = json.loads(terminal.read_text())
        if t['successful_owned_execution'] or t['complete_result_present'] or t['owned_worker_exit_code'] != 1 or t['guard_stop_reason'] is not None:
            raise ValueError('Original source-gate transport failure unchanged')
        profile = {k: v for k, v in profile.items() if k != 'registration_recovery'}
        profile['output_directory'] = 'outputs/trackocd_cvpr/cvpr4/effective_Mean_policy_audit_v1'
    if (profile['status'] != 'REGISTERED_EXTERNAL_EFFECTIVE_MEAN_FULL_POLICY_AUDIT_NOT_M_EMA_PRED_OR_GOAL'
            or profile['base_config'] != original.CONFIG
            or profile['prior_result_sha256'] != '9987e002ece8e906fe4ad85750c8f08f622dee8253011587b78dfe996b9df914'
            or profile['output_directory'] != 'outputs/trackocd_cvpr/cvpr4/effective_Mean_policy_audit_v1'
            or profile['M_EMA_full_development_semantic_truth_statistical_independence_or_Goal_acceptance']
            or profile['new_training_encoder_pixels_protected_labels_or_private_trace_upload']):
        raise ValueError('Exact external whole-policy observer, no changed policy or scope')
    base = json.loads((root / original.CONFIG).read_text()); original.checked_configuration(base)
    for key in ('execution_device', 'CPU_compute_workers', 'rss_bytes_ceiling', 'wall_seconds_ceiling', 'output_bytes_ceiling'):
        if profile[key] != base[key]: raise ValueError('Unchanged original CPU/RAM/time/output guards')
    if sha256_file(root / (PILOT + '/result.json')) != profile['prior_result_sha256']:
        raise ValueError('Actual complete original control/probe result required')
    terminal = json.loads((root / (PILOT + '_supervisor_attempt_1.json')).read_text())
    if not terminal['successful_owned_execution'] or terminal['owned_worker_exit_code'] != 0 or terminal['guard_stop_reason'] is not None:
        raise ValueError('Original pilot actually completed normally, not a partial chosen case')
    cfg = dict(base); cfg['output_directory'] = output
    return cfg


def main():
    runtime = original.runtime; original_gate, original_stream = runtime.source_gate, rollout.readonly_excluded_stream
    runtime.SCRIPT, runtime.CONFIG = SCRIPT, RECOVERY
    runtime.SOURCES = tuple(dict.fromkeys((*runtime.SOURCES, *original.SOURCES, *SOURCES)))
    records = []
    def gate(args, profile, safety):
        cfg = configuration(ROOT, profile) if 'base_config' in profile else profile
        original.verify_fingerprint_shadow(ROOT)
        proof, sources, data = original_gate(args, json.loads((ROOT / original.BASE_CONFIG).read_text()), safety)
        return proof, sources, data
    def stream(videos, labels, known, bindings, cfg, mode, seed, guard, **kwargs):
        rename = kwargs.get('rename_and_reverse', False)
        order = [videos[i][0] for i in np.random.default_rng(seed).permutation(len(videos))]
        with observed_mean_constructors(labels, bindings, cfg['prefix_snapshots'], order, rename=rename, videos=videos) as observer:
            measured, trace = original_stream(videos, labels, known, bindings, cfg, mode, seed, guard, **kwargs)
            effective, private = observer.finish()
        records.append((mode, seed, rename, effective, private))
        return measured, trace  # original actor/probe sees NO audit feedback
    def worker(args, profile, safety):
        cfg = configuration(ROOT, profile)
        original.worker(args, cfg, safety)
        if len(records) != 18: raise ValueError('Complete original old/cache/orders/ID/future18 streams, not subset')
        out = ROOT / cfg['output_directory']; measured = json.loads((out / 'result.json').read_text())
        prior = json.loads((ROOT / (PILOT + '/result.json')).read_text()); results = {}
        for index, mode in enumerate(cfg['controls']):
            chunk = records[index * 6:index * 6 + 6]
            if [(r[0], r[1], r[2]) for r in chunk] != [(mode, s, flag) for s, flag in
                    ((1027, False), (1027, False), (1028, False), (1029, False), (1027, True), (1027, False))]:
                raise ValueError('Exact actual full stream ordering/observer alignment')
            if chunk[0][3:] != chunk[1][3:] or chunk[1][3:] != chunk[4][3:]:
                raise ValueError('Actual full effective audit old/cache/ID/reverse differs')
            first_trace, future_trace = chunk[1][4], chunk[5][4]
            first_video = first_trace[0][0]
            # Compare all earlier frames through first nonempty observed frame;
            # the original probe independently checks real39/141 mutations.
            split = next(row[1] for row in first_trace if row[2])
            left = tuple(r for r in first_trace if r[0] == first_video and r[1] <= split)
            right = tuple(r for r in future_trace if r[0] == first_video and r[1] <= split)
            if not left or left != right: raise ValueError('Actual effective selected-support audit changed from future input')
            current = measured['controls'][mode]; old = prior['controls'][mode]
            if (current['orders'] != old['orders'] or current['private_order1027_trace_sha256'] != old['private_order1027_trace_sha256']):
                raise ValueError('Every original control/order and full actor trace must stay unchanged')
            results[mode] = {'effective_orders': {str(chunk[i][1]): chunk[i][3] for i in (1, 2, 3)},
                'actual_full_effective_old_cache_ID_reverse_and_prior_future_equal': True,
                'original_actor_all_orders_and_private_trace_match_previous_pilot': True,
                'private_order1027_effective_trace_sha256': runtime.trace_digest(first_trace)}
        atomic_json(out / 'effective_support_result.json', {
            'status': 'ACTUAL_EXTERNAL_EFFECTIVE_MEAN_POLICY_AUDIT_NOT_M_EMA_FULL_TRUTH_OR_GOAL',
            'original_actor_result_sha256': sha256_file(out / 'result.json'),
            'controls': results, 'new_optimizer_steps': 0,
            'full_development_Pred_statistical_independence_or_Goal_accepted': False})
    runtime.source_gate, runtime.worker = gate, worker
    rollout.readonly_excluded_stream = stream
    try: return runtime.main()
    finally: rollout.readonly_excluded_stream = original_stream


if __name__ == '__main__': sys.exit(main())
