#!/usr/bin/env python3
"""Owned whole-policy Mean support-exclusion pilot with real full probes.

No trained models selected, no new optimizer/encoder or protected data. Cache
overlay is qualified by actual complete OLD/new streams before using results.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_cvpr.M_fingerprint_development_profile import verify_fingerprint_shadow, SOURCES as PROOF_SOURCES
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/pilot_self_excluded_Mean_policy.py'
CONFIG = 'configs/trackocd_cvpr/self_excluded_Mean_policy_pilot_v1.json'
BASE_CONFIG = runtime.CONFIG
SOURCES = (SCRIPT, CONFIG, 'src/trackocd_cvpr/self_excluded_policy_rollout.py',
    'src/trackocd_cvpr/self_excluded_mean_control.py', 'src/trackocd_cvpr/independent_support_view.py',
    'configs/trackocd_cvpr/source_self_excluded_mean_control_v1.json',
    'configs/trackocd_cvpr/source_independent_support_view_v1.json', *PROOF_SOURCES)


def checked_configuration(cfg):
    if (cfg['status'] != 'REGISTERED_FULL_POLICY_MEAN_EXCLUSION_PILOT_NOT_TRAINED_M_DEV_PRED_OR_GOAL'
            or cfg['controls'] != ['Mean_latest', 'exclude_query_alias_scope', 'exclude_current_video']
            or cfg['order_seeds'] != [1027, 1028, 1029] or cfg['cache_shadow_order'] != 1027
            or cfg['data_config'] != 'configs/trackocd_cvpr/memory_optimizer_pilot_v1.json'
            or (cfg['physical_tracks'], cfg['observations'], cfg['metadata_frames'], cfg['empty_frames']) != (54, 1268, 374, 1)
            or cfg['raw_fixed_thresholds'] != {'tau_known': .65, 'tau_existing': .55}
            or cfg['prefix_snapshots'] != [1, 2, 4, 8, 16] or cfg['max_versions'] != 65536
            or (cfg['execution_device'], cfg['CPU_compute_workers'], cfg['rss_bytes_ceiling'],
                cfg['wall_seconds_ceiling'], cfg['output_bytes_ceiling']) != ('CPU', 1, 3221225472, 3000, 3145728)
            or cfg['output_directory'] != 'outputs/trackocd_cvpr/cvpr4/self_excluded_Mean_policy_pilot_v1'
            or cfg['new_optimizer_steps'] != 0 or cfg['new_checkpoint_pixels_encoder_dev_final_Val_Test_or_Novel_values_allowed']
            or cfg['private_trace_or_GT_upload'] or cfg['scientific_self_excluded_M_EMA_full_development_active_purity_or_Goal_accepted']):
        raise ValueError('Exact whole-policy complete controls/probes/guards; no selected subset or scientific acceptance')


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh isolated own worker')
    proof, sources, data = runtime.source_gate(args, cfg, safety)
    runtime.verified_cohort(ROOT)  # all9 endpoint hashes, no model deserialization
    if out.exists(): raise ValueError('Never overwrite an earlier pilot')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, runtime.process_memory()['guarded_process_RSS_bytes'])
        reason = runtime.resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned self-exclusion policy pilot guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources}})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
        from src.trackocd_cvpr.R_development_evaluation import known_prototypes
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.M_development_rollout import MFrameTrace
        from src.trackocd_cvpr.self_excluded_policy_rollout import readonly_excluded_stream, actual_excluded_probes, excluded_raw_cache_scope
        torch.set_num_threads(1)
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or p.suffix in ('.pt', '.pth'):
                    raise PermissionError('Qualified fit/policy cache only, no original data or checkpoint deserialization')
        sys.addaudithook(barrier)
        guard(); videos, targets, policy_scope = load_completed_policy_videos(ROOT, data, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, data, guard)
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i: row for i, row in enumerate(raw)}
        bindings = {slot: category for category, slot in targets.fit_bindings.items()}
        pseudo = {slot: category for category, slot in targets.pseudo_bindings.items()}; labels = {}
        for video, arrays in videos:
            for track in arrays['track_id']:
                role, slot = targets.targets.get((video, int(track)), ('no_retained_semantic_truth', None))
                labels[(video, str(int(track)))] = (('Known', bindings[slot]) if role == 'Known'
                    else ('declared_pseudoNovel', pseudo[slot]) if role == 'policy_pseudo' else (role, None))
        controls = {}
        for mode in cfg['controls']:
            first = time.monotonic()
            old, old_trace = readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, 1027, guard)
            old_seconds = time.monotonic() - first; first = time.monotonic()
            with excluded_raw_cache_scope() as (raw_cache, fingerprints):
                measured, trace = readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, 1027, guard)
                if old != measured or old_trace != trace:
                    raise ValueError('Complete actual OLD/cache events/scores/history/query views/outcomes differ')
                cache_seconds = time.monotonic() - first
                if mode == 'Mean_latest':
                    standard = tuple(MFrameTrace(r.video, r.result, r.chosen_versions, r.version_history_sha256) for r in trace)
                    if runtime.trace_digest(standard) != '7461015c0f64bf89c294630f08fac2cc8c8083a664b541c221d49bd5121f5ed5':
                        raise ValueError('Exact already qualified full policy Mean baseline, not a changed control')
                table = {'1027': measured}; timings = {'1027': cache_seconds}
                for order in cfg['order_seeds'][1:]:
                    began = time.monotonic()
                    table[str(order)], _ = readonly_excluded_stream(videos, labels, known, bindings, cfg, mode, order, guard)
                    timings[str(order)] = time.monotonic() - began
                probes = actual_excluded_probes(videos, labels, known, bindings, cfg, mode, 1027, guard, measured, trace)
                retained = raw_cache.retained_bytes + fingerprints.retained_bytes
            controls[mode] = {'orders': table, **probes, 'actual_complete_OLD_cache_shadow_equal': True,
                'private_order1027_trace_sha256': runtime.trace_digest(trace), 'combined_derived_cache_bytes': retained,
                'OLD_shadow_seconds': old_seconds, 'cached_full_order_seconds': timings,
                'timings_shared_host_not_isolated_speedup_or_matched_purity_superiority': True}
            atomic_json(out / 'progress.json', {'complete_controls': len(controls), 'new_optimizer_steps': 0})
        guard()
        if len(controls) != 3 or torch.cuda.is_initialized(): raise ValueError('Complete all3 CPU controls required')
        result = {'status': 'ACTUAL_FULL_POLICY_MEAN_EXCLUSION_CONTROL_NOT_TRAINED_M_DEV_ACTIVE_PURITY_OR_GOAL',
            'live_remote_registration_proof': proof, 'policy_scope': policy_scope, 'offline_fit_scope': fit_scope,
            'controls': controls, 'new_optimizer_steps': 0,
            'video_orders_are_independent_training_seeds': False,
            'strict_history_certificate_is_filtered_active_support_semantic_independence': False,
            'full_development_Pred_joint_or_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_registration': time.monotonic() - started,
                'CUDA_initialized': False, 'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'controls': 3}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    original_gate = runtime.source_gate
    runtime.SCRIPT, runtime.CONFIG = SCRIPT, CONFIG
    runtime.SOURCES = tuple(dict.fromkeys((*runtime.SOURCES, *SOURCES)))
    def gate(args, cfg, safety):
        checked_configuration(cfg)
        verify_fingerprint_shadow(ROOT)
        return original_gate(args, json.loads((ROOT / BASE_CONFIG).read_text()), safety)
    runtime.source_gate, runtime.worker = gate, worker
    return runtime.main()


if __name__ == '__main__': sys.exit(main())
