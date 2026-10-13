#!/usr/bin/env python3
"""Actual frozen-M complete policy CPU profile, not a new training trial.

Compose the existing own-child guarded persistent driver. Exact ALL9 endpoint,
authority/source and actual fingerprint-shadow gates before private inputs.
Only the fixed relative_quarantine/1027 endpoint is deserialized; no selection.
Public function timings only, with exact whole private trace parity required.
"""
import cProfile
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import shadow_M_raw_cache as runtime
from src.trackocd_cvpr.M_fingerprint_development_profile import verify_fingerprint_shadow, SOURCES as CACHE_SOURCES
from src.trackocd_cvpr.public_profile_summary import public_profile_summary
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/profile_M_policy.py'
CONFIG = 'configs/trackocd_cvpr/M_policy_cpu_profile_v1.json'
BASE_CONFIG = runtime.CONFIG
FAILED = 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_fingerprint_cache_v1_relative_quarantine_s1027_supervisor_attempt_1.json'
FAILED_SHA = '3c4592996c88992fac9cd02c28fe2df38d962b667b735492901d5a45d9ad643e'
TRACE_SHA = 'c98657973810452d4c4c00c8ad7bee98fa290c393a9e3de01f51a318c7a0ce68'
PROFILE_SOURCES = (SCRIPT, CONFIG, FAILED, 'src/trackocd_cvpr/public_profile_summary.py', *CACHE_SOURCES)
PROFILE_FUNCTION_SOURCES = tuple(p for p in runtime.SOURCES if p.startswith('src/trackocd_cvpr/') and p.endswith('.py'))


def checked_profile_configuration(root, cfg):
    expected = {
        'status': 'REGISTERED_FIXED_M_FULL_POLICY_CPROFILE_NOT_TRAINING_OR_METHOD_SELECTION',
        'data_config': 'configs/trackocd_cvpr/memory_optimizer_pilot_v1.json',
        'qualified_profile': 'configs/trackocd_cvpr/M_GT_causal_development_fingerprint_cache_v1.json',
        'failed_terminal': FAILED, 'failed_terminal_sha256': FAILED_SHA,
        'model': 'relative_quarantine', 'training_seed': 1027, 'order_seed': 1027,
        'endpoint': 'Fixed200', 'expected_private_full_trace_sha256': TRACE_SHA,
        'scope': [10, 54, 1268, 374, 1], 'execution_device': 'CPU', 'CPU_compute_workers': 1,
        'rss_bytes_ceiling': 3221225472, 'wall_seconds_ceiling': 3000,
        'output_bytes_ceiling': 3145728, 'output_directory': 'outputs/trackocd_cvpr/cvpr4/M_policy_cpu_profile_v1',
        'optimizer_steps': 0, 'new_labels_pixels_dev_final_Val_Test_payloads': False,
        'raw_profile_trace_callers_or_arguments_export': False,
        'performance_gain_full_dev_bottleneck_or_Goal_acceptance': False}
    if cfg != expected:
        raise ValueError('Only fixed entire policy M profile, no selection, new training or relaxed guards')
    if (root / FAILED).is_symlink() or sha256_file(root / FAILED) != FAILED_SHA:
        raise ValueError('Exact preserved first trained-M wall failure required')
    t = json.loads((root / FAILED).read_text())
    if (t['guard_stop_reason'] != 'owned_job_wall' or t['owned_worker_exit_code'] != -9
            or t['successful_owned_execution'] or t['complete_result_present']):
        raise ValueError('No partial old full-development result acceptance')
    qualified = verify_fingerprint_shadow(root)
    shadow = json.loads((root / 'outputs/trackocd_cvpr/cvpr4/M_fingerprint_cache_controller_shadow_v1/result.json').read_text())
    if shadow['cases']['relative_quarantine_s1027']['private_trace_sha256'] != TRACE_SHA:
        raise ValueError('Exact prior actual complete fixed-M policy trace required')
    return qualified


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write own worker')
    proof, sources, data = runtime.source_gate(args, cfg, safety)
    cohort = runtime.verified_cohort(ROOT)  # ALL9 byte hashes before labels or endpoint deserialization
    path, receipt = cohort[(cfg['model'], cfg['training_seed'])]
    if out.exists(): raise ValueError('Preserve every previous namespace')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, runtime.process_memory()['guarded_process_RSS_bytes'])
        reason = runtime.resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned fixed-M policy profile guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources}})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
        from src.trackocd_cvpr.R_development_evaluation import known_prototypes
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
        from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
        from src.trackocd_cvpr.qualified_fingerprint_cache_scope import qualified_fingerprint_cache_scope
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1); allowed = path.resolve()
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt', '.pth') and p != allowed):
                    raise PermissionError('Qualified fit/policy caches and single fixed own endpoint only')
        sys.addaudithook(barrier)
        guard(); model = MemberReliabilityMLP(pooled_control=False)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_state'], strict=True)
        model.eval().requires_grad_(False)
        if digest(model) != receipt['summary']['model_parameter_sha256_after']:
            raise ValueError('Exact frozen original Fixed200 endpoint')
        videos, targets, policy_scope = load_completed_policy_videos(ROOT, data, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, data, guard)
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i: row for i, row in enumerate(raw)}
        bindings = {slot: category for category, slot in targets.fit_bindings.items()}
        pseudo = {slot: category for category, slot in targets.pseudo_bindings.items()}
        labels = {}
        for video, arrays in videos:
            for track in arrays['track_id']:
                role, slot = targets.targets.get((video, int(track)), ('no_retained_semantic_truth', None))
                labels[(video, str(int(track)))] = (('Known', bindings[slot]) if role == 'Known'
                    else ('declared_pseudoNovel', pseudo[slot]) if role == 'policy_pseudo' else (role, None))
        stream_cfg = {'physical_tracks': 54, 'observations': 1268, 'metadata_frames': 374, 'empty_frames': 1,
            'raw_fixed_thresholds': data['raw_fixed_thresholds'], 'prefix_snapshots': [1, 2, 4, 8, 16], 'max_versions': 65536}
        profiler = cProfile.Profile(); first = time.monotonic()
        with qualified_fingerprint_cache_scope() as (raw_cache, fingerprints):
            profiler.enable()
            try:
                measured, trace = readonly_M_stream(videos, labels, known, bindings, stream_cfg,
                    cfg['model'], cfg['order_seed'], guard, model=model)
            finally: profiler.disable()
            counters = {'Raw_statistics_bytes': raw_cache.retained_bytes,
                'fingerprint_bytes': fingerprints.retained_bytes,
                'fingerprint_hits': fingerprints.hits, 'fingerprint_misses': fingerprints.misses}
        seconds = time.monotonic() - first; profiler.create_stats(); guard()
        trace_sha = runtime.trace_digest(trace)
        if (trace_sha != TRACE_SHA or torch.cuda.is_initialized() or len(videos) != 10
                or any(counters[k] for k in ('fingerprint_bytes', 'fingerprint_hits', 'fingerprint_misses'))
                or counters['Raw_statistics_bytes'] > 33554432 or not measured['model_state_unchanged']):
            raise ValueError('Whole trace/unchanged endpoint/exact weighted-M cache scope/CPU required')
        result = {'status': 'ACTUAL_FIXED_M_FULL_POLICY_CPU_PROFILE_NOT_FULL_DEV_OR_METHOD_GAIN',
            'live_remote_registration_proof': proof, 'policy_scope': policy_scope, 'offline_fit_scope': fit_scope,
            'model': cfg['model'], 'training_seed': cfg['training_seed'], 'order_seed': cfg['order_seed'],
            'fixed_endpoint_sha256': receipt['endpoint']['sha256'], 'preserved_failed_terminal_sha256': FAILED_SHA,
            'profile': public_profile_summary(profiler.stats, ROOT, PROFILE_FUNCTION_SOURCES),
            'instrumented_replay_wall_seconds': seconds, 'private_trace_sha256': trace_sha,
            'actual_complete_trace_exact_previous_shadow': True, 'model_state_unchanged': True,
            'metadata_frames': measured['metadata_frames'], 'empty_frames': measured['empty_frames'],
            'physical_tracks': measured['full_physical_bitset_scope'], 'observations': measured['observations'],
            'derived_cache_counters': counters, 'optimizer_steps': 0,
            'full_dev_bottleneck_isolated_speedup_or_scientific_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_registration': time.monotonic() - started,
                'CUDA_initialized': False, 'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'profiled_observations': 1268}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    original_gate = runtime.source_gate
    runtime.SCRIPT, runtime.CONFIG = SCRIPT, CONFIG
    runtime.SOURCES = tuple(dict.fromkeys((*runtime.SOURCES, *PROFILE_SOURCES)))
    def gate(args, cfg, safety):
        checked_profile_configuration(ROOT, cfg)
        return original_gate(args, json.loads((ROOT / BASE_CONFIG).read_text()), safety)
    runtime.source_gate, runtime.worker = gate, worker
    return runtime.main()


if __name__ == '__main__': sys.exit(main())
