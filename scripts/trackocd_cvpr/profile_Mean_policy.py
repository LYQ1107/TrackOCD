#!/usr/bin/env python3
"""Preregistered actual full-policy cached Mean CPU function profile.

Own-process composition of existing guarded persistent supervisor. No original
driver/controller modified. Fresh source/authority/all9/public-qualified gates
before fit/policy inputs. Existing endpoint fingerprints are hashed by the
cohort gate, but no checkpoint is deserialized; no pixels/optimizer/trace export.
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
from src.trackocd_cvpr.M_cached_development_profile import verify_actual_shadow, SOURCES as CACHE_SOURCES
from src.trackocd_cvpr.public_profile_summary import public_profile_summary
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/profile_Mean_policy.py'
CONFIG = 'configs/trackocd_cvpr/Mean_policy_cpu_profile_v1.json'
BASE_CONFIG = runtime.CONFIG
PROFILE_SOURCES = (SCRIPT, CONFIG, 'src/trackocd_cvpr/public_profile_summary.py',
    'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_raw_cache_v1_Mean_latest_supervisor_attempt_1.json', *CACHE_SOURCES)
PROFILE_FUNCTION_SOURCES = tuple(p for p in runtime.SOURCES if p.startswith('src/trackocd_cvpr/') and p.endswith('.py'))


def checked_profile_configuration(root, cfg):
    if (cfg['status'] != 'REGISTERED_FULL_POLICY_CACHED_MEAN_CPROFILE_NOT_SCIENTIFIC_REPLAY_OR_SELECTION'
            or (cfg['model'], cfg['order_seed'], cfg['optimizer_steps']) != ('Mean_latest', 1027, 0)
            or cfg['data_config'] != 'configs/trackocd_cvpr/memory_optimizer_pilot_v1.json'
            or cfg['qualified_profile'] != 'configs/trackocd_cvpr/M_GT_causal_development_raw_cache_v1.json'
            or (cfg['execution_device'], cfg['CPU_compute_workers'], cfg['rss_bytes_ceiling'],
                cfg['wall_seconds_ceiling'], cfg['output_bytes_ceiling']) != ('CPU', 1, 3221225472, 3000, 3145728)
            or cfg['output_directory'] != 'outputs/trackocd_cvpr/cvpr4/Mean_policy_cpu_profile_v1'
            or cfg['new_labels_pixels_checkpoints_or_dev_final_Val_Test_payloads']
            or cfg['raw_profile_trace_callers_or_arguments_export']
            or cfg['performance_gain_full_dev_bottleneck_or_Goal_acceptance']):
        raise ValueError('Only complete frozen policy Mean profile, not a scientific repair or selection')
    if (cfg['cached_failed_terminal'] != 'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_raw_cache_v1_Mean_latest_supervisor_attempt_1.json'
            or cfg['cached_failed_terminal_sha256'] != '04d07895d893c67b3fd6ef7a2b50ec0263f72156d3ab2f4cc88a9839819ab5e9'
            or sha256_file(root / cfg['cached_failed_terminal']) != cfg['cached_failed_terminal_sha256']):
        raise ValueError('Exact unchanged cached Mean failure required')
    qualified = verify_actual_shadow(root)
    shadow = json.loads((root / 'outputs/trackocd_cvpr/cvpr4/M_raw_cache_controller_shadow_v1/result.json').read_text())
    if cfg['expected_private_full_trace_sha256'] != shadow['cases']['Mean_latest']['private_trace_sha256']:
        raise ValueError('Exact original complete policy Mean trace fingerprint')
    return qualified


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh own worker without old bytecode')
    proof, sources, data = runtime.source_gate(args, cfg, safety)
    runtime.verified_cohort(ROOT)  # public ALL9 plus endpoint byte hashes before private labels; no deserialization
    if out.exists(): raise ValueError('Preserve previous namespace')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, runtime.process_memory()['guarded_process_RSS_bytes'])
        reason = runtime.resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned policy profile guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources}})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
        from src.trackocd_cvpr.R_development_evaluation import known_prototypes
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
        from src.trackocd_cvpr.qualified_raw_cache_scope import qualified_raw_cache_scope
        torch.set_num_threads(1)
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or p.suffix in ('.pt', '.pth'):
                    raise PermissionError('Qualified fit/policy cache only, no original data or weights')
        sys.addaudithook(barrier)
        guard(); videos, targets, policy_scope = load_completed_policy_videos(ROOT, data, guard)
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
        with qualified_raw_cache_scope() as cache:
            profiler.enable()
            try:
                measured, trace = readonly_M_stream(videos, labels, known, bindings, stream_cfg,
                    'Mean_latest', cfg['order_seed'], guard)
            finally:
                profiler.disable()
            retained = cache.retained_bytes
        seconds = time.monotonic() - first; profiler.create_stats(); guard()
        trace_sha = runtime.trace_digest(trace)
        if trace_sha != cfg['expected_private_full_trace_sha256'] or torch.cuda.is_initialized():
            raise ValueError('Actual whole trace differs under instrumentation or CUDA initialized')
        summary = public_profile_summary(profiler.stats, ROOT, PROFILE_FUNCTION_SOURCES)
        result = {'status': 'ACTUAL_FULL_POLICY_CACHED_MEAN_CPU_PROFILE_NOT_FULL_DEV_OR_METHOD_GAIN',
            'live_remote_registration_proof': proof, 'policy_scope': policy_scope, 'offline_fit_scope': fit_scope,
            'profile': summary, 'instrumented_replay_wall_seconds': seconds,
            'private_trace_sha256': trace_sha, 'actual_complete_trace_exact_previous_shadow': True,
            'metadata_frames': measured['metadata_frames'], 'empty_frames': measured['empty_frames'],
            'physical_tracks': measured['full_physical_bitset_scope'], 'observations': measured['observations'],
            'retained_derived_cache_bytes': retained, 'optimizer_steps': 0,
            'full_dev_bottleneck_isolated_speedup_or_scientific_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_registration': time.monotonic() - started,
                'CUDA_initialized': False, 'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'profiled_observations': 1268}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def main():
    # Own-process composition only; original source/control/foreign process untouched.
    original_gate = runtime.source_gate
    runtime.SCRIPT, runtime.CONFIG = SCRIPT, CONFIG
    runtime.SOURCES = tuple(dict.fromkeys((*runtime.SOURCES, *PROFILE_SOURCES)))
    def gate(args, cfg, safety):
        checked_profile_configuration(ROOT, cfg)
        base = json.loads((ROOT / BASE_CONFIG).read_text())
        return original_gate(args, base, safety)
    runtime.source_gate, runtime.worker = gate, worker
    return runtime.main()


if __name__ == '__main__': sys.exit(main())
