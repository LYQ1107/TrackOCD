#!/usr/bin/env python3
"""Owned full-policy ALL11 old/cache controller shadow; no training/GT repair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_cvpr.remote_registration import verify_registration
from src.trackocd_cvpr.M_development_cohort import cohort_sources, verified_cohort, VARIANTS, SEEDS
from scripts.trackocd_cvpr.evaluate_M_development import SOURCES as M_SOURCES
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.qualify_policy_training_input import settled_after_counter_error

CONFIG = 'configs/trackocd_cvpr/M_raw_cache_controller_shadow_v1.json'
SCRIPT = 'scripts/trackocd_cvpr/shadow_M_raw_cache.py'
SOURCES = tuple(dict.fromkeys((*M_SOURCES, CONFIG, SCRIPT,
    'src/trackocd_cvpr/qualified_raw_cache_scope.py', 'src/trackocd_cvpr/frozen_raw_statistics_cache.py',
    'configs/trackocd_cvpr/source_frozen_raw_statistics_cache_v1.json',
    'outputs/trackocd_cvpr/cvpr4/qualified_frozen_raw_statistics_v1/launch.json',
    'outputs/trackocd_cvpr/cvpr4/qualified_frozen_raw_statistics_v1/result.json',
    'outputs/trackocd_cvpr/cvpr4/M_GT_causal_dev_v1_Mean_latest_supervisor_attempt_1.json')))


def source_gate(args, cfg, safety):
    sources = (*SOURCES, *cohort_sources())
    for path in sources:
        if not (ROOT / path).is_file(): raise FileNotFoundError(ROOT / path)
        if (ROOT / path).is_symlink(): raise ValueError('No redirected shadow source/cohort')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in sources:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Shadow source/ALL9/cohort/numeric proof not actually registered')
    data = json.loads((ROOT / cfg['data_config']).read_text())
    authority = json.loads((ROOT / data['approval']).read_text())
    if (authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'
            or cfg['status'] != 'REGISTERED_FULL_POLICY_CONTROLLER_RAW_CACHE_SHADOW_ALL9_ENDPOINTS_REQUIRED'
            or cfg['training_seeds'] != list(SEEDS) or cfg['order_seed'] != 1027
            or cfg['models'] != ['Mean_latest', 'StateEMA_latest', *VARIANTS]
            or (cfg['execution_device'], cfg['CPU_compute_workers'], cfg['rss_bytes_ceiling'],
                cfg['wall_seconds_ceiling'], cfg['output_bytes_ceiling']) != ('CPU', 1, 3221225472, 3000, 3145728)
            or cfg['old_failure_overwritten_or_guards_relaxed']
            or cfg['full72development_parity_speedup_purity_or_Goal_acceptance']):
        raise ValueError('Exact authorized full-policy shadow, not scientific selection')
    if sha256_file(ROOT / cfg['numeric_qualification']) != cfg['numeric_qualification_sha256']:
        raise ValueError('Actual complete numeric qualification required')
    q = json.loads((ROOT / cfg['numeric_qualification']).read_text())['numeric_qualification']
    if (q['canonical_visible_prefixes_checked'] != 8753 or not q['exact_CPU_original_cached_miss_hit_and_duplicate_scope_tensor_bytes']
            or q['semantic_class_final_values_Known_index_or_GT_labels_decoded']):
        raise ValueError('Actual8753 semantic-free exact prefix tensor proof')
    return proof, sources, data


def trace_digest(rows):
    h = hashlib.sha256()
    for row in rows:
        b = repr(row).encode('utf8'); h.update(len(b).to_bytes(8, 'little')); h.update(b)
    return h.hexdigest()


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write own worker')
    proof, sources, data = source_gate(args, cfg, safety)
    cohort = verified_cohort(ROOT)  # before any privateGT/policy/fit payload.
    if out.exists(): raise ValueError('Preserve previous attempt')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned full-policy shadow guard:' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof, 'source_sha256': {p: sha256_file(ROOT / p) for p in sources}})
    try:
        import torch
        from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
        from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
        from src.trackocd_cvpr.R_development_evaluation import known_prototypes
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
        from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
        from src.trackocd_cvpr.qualified_raw_cache_scope import qualified_raw_cache_scope
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1); allowed = {p.resolve() for p, _ in cohort.values()}
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt', '.pth') and p not in allowed):
                    raise PermissionError('Qualified fit/policy caches and ownMendpoints only')
        sys.addaudithook(barrier)
        guard(); videos, targets, policy_scope = load_completed_policy_videos(ROOT, data, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, data, guard)
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i: row for i, row in enumerate(raw)}
        known_binding = {slot: category for category, slot in targets.fit_bindings.items()}
        pseudo_binding = {slot: category for category, slot in targets.pseudo_bindings.items()}
        labels = {}
        for video, arrays in videos:
            for track in arrays['track_id']:
                role, slot = targets.targets.get((video, int(track)), ('no_retained_semantic_truth', None))
                value = ('Known', known_binding[slot]) if role == 'Known' else ('declared_pseudoNovel', pseudo_binding[slot]) if role == 'policy_pseudo' else (role, None)
                labels[(video, str(int(track)))] = value
        stream_cfg = {'physical_tracks': 54, 'observations': 1268, 'metadata_frames': 374, 'empty_frames': 1,
            'raw_fixed_thresholds': data['raw_fixed_thresholds'], 'prefix_snapshots': [1, 2, 4, 8, 16], 'max_versions': 65536}
        cases = [('Mean_latest', None), ('StateEMA_latest', None)] + [(v, s) for v in VARIANTS for s in SEEDS]
        results = {}
        for name, seed in cases:
            guard(); model = None
            if seed is not None:
                path, receipt = cohort[(name, seed)]
                model = MemberReliabilityMLP(pooled_control=name == 'pooled_weighted')
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_state'], strict=True)
                model.eval().requires_grad_(False)
                if digest(model) != receipt['summary']['model_parameter_sha256_after']: raise ValueError('Actual fixed owned endpoint')
            first = time.monotonic()
            old, old_trace = readonly_M_stream(videos, labels, known, known_binding, stream_cfg, name, cfg['order_seed'], guard, model=model)
            old_seconds = time.monotonic() - first; first = time.monotonic()
            with qualified_raw_cache_scope() as cache:
                new, new_trace = readonly_M_stream(videos, labels, known, known_binding, stream_cfg, name, cfg['order_seed'], guard, model=model)
                retained = cache.retained_bytes
            new_seconds = time.monotonic() - first
            if old != new or old_trace != new_trace: raise ValueError('Entire full-policy outcome/event/score/routing/chosen-version/numeric-ledger shadow differs')
            key = name + ('_s' + str(seed) if seed is not None else '')
            results[key] = {'actual_full_policy_event_score_routing_chosen_version_numeric_ledger_and_outcome_exact': True,
                'private_trace_sha256': trace_digest(old_trace), 'retained_derived_cache_bytes': retained,
                'old_seconds': old_seconds, 'cache_seconds': new_seconds,
                'timing_is_shared_host_sequential_pair_not_isolated_speedup_claim': True}
            atomic_json(out / 'progress.json', {'completed_case': key, 'completed_cases': len(results), 'optimizer_steps': 0})
        guard()
        if len(results) != 11 or torch.cuda.is_initialized(): raise ValueError('ALL11 CPU shadows, no favorable subset')
        result = {'status': 'ACTUAL_ALL11_FULL_POLICY_RAW_CACHE_CONTROLLER_SHADOW_NOT_FULL_DEV_OR_METHOD_GAIN',
            'live_remote_registration_proof': proof, 'policy_scope': policy_scope, 'offline_fit_scope': fit_scope,
            'cases': results, 'physical_tracks_per_case': 54, 'observations_per_replay': 1268,
            'full72development_or_Pred_purity_or_Goal_accepted': False, 'optimizer_steps': 0,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_registration': time.monotonic() - started,
                          'CUDA_initialized': False, 'training_GPU_hours': 0}}
        atomic_json(out / 'result.json', result); print(json.dumps({'status': result['status'], 'cases': len(results)}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False}); raise


def command(args, flag):
    return [sys.executable, '-B', str(ROOT / SCRIPT), flag, '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy]


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']; terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or terminal.exists(): raise ValueError('No attempt overwrite')
    started, peak, reason = time.monotonic(), 0, None
    if resources_ok(cfg, safety, ROOT, started, 0): raise RuntimeError('Pre-spawn resource guard')
    cmd = command(args, '--worker'); cmd = cmd[:2] + ['-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache')] + cmd[2:]
    child = subprocess.Popen(cmd, cwd=ROOT)
    try:
        while child.poll() is None:
            try:
                peak = max(peak, process_memory(Path(f'/proc/{child.pid}/status').read_text(), Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None or settled_after_counter_error(child): break
                if time.monotonic() - started <= 2: time.sleep(.2); continue
                reason = 'owned_memory_counter_unavailable'; child.kill(); break
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']: reason = 'owned_output_size'
            if reason:
                if child.poll() is None: child.kill()
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    complete = (out / 'result.json').is_file()
    atomic_json(terminal, {'successful_owned_execution': code == 0 and reason is None and complete,
        'owned_worker_exit_code': code, 'guard_stop_reason': reason, 'complete_result_present': complete,
        'peak_sampled_owned_worker_RSS_bytes': peak, 'wall_seconds_including_registration': time.monotonic() - started,
        'only_new_owned_Popen_worker_managed': True, 'training_GPU_hours': 0})
    return 0 if code == 0 and reason is None and complete else 1


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    g = p.add_mutually_exclusive_group(); g.add_argument('--worker', action='store_true'); g.add_argument('--supervisor', action='store_true')
    args = p.parse_args(); cfg = json.loads((ROOT / CONFIG).read_text()); safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args, cfg, safety); return 0
    if args.supervisor: return supervise(args, cfg, safety)
    proof, _, _ = source_gate(args, cfg, safety); out = ROOT / cfg['output_directory']
    receipt = out.with_name(out.name + '_persistent_launch.json'); log = out.with_name(out.name + '_persistent_stdout.log')
    if any(x.exists() for x in (out, receipt, log, out.with_name(out.name + '_supervisor_attempt_1.json'))): raise ValueError('Never relaunch old namespace')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try: body = path.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if os.fsencode(str(ROOT / SCRIPT)) + b'\0' in body and (b'--worker\0' in body or b'--supervisor\0' in body): raise ValueError('Owned shadow already live')
    if resources_ok(cfg, safety, ROOT, time.monotonic(), 0): raise RuntimeError('Pre-spawn resource guard')
    out.parent.mkdir(parents=True, exist_ok=True); cmd = command(args, '--supervisor')
    with log.open('xb', buffering=0) as stream:
        os.chmod(log, 0o600); child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    try: ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: ticks = None
    atomic_json(receipt, {'status': 'OWNED_SHADOW_LAUNCHED_NOT_COMPLETED', 'supervisor_pid': child.pid, 'linux_start_ticks': ticks,
        'command_sha256': hashlib.sha256(b'\0'.join(os.fsencode(x) for x in cmd)).hexdigest(), 'output_namespace': cfg['output_directory'],
        'live_remote_registration_proof': proof, 'normal_terminal_receipt': str(out.with_name(out.name + '_supervisor_attempt_1.json').relative_to(ROOT)),
        'start_new_session': True, 'no_private_GT_in_launcher_or_foreign_process_changes': True})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETED', 'receipt': str(receipt.relative_to(ROOT))}), flush=True); return 0


if __name__ == '__main__': sys.exit(main())
