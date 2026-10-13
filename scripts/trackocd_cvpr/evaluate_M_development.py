#!/usr/bin/env python3
"""All-nine gated, read-only M/Mean/state-EMA causal GT-development worker.

No automatic cohort completion, checkpoint selection, GT memory repair or
training. Default CLI detaches only its own guarded supervisor, retaining
actual process identity and immutable completion receipts.
"""
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
from scripts.trackocd_cvpr.evaluate_C_development import BASE_SOURCES
from scripts.trackocd_cvpr.pilot_memory_optimizer import SOURCES as PILOT_SOURCES
from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
from scripts.trackocd_cvpr.audit_train_metadata_containment import process_memory
from scripts.trackocd_cvpr.qualify_policy_training_input import settled_after_counter_error
from src.trackocd_cvpr.memory_transport_recovery import RECOVERY_SOURCES
from src.trackocd_cvpr.M_terminal_replica_recovery import SOURCES as TERMINAL_REPLICA_SOURCES

CONFIG = 'configs/trackocd_cvpr/M_GT_causal_development_v1.json'
SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development.py'
MODELS = ('Mean_latest', 'StateEMA_latest', *VARIANTS)
SOURCES = tuple(dict.fromkeys((*BASE_SOURCES, *PILOT_SOURCES, *RECOVERY_SOURCES, *TERMINAL_REPLICA_SOURCES, CONFIG, SCRIPT,
    'configs/trackocd_cvpr/autonomous_research.json',
    'configs/trackocd_cvpr/M_GT_scientific_window_v1.json',
    'src/trackocd_cvpr/M_development_cohort.py', 'src/trackocd_cvpr/M_development_rollout.py',
    'src/trackocd_cvpr/M_development_probes.py', 'src/trackocd_cvpr/frame_atomic_state_ema.py',
    'src/trackocd_cvpr/versioned_causal_outcome_audit.py',
    'src/trackocd_cvpr/memory_scientific_training.py', 'src/trackocd_cvpr/version_lineage_audit.py')))


def configuration(root, args):
    m = json.loads((root / CONFIG).read_text())
    data = json.loads((root / m['data_config']).read_text())
    if (m['status'] != 'REGISTERED_READONLY_CAUSAL_M_DEV_ALL9_FIXED_FITS_REQUIRED'
            or tuple(m['models']) != MODELS or tuple(m['training_seeds']) != SEEDS
            or tuple(m['order_seeds']) != SEEDS or m['optimizer_steps'] != 0
            or m['select_model_or_checkpoint'] or m['execution_or_Model_Goal_acceptance_from_this_manifest']
            or m['final_Val_Test_encoder_externalweights_newNonKnown_truth_or_vocabulary']
            or m['execution_device'] != 'CPU' or m['CPU_compute_workers'] != 1
            or (m['physical_tracks'], m['observations'], m['metadata_frames'], m['empty_frames']) != (472, 8753, 2562, 116)
            or m['raw_fixed_thresholds'] != {'tau_known': .65, 'tau_existing': .55}
            or m['prefix_snapshots'] != [1, 2, 4, 8, 16] or m['max_versions'] != 65536
            or m['rss_bytes_ceiling'] != 3221225472 or m['wall_seconds_ceiling'] != 3000
            or m['output_bytes_ceiling'] != 16777216):
        raise ValueError('Exact registered M development protocol, no selected subset or changed guards')
    if args.model not in MODELS or args.seed not in SEEDS or (args.model not in VARIANTS and args.seed != 1027):
        raise ValueError('Baselines have zero fits, not three independent trained seeds')
    cfg = {**data, **m}
    cfg['output_directory'] = m['output_prefix'] + '_' + args.model + ('_s' + str(args.seed) if args.model in VARIANTS else '')
    return cfg


def source_gate(args, cfg, safety):
    # Require every public result FIRST, before private endpoints or dev inputs.
    for path in cohort_sources():
        p = ROOT / path
        if not p.is_file(): raise FileNotFoundError(p)
        if p.is_symlink(): raise ValueError('No redirected public cohort')
    sources = (*SOURCES, *cohort_sources())
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit, args.git_proxy, allow_public_api=True)
    for path in sources:
        if subprocess.check_output(['git', 'show', args.preregistration_commit + ':' + path], cwd=ROOT) != (ROOT / path).read_bytes():
            raise ValueError('Actual M evaluation dependency/cohort/cache/source not registered: ' + path)
    authority = json.loads((ROOT / cfg['approval']).read_text())
    if authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN':
        raise ValueError('Existing formal authorization required')
    return proof, sources


def worker(args, cfg, safety):
    out = ROOT / cfg['output_directory']
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(out / '_fresh_no_write_cache'):
        raise ValueError('Fresh no-write CPU worker required')
    proof, sources = source_gate(args, cfg, safety)
    cohort = verified_cohort(ROOT)  # all nine normal endpoints BEFORE private dev loaders.
    if out.exists(): raise ValueError('Preserve all previous attempts')
    out.mkdir(parents=True); started, peak = time.monotonic(), 0
    prior_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, process_memory()['guarded_process_RSS_bytes'])
        reason = resources_ok(cfg, safety, ROOT, started, peak)
        size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if reason or size > cfg['output_bytes_ceiling'] or prior_bytes + size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned M-development resource guard: ' + str(reason))
    atomic_json(out / 'launch.json', {'live_remote_registration_proof': proof,
        'source_sha256': {p: sha256_file(ROOT / p) for p in sources},
        'scope': 'ALL9 completed M endpoints/causal GT development/no selection'})
    try:
        import torch
        from src.trackocd_cvpr.C_development_rollout import dev_videos_and_external_labels
        from src.trackocd_cvpr.M_development_rollout import readonly_M_stream
        from src.trackocd_cvpr.M_development_probes import actual_M_probes
        from src.trackocd_cvpr.R_development_evaluation import load_completed_fit_tracks, known_prototypes
        from src.trackocd_cvpr.fit_loss_supervision import project_fit_loss_targets
        from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
        from src.trackocd_cvpr.member_reliability import MemberReliabilityMLP
        from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
        torch.set_num_threads(1)
        allowed = {p.resolve() for p, _ in cohort.values()}
        def barrier(event, values):
            if event == 'socket.connect': raise PermissionError('No network after registration')
            if event == 'open' and isinstance(values[0], (str, bytes)):
                p = Path(os.fsdecode(values[0])).resolve()
                if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or (p.suffix in ('.pt', '.pth') and p not in allowed):
                    raise PermissionError('Only qualified cache and own M endpoints, no original data or encoder weights')
        sys.addaudithook(barrier)
        guard(); videos, labels, scope = dev_videos_and_external_labels(ROOT, cfg, guard)
        fit, fit_scope = load_completed_fit_tracks(ROOT, cfg, guard)
        bindings, _, _ = project_fit_loss_targets(ROOT / cfg['private_assignment'], ROOT / cfg['private_Known_support'])
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i: row for i, row in enumerate(raw)}
        known_bindings = {slot: category for category, slot in bindings.items()}
        model = None
        if args.model in VARIANTS:
            path, receipt = cohort[(args.model, args.seed)]
            model = MemberReliabilityMLP(pooled_control=args.model == 'pooled_weighted')
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model_state'], strict=True)
            model.eval().requires_grad_(False)
            if digest(model) != receipt['summary']['model_parameter_sha256_after']:
                raise ValueError('Exact owned fixed endpoint; no pilot or other-model warmstart')
        table = {}; first_private = None
        for order in cfg['order_seeds']:
            measured, private = readonly_M_stream(videos, labels, known, known_bindings, cfg,
                args.model, order, guard, model=model)
            table[str(order)] = measured
            if first_private is None: first_private = private
            atomic_json(out / 'progress.json', {'completed_order': order, 'physical_tracks': 472, 'no_optimizer': True})
        probes = actual_M_probes(videos, labels, known, known_bindings, cfg, args.model,
            cfg['order_seeds'][0], guard, table[str(cfg['order_seeds'][0])], first_private, model=model)
        guard()
        result = {'status': 'MEASURED_CAUSAL_GT_M_DEVELOPMENT_NOT_ACTIVE_PURITY_SELF_EXCLUSION_PRED_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof': proof, 'model': args.model,
            'training_seed': args.seed if model is not None else None,
            'development_scope': scope, 'offline_fit_prior_scope': fit_scope, 'orders': table, **probes,
            'optimizer_steps': 0, 'model_or_checkpoint_selection': False,
            'final_Val_Test_or_new_nonKnown_semantic_values_read': False,
            'coverage_or_latency_matched_purity_superiority_accepted': False,
            'current_active_purity_official_CT_self_exclusion_Pred_joint_or_Goal_accepted': False,
            'resources': {'peak_guarded_RSS_bytes': peak, 'wall_seconds_excluding_network': time.monotonic() - started,
                          'training_GPU_hours': 0, 'CUDA_initialized': torch.cuda.is_initialized()}}
        if result['resources']['CUDA_initialized']: raise ValueError('CPU-only worker')
        atomic_json(out / 'result.json', result)
        print(json.dumps({'status': result['status'], 'model': args.model, 'training_seed': result['training_seed']}), flush=True)
    except Exception as error:
        atomic_json(out / 'failure.json', {'exception_type': type(error).__name__, 'error': str(error), 'complete': False})
        raise


def command(args, flag):
    return [sys.executable, '-B', str(ROOT / SCRIPT), flag, '--model', args.model,
        '--seed', str(args.seed), '--preregistration-commit', args.preregistration_commit, '--git-proxy', args.git_proxy]


def worker_command(args, out):
    cmd = command(args, '--worker')
    return cmd[:2] + ['-X', 'pycache_prefix=' + str(out / '_fresh_no_write_cache')] + cmd[2:]


def supervise(args, cfg, safety):
    out = ROOT / cfg['output_directory']; terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    if out.exists() or terminal.exists(): raise ValueError('No overwriting previous attempt')
    started, peak, threads, reason = time.monotonic(), 0, 0, None
    if resources_ok(cfg, safety, ROOT, started, 0): raise RuntimeError('Pre-spawn resource guard')
    child = subprocess.Popen(worker_command(args, out), cwd=ROOT)
    try:
        while child.poll() is None:
            try:
                status = Path(f'/proc/{child.pid}/status').read_text()
                rollup = Path(f'/proc/{child.pid}/smaps_rollup').read_text()
                peak = max(peak, process_memory(status, rollup)['guarded_process_RSS_bytes'])
                fields = {x.split(':', 1)[0]: x.split(':', 1)[1] for x in status.splitlines() if ':' in x}
                threads = max(threads, int(fields['Threads'].strip()))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None or settled_after_counter_error(child): break
                if time.monotonic() - started <= 2:
                    time.sleep(.2); continue
                reason = 'owned_memory_counter_unavailable'; child.kill(); break
            reason = resources_ok(cfg, safety, ROOT, started, peak)
            if out.exists() and sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            if reason:
                if child.poll() is None: child.kill()
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None: child.kill()
        child.wait(timeout=10); raise
    complete = (out / 'result.json').is_file()
    atomic_json(terminal, {'owned_worker_exit_code': code, 'guard_stop_reason': reason,
        'successful_owned_execution': code == 0 and reason is None and complete,
        'complete_result_present': complete, 'only_new_owned_Popen_worker_managed': True,
        'peak_sampled_owned_worker_RSS_bytes': peak, 'peak_sampled_worker_threads': threads,
        'wall_seconds_including_imports_and_network': time.monotonic() - started, 'training_GPU_hours': 0})
    return 0 if code == 0 and reason is None and complete else 1


def launch(args, cfg, safety):
    proof, _ = source_gate(args, cfg, safety)
    out = ROOT / cfg['output_directory']; terminal = out.with_name(out.name + '_supervisor_attempt_1.json')
    receipt = out.with_name(out.name + '_persistent_launch.json'); log = out.with_name(out.name + '_persistent_stdout.log')
    if any(p.exists() for p in (out, terminal, receipt, log)): raise ValueError('Never relaunch a previous namespace')
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try: content = p.read_bytes()
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
        if os.fsencode(str(ROOT / SCRIPT)) + b'\0' in content and (b'--worker\0' in content or b'--supervisor\0' in content):
            raise ValueError('Observe the live M evaluator, do not duplicate it')
    if resources_ok(cfg, safety, ROOT, time.monotonic(), 0): raise RuntimeError('Pre-spawn resource guard')
    out.parent.mkdir(parents=True, exist_ok=True); cmd = command(args, '--supervisor')
    with log.open('xb', buffering=0) as stream:
        os.chmod(log, 0o600)
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
    try: ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError: ticks = None
    atomic_json(receipt, {'status': 'OWNED_SUPERVISOR_LAUNCHED_NOT_COMPLETED', 'supervisor_pid': child.pid,
        'linux_start_ticks': ticks, 'command_sha256': hashlib.sha256(b'\0'.join(os.fsencode(x) for x in cmd)).hexdigest(),
        'output_namespace': cfg['output_directory'], 'live_remote_registration_proof': proof,
        'start_new_session': True, 'normal_terminal_receipt': str(terminal.relative_to(ROOT)),
        'no_private_GT_inputs_or_foreign_process_changes': True, 'training_GPU_hours': 0})
    print(json.dumps({'status': 'LAUNCHED_NOT_COMPLETED', 'receipt': str(receipt.relative_to(ROOT))}), flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--worker', action='store_true'); mode.add_argument('--supervisor', action='store_true')
    p.add_argument('--model', choices=MODELS, required=True); p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    args = p.parse_args(); cfg = configuration(ROOT, args)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker: worker(args, cfg, safety); return 0
    if args.supervisor: return supervise(args, cfg, safety)
    return launch(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
