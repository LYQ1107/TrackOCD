#!/usr/bin/env python3
"""One gated fresh pair per seed, four full policy episodes, fixed200 endpoint."""
import argparse
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
from src.trackocd_cvpr.spread_C_formal_protocol import configuration, CONFIG
from src.trackocd_cvpr.spread_C_pilot_qualification import HASHES
from scripts.trackocd_cvpr import pilot_spread_known_rejection as pilot

SCRIPT = 'scripts/trackocd_cvpr/fit_spread_known_fixed200.py'
SOURCES = tuple(dict.fromkeys((*pilot.SOURCES, CONFIG, SCRIPT,
    'src/trackocd_cvpr/spread_C_formal_protocol.py',
    'src/trackocd_cvpr/spread_C_pilot_qualification.py',
    'configs/trackocd_cvpr/autonomous_research.json', *HASHES)))


def source_gate(args, cfg, safety):
    formal = configuration(ROOT, args.seed)
    if any(cfg[k] != v for k, v in formal.items()):
        raise ValueError('Exact independent formal invocation')
    proof = verify_registration(ROOT, safety['branch'], args.preregistration_commit,
                                args.git_proxy, allow_public_api=True)
    for path in SOURCES:
        if ((ROOT/path).is_symlink() or subprocess.check_output(
                ['git', 'show', args.preregistration_commit+':'+path], cwd=ROOT) != (ROOT/path).read_bytes()):
            raise ValueError('Every qualified source/public prerequisite registered before private input')
    authority = json.loads((ROOT/cfg['approval']).read_text())
    if authority['status'] != 'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN':
        raise ValueError('Real frozen-scope training authority')
    qualified = ROOT/'outputs/trackocd_cvpr/cvpr2/policy_training_input_qualification_v1'
    t = json.loads(qualified.with_name(qualified.name+'_supervisor_attempt_1.json').read_text())
    if (sha256_file(qualified/'result.json') != cfg['policy_input_qualification_sha256']
            or not t['successful_owned_execution'] or t['owned_worker_exit_code'] != 0
            or t['guard_stop_reason'] is not None):
        raise ValueError('Exact complete normal full-policy input qualification')
    return proof


def busy():
    if pilot.busy():
        return True
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) == os.getpid():
            continue
        try:
            words = p.read_bytes().split(b'\0')
            state = (p.parent/'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if os.fsencode(str(ROOT/SCRIPT)) in words and state not in ('Z', 'X', 'x'):
            return True
    return False


def supervise(args, cfg, safety):
    out = ROOT/cfg['output_directory']
    terminal = out.with_name(out.name+'_supervisor_attempt_1.json')
    if out.exists() or terminal.exists():
        raise ValueError('Single fresh seed pair; preserve all existing trials')
    if busy():
        raise RuntimeError('Observe existing owned research work; no parallel fit')
    source_gate(args, cfg, safety)
    start = time.monotonic(); peak = threads = helpers = 0; reason = None
    mem = {k: int(v.split()[0])*1024 for k, v in
           (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())}
    if (pilot.resources_ok(cfg, safety, ROOT, start, 0)
            or mem['MemAvailable'] < safety['system_ram_reserve_fraction']*mem['MemTotal']+cfg['rss_bytes_ceiling']):
        raise RuntimeError('Pre-spawn resource reserve plus whole planned RSS')
    child = subprocess.Popen([sys.executable, '-B', '-X', 'pycache_prefix='+str(out/'_fresh_no_write_cache'),
        str(ROOT/SCRIPT), '--worker', '--seed', str(args.seed), '--preregistration-commit',
        args.preregistration_commit, '--git-proxy', args.git_proxy], cwd=ROOT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    try:
        while child.poll() is None:
            p = Path(f'/proc/{child.pid}/status')
            try:
                status = p.read_text()
                peak = max(peak, pilot.process_memory(status,
                    Path(f'/proc/{child.pid}/smaps_rollup').read_text())['guarded_process_RSS_bytes'])
                fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
                threads = max(threads, int(fields['Threads']))
            except (FileNotFoundError, ProcessLookupError, RuntimeError):
                if child.poll() is not None:
                    break
                try:
                    settled = pilot.kernel_terminal_status(p.read_text())
                except (FileNotFoundError, ProcessLookupError):
                    settled = False
                if settled:
                    child.wait(timeout=1); break
                if time.monotonic()-start > 2:
                    reason = 'owned_memory_counter_unavailable'; child.kill(); break
                time.sleep(.2); continue
            reason = pilot.resources_ok(cfg, safety, ROOT, start, peak)
            if pilot.output_bytes(out) > cfg['output_bytes_ceiling']:
                reason = 'owned_output_size'
            try:
                helpers = max(helpers, pilot.inspect_owned_threads(child.pid))
            except ValueError:
                reason = 'owned_worker_threads'
            except (FileNotFoundError, ProcessLookupError):
                if child.poll() is not None:
                    break
                reason = 'owned_thread_inventory_unavailable'
            if reason:
                if child.poll() is None:
                    child.kill()  # Only this newly created owned child.
                break
            time.sleep(.2)
        code = child.wait(timeout=10)
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10); raise
    complete = (out/'result.json').is_file()
    atomic_json(terminal, {'owned_worker_exit_code':code, 'guard_stop_reason':reason,
        'successful_owned_execution':code == 0 and reason is None and complete,
        'complete_result_present':complete, 'peak_sampled_owned_worker_RSS_bytes':peak,
        'peak_sampled_worker_threads':threads, 'peak_identified_native_driver_helpers':helpers,
        'CPU_compute_workers':1, 'training_GPU_hours':0, 'only_new_owned_Popen_worker_managed':True,
        'wall_seconds_including_worker_imports_network':time.monotonic()-start})
    return 0 if code == 0 and reason is None and complete else 1


def worker(args, cfg, safety):
    out = ROOT/cfg['output_directory']
    if (not sys.dont_write_bytecode or sys.pycache_prefix != str(out/'_fresh_no_write_cache')
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        raise ValueError('Fresh hidden-GPU no-write CPU worker')
    proof = source_gate(args, cfg, safety)
    import torch
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from src.trackocd_cvpr.R_development_evaluation import known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    from src.trackocd_cvpr.paired_spread_known_rejection_training import paired_candidate_fit
    from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
    from src.trackocd_cvpr.candidate_score_contract import CandidateScoreMLP
    from src.trackocd_cvpr.fresh_A_training import checkpoint_state_equal
    from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def barrier(event, values):
        if event == 'socket.connect':
            raise PermissionError('No network after registration')
        if event == 'open' and isinstance(values[0], (str, bytes)):
            p = Path(os.fsdecode(values[0])).resolve()
            if (p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal'))
                    or (p.suffix in ('.pt', '.pth', '.partial') and p.parent != out)):
                raise PermissionError('Qualified caches/own fresh checkpoints only, never pilot or old weights')
    sys.addaudithook(barrier)
    if out.exists():
        raise ValueError('No output overwrite')
    out.mkdir(parents=True); start = time.monotonic(); peak = 0
    prior = sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*')
                if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak = max(peak, pilot.process_memory()['guarded_process_RSS_bytes'])
        size = pilot.output_bytes(out); reason = pilot.resources_ok(cfg, safety, ROOT, start, peak)
        if reason or size > cfg['output_bytes_ceiling'] or prior+size > safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Owned fresh paired200 guard:'+str(reason or 'output_storage'))
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or torch.cuda.is_initialized():
            raise RuntimeError('One CPU compute worker/no CUDA')
    atomic_json(out/'launch.json', {'preregistration_commit':args.preregistration_commit,
        'live_remote_registration_proof':proof, 'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},
        'scope':'Fresh same-seed pair200/four full policy episodes/fixed offline Known-rank plan; no dev auto-start'})
    try:
        guard(); videos, projection, policy = load_completed_policy_videos(ROOT, cfg, guard)
        fit, fit_proof = load_completed_fit_tracks(ROOT, cfg, guard)
        raw, _ = known_prototypes(RawPreservingEvidence().eval().requires_grad_(False), fit)
        known = {i:v for i, v in enumerate(raw)}; saved = []
        def save(arm, model, optimizer, step, episode, episode_step, rng):
            path = out/f'private_{arm}_step{step}.pt'; tmp = path.with_suffix('.partial')
            with tmp.open('xb') as f:
                os.chmod(tmp, 0o600)
                torch.save({'model_state':model.state_dict(), 'optimizer_state':optimizer.state_dict(),
                    'arm':arm, 'seed':args.seed, 'step':step, 'episode':episode,
                    'episode_step':episode_step, 'numpy_rng_state':rng,
                    'torch_rng_state':torch.get_rng_state()}, f)
                f.flush(); os.fsync(f.fileno())
            if path.exists():
                raise ValueError('Preserve private checkpoint')
            os.replace(tmp, path); guard(); value = torch.load(path, map_location='cpu', weights_only=True)
            with torch.random.fork_rng(devices=[]):
                verifier = CandidateScoreMLP()
            verifier.load_state_dict(value['model_state'], strict=True)
            if (digest(verifier) != digest(model) or value['arm'] != arm or value['step'] != step
                    or value['seed'] != args.seed or value['episode'] != episode
                    or value['episode_step'] != episode_step
                    or not checkpoint_state_equal(value['optimizer_state'], optimizer.state_dict())
                    or value['numpy_rng_state'] != rng
                    or not torch.equal(value['torch_rng_state'], torch.get_rng_state())):
                raise ValueError('Exact actual fresh model/AdamW/RNG checkpoint reopen')
            saved.append({'arm':arm, 'step':step, 'sha256':sha256_file(path),
                          'bytes':path.stat().st_size, 'reopen_exact':True})
        def progress(rows, curves):
            atomic_json(out/'progress.json', rows); atomic_json(out/'loss_curves.json', curves)
        frozen = freeze_calendar(videos, projection, args.seed, cfg['episodes'],
                                 cfg['updates_per_episode_each'], guard)
        atomic_json(out/'private_frozen_calendar.json', frozen)
        os.chmod(out/'private_frozen_calendar.json', 0o600)
        _, _, summary, private = paired_candidate_fit(videos, projection, known, args.seed,
            episodes=cfg['episodes'], updates_per_episode=cfg['updates_per_episode_each'],
            checkpoint_steps=cfg['checkpoint_steps'], guard=guard, progress=progress,
            checkpoint=save, repair_coefficient=cfg['repair_coefficient'], frozen_calendar=frozen)
        if (summary['optimizer_steps_each'] != 200 or len(saved) != 8
                or any((e['videos'], e['metadata_frames'], e['physical_observations'], e['empty_frames'])
                       != (10, 374, 1268, 1) for episodes in summary['episodes'].values() for e in episodes)
                or any([c['step'] for c in saved if c['arm'] == arm] != cfg['checkpoint_steps']
                       for arm in cfg['arms'])):
            raise ValueError('Both full four-episode clocks/fixed200/all eight real snapshots required')
        result = {'status':'COMPLETED_FRESH_SPREAD_PAIRED_FIXED200_NOT_DEV_CALIBRATION_OR_GOAL_ACCEPTANCE',
            'live_remote_registration_proof':proof, 'training_seed':args.seed,
            'policy_scope':policy, 'offline_fit_scope':fit_proof, 'summary':summary,
            'private_checkpoints':saved, 'checkpoint_selection':cfg['checkpoint_selection'],
            'pilot_or_historical_checkpoint_warmstart':False, 'dev_final_Val_Test_inputs_used':False,
            'raw_private_history_or_GT_exported':False,
            'formal6fits_development_calibration_or_Goal_accepted':False,
            'resources':{'peak_guarded_RSS_bytes':peak, 'wall_seconds_excluding_network':time.monotonic()-start,
                         'training_GPU_hours':0, 'CUDA_initialized':torch.cuda.is_initialized()}}
        guard(); atomic_json(out/'result.json', result)
        print(json.dumps({'status':result['status'], 'seed':args.seed,
            'extra_gradient_steps':sum(r['rejection_parameter_gradient_norm'] > 0
                                     for r in summary['loss_curves']['Known_set_rejection'])}), flush=True)
    except Exception as error:
        atomic_json(out/'failure.json', {'exception_type':type(error).__name__, 'error':str(error), 'complete':False})
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True)
    p.add_argument('--git-proxy', default=''); p.add_argument('--worker', action='store_true')
    p.add_argument('--seed', type=int, choices=(1027, 1028, 1029), required=True); args = p.parse_args()
    formal = configuration(ROOT, args.seed)
    cfg = {**pilot.configuration(ROOT), **formal}
    safety = json.loads((ROOT/'configs/trackocd_cvpr/autonomous_research.json').read_text())
    if args.worker:
        worker(args, cfg, safety); return 0
    return supervise(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
