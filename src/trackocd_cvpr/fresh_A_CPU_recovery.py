"""Exactly one new namespace after actual thread-guard failure, no resume."""
import json
import subprocess
import os
import re
from pathlib import Path
from src.trackocd_v2.io import sha256_file

CONFIG='configs/trackocd_cvpr/fresh_A_cpu_visibility_retry1.json'
ORIGINAL='outputs/trackocd_cvpr/cvpr2/fresh_A_fit5_optimizer_pilot_v1'
RETRY=ORIGINAL+'_CPU_visibility_retry1'
TERMINAL=ORIGINAL+'_supervisor_attempt_1.json'
SOURCES=(CONFIG,'src/trackocd_cvpr/fresh_A_CPU_recovery.py',TERMINAL,ORIGINAL+'/launch.json')
NUMERIC_SOURCES=('configs/trackocd_cvpr/fresh_A_fit5_baselines_v1.json',
    'configs/trackocd_cvpr/fresh_A_optimizer_pilot_v1.json',
    'src/trackocd_cvpr/fresh_A_baseline.py','src/trackocd_cvpr/fresh_A_training.py')


def verify_original(root):
    root=Path(root); cfg=json.loads((root/CONFIG).read_text())
    if (cfg['status']!='REGISTERED_NAMESPACE_ONLY_CPU_VISIBILITY_RECOVERY_ACTUAL_PILOT_PENDING'
            or cfg['original_namespace']!=ORIGINAL or cfg['new_namespace']!=RETRY
            or cfg['original_preregistration_commit']!='bcc2044d88250eb5d5b3bc5544374c5cbdc2b697'
            or cfg['allowed_original_guard']!='owned_worker_threads'
            or cfg['CPU_compute_workers']!=1 or cfg['native_driver_helper_threads_ceiling']!=1
            or cfg['native_driver_helper_CPU_seconds_ceiling']!=.1 or cfg['total_OS_threads_ceiling']!=2
            or cfg['infinite_retry_or_training_success_claim']):
        raise ValueError('Frozen single CPU-visibility recovery only')
    terminal=root/TERMINAL; launch=root/ORIGINAL/'launch.json'
    if (terminal.is_symlink() or launch.is_symlink()
            or sha256_file(terminal)!=cfg['original_terminal_sha256']
            or sha256_file(launch)!=cfg['original_launch_sha256']):
        raise ValueError('Preserve exact original terminal/launch')
    t=json.loads(terminal.read_text()); l=json.loads(launch.read_text())
    if (t['owned_worker_exit_code']!=-9 or t['guard_stop_reason']!='owned_worker_threads'
            or t['successful_owned_execution'] or t['full_result_present']
            or not t['only_new_owned_Popen_worker_managed'] or t['peak_sampled_worker_threads']!=2
            or t['training_GPU_hours']!=0 or l['preregistration_commit']!=cfg['original_preregistration_commit']
            or l['live_remote_registration_proof']['sha']!=cfg['original_preregistration_commit']
            or {p.name for p in (root/ORIGINAL).iterdir()}!={'launch.json'}):
        raise ValueError('Actual thread failure with launch only, no partial results/weights')
    for p in NUMERIC_SOURCES:
        if (root/p).is_symlink() or subprocess.check_output(['git','show',cfg['original_preregistration_commit']+':'+p],cwd=root)!=(root/p).read_bytes():
            raise ValueError('Frozen scientific/numeric protocol unchanged')
    return cfg


def recovery_configuration(root,cfg,attempt):
    if attempt is None: return cfg
    if attempt!='cpu_visibility_retry1' or cfg['output_directory']!=ORIGINAL:
        raise ValueError('Only new namespace, not repeated or different recovery')
    verify_original(root)
    return {**cfg,'output_directory':RETRY,'CPU_visibility_recovery':attempt}


def validate_thread_inventory(records,main_pid,clock_ticks_per_second):
    """One main compute worker + at most one bounded identified driver helper.

    This fixes OS-thread != compute-worker classification, not a generic
    larger pool. Helper can briefly run during driver startup/shutdown; its
    accumulated CPU time must stay <=100ms. Every other native thread refuses.
    """
    main=[r for r in records if r['tid']==main_pid]
    helpers=[r for r in records if r['tid']!=main_pid]
    if len(main)!=1 or len(helpers)>1 or clock_ticks_per_second<=0:
        raise ValueError('One CPU worker, at most one native driver helper')
    for r in helpers:
        if (re.fullmatch(r'cuda[0-9a-f]{11}',r['name']) is None
                or r['CPU_ticks']<0 or r['CPU_ticks']/clock_ticks_per_second>.1):
            raise ValueError('Unknown/autograd/CPU-consuming helper thread refused')
    return len(helpers)


def inspect_owned_threads(pid):
    records=[]
    for p in Path(f'/proc/{pid}/task').iterdir():
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        records.append({'tid':int(p.name),'name':(p/'comm').read_text().strip(),
                        'CPU_ticks':int(fields[11])+int(fields[12])})
    return validate_thread_inventory(records,pid,os.sysconf('SC_CLK_TCK'))
