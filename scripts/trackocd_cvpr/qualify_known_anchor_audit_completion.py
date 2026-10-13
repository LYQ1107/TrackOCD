#!/usr/bin/env python3
"""One PUBLIC-only exit0/full-result counter exception, never rewrite terminal."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import sha256_file,atomic_json
from src.trackocd_cvpr.remote_registration import verify_registration
from scripts.trackocd_cvpr.audit_known_anchor_opportunities import SOURCES as AUDIT_SOURCES

NS='outputs/trackocd_cvpr/cvpr3/Known_anchor_opportunity_audit_v1'
RESULT=NS+'/result.json'; LAUNCH=NS+'/launch.json'; TERMINAL=NS+'_supervisor_attempt_1.json'
OUTPUT=NS+'_counter_completion_qualification_v1.json'
SCRIPT='scripts/trackocd_cvpr/qualify_known_anchor_audit_completion.py'
HASHES={RESULT:'b5c868c4eb5396616292a9cc762607a607a65086d3531972904e7eb5929a46b4',
    LAUNCH:'1f366ad67bb660f3d30e37e92623b56510905a87ff4fbbf3c1436a1ee245127b',
    TERMINAL:'ed36aede312c0f4f46a7e21df41d9d2b91ef64a8eb4deb958ad7fe3bbbadc5af'}
SOURCES=tuple(dict.fromkeys((*AUDIT_SOURCES,SCRIPT,*HASHES)))


def validate(root):
    for path,digest in HASHES.items():
        if (root/path).is_symlink() or sha256_file(root/path)!=digest:
            raise ValueError('Exact one original audit receipt, no substituted exception')
    r=json.loads((root/RESULT).read_text()); t=json.loads((root/TERMINAL).read_text())
    l=json.loads((root/LAUNCH).read_text())
    if (t['owned_worker_exit_code']!=0 or t['guard_stop_reason']!='owned_memory_counter_unavailable'
            or t['successful_owned_execution'] or not t['complete_result_present']
            or not t['only_new_owned_Popen_worker_managed'] or t['training_GPU_hours']!=0
            or t['peak_sampled_owned_worker_RSS_bytes']>3221225472
            or t['wall_seconds_including_worker_imports_network']>3000
            or r['resources']['peak_guarded_RSS_bytes']>3221225472
            or r['resources']['wall_seconds_excluding_network']>3000 or r['resources']['CUDA_initialized']
            or r['optimizer_steps']!=0 or not r['actual_original_calendar_exact']
            or r['GT_bank_repair_new_nonKnown_values_or_dev_final_Val_Test_used']
            or r['status']!='ACTUAL_READONLY_KNOWN_ANCHOR_OPPORTUNITIES_NOT_TRAINING_OR_NEW_CALENDAR_SELECTION'):
        raise ValueError('Only complete exit0 read-only audit/counter-observation exception under original bounds')
    counts=r['analysis']['scope_counts']
    if (counts['metadata_frames'],counts['empty_frames'],counts['physical_observations'],counts['selected_update_frames'])!=(374,1,1268,50):
        raise ValueError('Complete original clock and selected update accounting')
    if r['analysis']['original_first50_known_calendar_sha256']!='8a3ee08efb2718c96d186156e5ae6ccb3a8cb86a66928b8a8f21838a78db9bc6':
        raise ValueError('Exact actual original pilot schedule')
    for row in (r,l):
        if row['live_remote_registration_proof']['sha']!='83c97a5a118078d549bb428b18b53624fe175b82':
            raise ValueError('Actual original audit registration, not selected rerun')
    for path in AUDIT_SOURCES:
        if sha256_file(root/path)!=l['source_sha256'].get(path): raise ValueError('Executed audit source changed')
    return counts


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preregistration-commit',required=True)
    p.add_argument('--git-proxy',default=''); args=p.parse_args()
    if (ROOT/OUTPUT).exists(): raise ValueError('Preserve every qualification')
    proof=verify_registration(ROOT,'research/cvpr-trackocd-autonomous-v2',args.preregistration_commit,args.git_proxy,allow_public_api=True)
    for path in SOURCES:
        if subprocess.check_output(['git','show',args.preregistration_commit+':'+path],cwd=ROOT)!=(ROOT/path).read_bytes():
            raise ValueError('Exact public qualification source/result registration')
    counts=validate(ROOT)
    atomic_json(ROOT/OUTPUT,{'status':'QUALIFIED_COMPLETE_READONLY_AUDIT_WITH_DECLARED_COUNTER_OBSERVATION_EXCEPTION',
        'live_remote_registration_proof':proof,'artifact_sha256':HASHES,'scope_counts':counts,
        'original_supervisor_successful_owned_execution':False,'original_terminal_rewritten':False,
        'effective_complete_worker_qualification':True,'counter_failure_cause_proven':False,
        'no_dataset_GT_weights_or_private_history_read':True,'new_training_model_or_calendar_selection':False})
    print(json.dumps({'status':'QUALIFIED_DECLARED_EXCEPTION_NOT_NORMAL_SUPERVISOR','output':OUTPUT}))


if __name__=='__main__': main()
