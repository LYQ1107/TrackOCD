#!/usr/bin/env python3
"""Registered read-only fit/policy opportunity audit, not a retry or new fit."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from scripts.trackocd_cvpr import pilot_paired_known_rejection as base
from src.trackocd_v2.io import atomic_json,sha256_file
from src.trackocd_cvpr.known_anchor_opportunities import audit_opportunities

SCRIPT='scripts/trackocd_cvpr/audit_known_anchor_opportunities.py'
OUT='outputs/trackocd_cvpr/cvpr3/Known_anchor_opportunity_audit_v1'
PILOT='outputs/trackocd_cvpr/cvpr3/paired_Known_set_rejection_pilot50_v1'
PILOT_SHA='9a67853e3070dec7d86f7b7b0b7638e9f68b48d374950bcd490692f6a58a8239'
SOURCES=tuple(dict.fromkeys((*base.SOURCES,SCRIPT,'src/trackocd_cvpr/known_anchor_opportunities.py',
    PILOT+'/launch.json',PILOT+'/result.json',PILOT+'_supervisor_attempt_1.json')))


def worker(args,cfg,safety):
    out=ROOT/OUT
    if not sys.dont_write_bytecode or sys.pycache_prefix!=str(out/'_fresh_no_write_cache') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Same bounded own no-write CPU worker')
    proof=base.source_gate(args,cfg,safety)
    if sha256_file(ROOT/PILOT/'result.json')!=PILOT_SHA: raise ValueError('Actual uninformed pilot bound before assets')
    pilot=json.loads((ROOT/PILOT/'result.json').read_text())
    terminal=json.loads((ROOT/(PILOT+'_supervisor_attempt_1.json')).read_text())
    if not terminal['successful_owned_execution'] or pilot['actual_nonzero_rejection_gradient']:
        raise ValueError('Exact recorded exposure gap, not altered successful intervention')
    import torch
    from src.trackocd_cvpr.policy_training_data import load_completed_policy_videos
    from src.trackocd_cvpr.fit_R_training import load_completed_fit_tracks
    from src.trackocd_cvpr.R_development_evaluation import known_prototypes
    from src.trackocd_cvpr.raw_preserving_evidence import RawPreservingEvidence
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    def barrier(event,values):
        if event=='socket.connect': raise PermissionError('No network after registration')
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).resolve()
            if p.is_relative_to(Path('/data3/liuyeqiang/TAO-Amodal')) or p.suffix in ('.pt','.pth'):
                raise PermissionError('Qualified cache only; no old/pilot weights/original labels/pixels')
    sys.addaudithook(barrier)
    if out.exists(): raise ValueError('No previous output overwrite')
    out.mkdir(parents=True); start=time.monotonic(); peak=0
    prior=sum(p.stat().st_size for p in (ROOT/'outputs/trackocd_cvpr').rglob('*') if p.is_file() and not p.is_symlink())
    def guard():
        nonlocal peak
        peak=max(peak,base.process_memory()['guarded_process_RSS_bytes']); size=base.output_bytes(out)
        reason=base.resources_ok(cfg,safety,ROOT,start,peak)
        if reason or size>cfg['output_bytes_ceiling'] or prior+size>safety['incremental_storage_soft_bytes']:
            raise RuntimeError('Own read-only opportunity audit guard:'+str(reason or 'storage'))
        if torch.cuda.is_initialized() or torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:
            raise RuntimeError('One CPU compute worker/no CUDA')
    atomic_json(out/'launch.json',{'live_remote_registration_proof':proof,
        'source_sha256':{p:sha256_file(ROOT/p) for p in SOURCES},'scope':'Read-only original policy frame opportunities, no fit'})
    try:
        guard(); videos,targets,scope=load_completed_policy_videos(ROOT,cfg,guard)
        fit,fit_scope=load_completed_fit_tracks(ROOT,cfg,guard)
        raw,_=known_prototypes(RawPreservingEvidence().eval().requires_grad_(False),fit)
        audit=audit_opportunities(videos,targets,{i:r for i,r in enumerate(raw)},guard)
        if audit['original_first50_known_calendar_sha256']!=pilot['summary']['actual_shared_physical_calendar_sha256']:
            raise ValueError('Original actual pilot update calendar must match before interpretation')
        counts=audit['scope_counts']
        if (counts['metadata_frames'],counts['empty_frames'],counts['physical_observations'])!=(374,1,1268):
            raise ValueError('Complete fixed physical scope')
        result={'status':'ACTUAL_READONLY_KNOWN_ANCHOR_OPPORTUNITIES_NOT_TRAINING_OR_NEW_CALENDAR_SELECTION',
            'live_remote_registration_proof':proof,'policy_scope':scope,'offline_fit_scope':fit_scope,
            'analysis':audit,'original_pilot_result_sha256':PILOT_SHA,'actual_original_calendar_exact':True,
            'optimizer_steps':0,'GT_bank_repair_new_nonKnown_values_or_dev_final_Val_Test_used':False,
            'resources':{'peak_guarded_RSS_bytes':peak,'wall_seconds_excluding_network':time.monotonic()-start,
                'training_GPU_hours':0,'CUDA_initialized':torch.cuda.is_initialized()}}
        guard(); atomic_json(out/'result.json',result); print(json.dumps({'status':result['status'],'counts':counts}),flush=True)
    except Exception as e:
        atomic_json(out/'failure.json',{'exception_type':type(e).__name__,'error':str(e),'complete':False}); raise


def main():
    base.SCRIPT,base.OUT,base.SOURCES,base.worker=SCRIPT,OUT,SOURCES,worker
    return base.main()


if __name__=='__main__': sys.exit(main())
