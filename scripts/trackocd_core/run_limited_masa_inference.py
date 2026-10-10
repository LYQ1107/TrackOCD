#!/usr/bin/env python3
"""All-ID frozen semantic inference; owned bounded workers, zero ValGT access.

Completed ledgers resumable by exact input/source identity. No worker restarts
after failure, no overwrite of partial attempts or unknown-ID filtering.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
CONFIG=ROOT/'configs/trackocd_core/limited_masa_evaluation.json'
SOURCES=('configs/trackocd_core/limited_masa_evaluation.json','scripts/trackocd_core/run_limited_masa_inference.py',
    'src/trackocd_core/limited_masa_plan.py','src/trackocd_core/limited_masa_bank.py','src/trackocd_core/sealed_parquet.py',
    'src/trackocd_core/exact_search.py','src/trackocd_core/exact_replay.py')


def inputs():
    from scripts.trackocd_core.extract_limited_masa_features import inputs as feature_inputs
    cfg=json.loads(CONFIG.read_text());fc,_,_=feature_inputs();out=ROOT/fc['output_directory']
    manifest_path=out/'full_manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='COMPLETE_FULL_LIMITED_MASA_ALL_ID_PREFIX_FEATURES' or (manifest['videos'],manifest['tracks'],manifest['observations'])!=(988,304561,1294110):raise ValueError('Complete frozen all-ID features required')
    if manifest['config_sha256']!=sha256_file(ROOT/cfg['feature_config']) or manifest['model_freeze_sha256']!=sha256_file(ROOT/fc['model_freeze']):raise ValueError('Feature configuration/freeze lineage differs')
    for n,sha in manifest['source_sha256'].items():
        if sha256_file(ROOT/n)!=sha:raise ValueError('Feature encoding source changed')
    for device in ('CPU','CUDA'):
        proof=json.loads((ROOT/f'outputs/trackocd_core/EXACT_SEARCH_{device}_PROOF.json').read_text())
        if proof['status']!='PASS_EXACT_DECISION_EQUIVALENCE' or len(proof['cases'])!=40:raise ValueError('Actual exact-seal CPU andCUDA proof required')
        for n,sha in proof['source_sha256'].items():
            if sha256_file(ROOT/n)!=sha:raise ValueError('Proven executor source changed')
    return cfg,fc,manifest,{'config_sha256':sha256_file(CONFIG),'feature_manifest_sha256':sha256_file(manifest_path),
        'model_freeze_sha256':sha256_file(ROOT/fc['model_freeze']),'source_sha256':{n:sha256_file(ROOT/n) for n in SOURCES}}


def ram():return {k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())}


def guard(cfg,started):
    m=ram()
    if m['MemAvailable']<m['MemTotal']*cfg['system_RAM_reserve_fraction'] or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>cfg['host_planned_peak_per_worker_bytes']:raise RuntimeError('Owned semantic RAM guard')
    if time.monotonic()-started>cfg['stage_wall_seconds'] or os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize<cfg['minimum_disk_free_bytes']:raise RuntimeError('Owned semantic wall/disk guard')


def inference_barrier():
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','socket.sendto','subprocess.Popen','os.system'}:raise PermissionError('Semantic inference has no network/child work')
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0])
            if ('/TAO-Amodal/annotations/' in p or '/frames/test/' in p or 'evaluator_only_geometry_join' in p or 'physical_cross_video_support' in p or p.endswith('/case_metrics.json')):raise PermissionError('Semantic inference cannot read ValGT/matches/metrics/Test')
    sys.addaudithook(audit)


def completed_case(out,job,identity):
    marker=out/'cases'/job['id']/'sealed_complete.json'
    if not marker.exists():return None
    d=json.loads(marker.read_text());filename=Path(d['ledger']['filename'])
    if str(filename)!=filename.name or marker.is_symlink():raise ValueError('Unexpected seal path/link')
    p=marker.parent/filename
    if d['job']!=job or d['input_identity']!=identity or d['status']!='SEALED_COMPLETE_ALL_PREDICTED_IDS' or p.is_symlink() or p.stat().st_size!=d['ledger']['bytes'] or sha256_file(p)!=d['ledger']['sha256']:raise ValueError('Invalid completed seal; never replace by favorable rerun')
    return d


def worker(assignment_path,group):
    import numpy as np
    import torch
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,prototypes,evidence_bank
    from src.trackocd_core.limited_masa_bank import FrozenVideoBank
    from src.trackocd_core.exact_replay import replay_stream
    from src.trackocd_core.persistent_policy import DecisionMLP
    from src.trackocd_core.sealed_parquet import write_sealed
    torch.set_num_threads(1);started=time.monotonic();cfg,fc,manifest,identity=inputs();assignment=json.loads(Path(assignment_path).read_text())
    if assignment['identity']!=identity:raise ValueError('Registered worker identity changed')
    plan_path=ROOT/cfg['output_directory']/'plan.json'
    if sha256_file(plan_path)!=assignment['plan_sha256']:raise ValueError('Exact prospective jobplan identity differs')
    plan=json.loads(plan_path.read_text());jobs=[plan['jobs'][i] for i in plan['groups'][group]]
    job=jobs[0];rep=job['representation'];seed=job['seed'];norel=bool(job['options'].get('no_reliability'))
    cache,labels,_=load_train(ROOT);proto_ids=sorted({r['category_id'] for r in labels if r['partition']=='prototype'})
    known=json.loads((ROOT/'configs/trackocd_core/roles.json').read_text())['known_ids']
    if (len(proto_ids),len(known))!=(48,78):raise ValueError('Legal prototype gap may not be repaired with ValGT')
    model,lineage=frozen_model(ROOT,rep,seed)
    if norel:
        by={r['key']:r for r in labels};rows=[r for r in cache.rows if by[r['key']]['partition']=='prototype']
        b=evidence_bank(cache,rows,model,True)[16];values=defaultdict(list)
        for r in rows:values[by[r['key']]['category_id']].append(b[r['key']]['embedding'])
        proto={c:np.mean(v,axis=0) for c,v in values.items()};proto={c:v/max(np.linalg.norm(v),1e-12) for c,v in proto.items()};del b,values
    else:proto=prototypes(cache,labels,proto_ids,model)
    del cache,labels
    policy=json.loads((ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    decisions={};checkpoints={}
    for p in {j['policy'] for j in jobs if j['policy']}:
        fit=next(f for f in policy['fits'] if (f['representation'],f['seed'],f['model'])==(rep,seed,p));cp=fit['selected_checkpoint']['checkpoint']
        if sha256_file(ROOT/cp['path'])!=cp['sha256']:raise ValueError('Policy checkpoint differs')
        d=DecisionMLP().eval().requires_grad_(False);d.load_state_dict(torch.load(ROOT/cp['path'],map_location='cpu',weights_only=True)['state_dict'],strict=True)
        decisions[p]=d;checkpoints[p]=cp
    del policy
    inference_barrier();out=ROOT/cfg['output_directory'];completed=[]
    def bank_progress(video,tracks):
        guard(cfg,started);atomic_json(out/f'{group}_progress.json',{'stage':'frozen_CPU_FP32_bank','last_video':video,'tracks':tracks,'seconds':time.monotonic()-started})
    bank=FrozenVideoBank(ROOT/fc['output_directory']/'shards',manifest['records'],model,norel,bank_progress)
    for job in jobs:
        existing=completed_case(out,job,identity)
        if existing is not None:completed.append(job['id']);continue
        case_start=time.monotonic();directory=out/'cases'/job['id'];directory.mkdir(parents=True,exist_ok=True)
        def progress(video,tracks,states):
            guard(cfg,started)
            if time.monotonic()-case_start>cfg['case_wall_seconds'] or torch.cuda.max_memory_reserved()>cfg['GPU_planned_peak_bytes'] or torch.cuda.mem_get_info()[0]<cfg['GPU_free_reserve_bytes']:raise RuntimeError('Owned case GPU/wall guard')
            atomic_json(out/f'{group}_progress.json',{'stage':'all_ID_semantic_replay','execution_id':job['id'],'completed_executions':len(completed),'last_video':video,'tracks':tracks,'states':states,'case_seconds':time.monotonic()-case_start})
        options={k:v for k,v in job['options'].items() if k!='no_reliability'}
        sealed,runtime=replay_stream(bank.videos,plan['orders'][job['order']],job['prefix'],proto,job['thresholds'],name=job['backend'],
            decision_model=decisions.get(job['policy']),wait_bias=job['wait_bias'],known_ids=known,device='cuda:0',progress=progress,**options)
        if len(sealed.events)!=304561:raise AssertionError('All predictedIDs includingshort/unmatched mandatory')
        payload=directory/f"sealed_{assignment['attempt']}.parquet";write_sealed(payload,sealed)
        d={'status':'SEALED_COMPLETE_ALL_PREDICTED_IDS','job':job,'input_identity':identity,'physical_tracks':len(sealed.events),
            'ledger':{'filename':payload.name,'bytes':payload.stat().st_size,'sha256':sha256_file(payload)},'runtime':runtime,
            'model_lineage':lineage,'policy_checkpoint':checkpoints.get(job['policy']),'bank_vector_sha256':bank.sha256,'bank_scalar_sha256':bank.scalar_sha256,
            'prototypes':{'available':len(proto),'inherited_known':len(known),'ValGTfill':False},'GT_or_Val_metrics_read':False,'TAO_Test_access':False,'new_optimization':False,
            'resources':{'wall_seconds':time.monotonic()-case_start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved()}}
        atomic_json(directory/'sealed_complete.json',d);completed.append(job['id']);del sealed
        print(json.dumps({'group':group,'completed':len(completed),'executions':len(jobs),'case_seconds':d['resources']['wall_seconds']}),flush=True)
    bank.close();atomic_json(out/f'{group}_{assignment["attempt"]}_done.json',{'group':group,'completed':completed,'wall_seconds':time.monotonic()-started,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved(),'GPU_UUID':os.environ['CUDA_VISIBLE_DEVICES']})


def supervise(commit):
    from src.trackocd_core.limited_masa_plan import make_plan
    from src.trackocd_core.limited_masa_features import completed_feature
    cfg,fc,manifest,identity=inputs();out=ROOT/cfg['output_directory'];started=time.monotonic()
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=commit:raise ValueError('Exact remote full semantic source registration required')
    for n in SOURCES:
        if subprocess.check_output(['git','show',f'{commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Source differs from exact registered remote')
    if (out/'full_inference_manifest.json').exists():raise ValueError('Preserve completed full inference')
    for r in manifest['records']:
        if completed_feature(ROOT/fc['output_directory']/'shards',r['video_id'],manifest['config_sha256'],r['input_npz_sha256'])!=r:raise ValueError('Every fullfeature atomic record/payload must agree')
    freeze=json.loads((ROOT/fc['model_freeze']).read_text());policy=json.loads((ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    plan=make_plan(cfg,freeze,policy,[r['video_id'] for r in manifest['records']]);out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    plan_path=out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Cannot alter prospective job plan')
    if not plan_path.exists():atomic_json(plan_path,plan)
    attempt=uuid.uuid4().hex;assignment_path=out/f'assignment_{attempt}.json';assignment={'attempt':attempt,'preregistration_commit':commit,'identity':identity,'plan_sha256':sha256_file(plan_path)};atomic_json(assignment_path,assignment)
    m=ram();limit=min(cfg['max_concurrent_workers'],int((m['MemAvailable']-m['MemTotal']*.25)//cfg['host_planned_peak_per_worker_bytes']))
    if limit<1:raise RuntimeError('Wait for safe RAM; never killforeign processes')
    pending=sorted(plan['groups'],key=lambda g:-len(plan['groups'][g]));active=[];children=[];logs=[];done=[];error=None
    try:
        while pending or active:
            guard(cfg,started)
            for p,g,gpu in list(active):
                code=p.poll()
                if code is not None:
                    if code!=0:raise RuntimeError(f'Owned semantic group failed {g}; retain attempts')
                    done.append(g);active.remove((p,g,gpu))
            if pending and len(active)<limit:
                rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
                occupied={g for p,n,g in active};choices=[]
                for line in rows.splitlines():
                    gpu,free,util=[s.strip() for s in line.split(',')]
                    if gpu not in occupied and int(free)*2**20>=cfg['GPU_planned_peak_bytes']+cfg['GPU_free_reserve_bytes']:choices.append((int(util)==0,int(free),gpu))
                m=ram()
                if choices and m['MemAvailable']-cfg['host_planned_peak_per_worker_bytes']>=m['MemTotal']*.25:
                    gpu=sorted(choices,reverse=True)[0][2];group=pending.pop(0);log=(out/f'{group}_{attempt}.log').open('x');logs.append(log)
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
                    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker-group',group,'--assignment',str(assignment_path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append((p,group,gpu));children.append(p)
                    print(json.dumps({'group':group,'GPU_UUID':gpu,'active_workers':len(active),'pending_groups':len(pending)}),flush=True)
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>cfg['new_stage_disk_ceiling_bytes']:raise RuntimeError('Private stage storage guard')
            time.sleep(5)
        records=[completed_case(out,j,identity) for j in plan['jobs'].values()]
        if any(r is None for r in records):raise ValueError('Every registered all-ID execution must complete')
        result={'status':'COMPLETE_FROZEN_FULL_LIMITED_MASA_SEMANTIC_PREDICTIONS_NOT_YET_METRICS','preregistration_commit':commit,'identity':identity,'plan_sha256':sha256_file(plan_path),
            'unique_executions':len(records),'logical_cases':len(plan['logical_cases']),'records':records,'all988videos':True,'all304561physicalIDs_each_execution':True,
            'GT_or_Val_metrics_in_inference':False,'TAO_Test_access':False,'foreign_interference':False,'wall_seconds':time.monotonic()-started}
        atomic_json(out/'full_inference_manifest.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','identity')}),flush=True)
    except Exception as exc:
        error=repr(exc)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
        raise
    finally:
        for log in logs:log.close()
        atomic_json(out/f'supervisor_{attempt}.json',{'error':error,'completed_groups':done,'owned_child_returncodes':[p.returncode for p in children],'foreign_process_interference':False,'wall_seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit');p.add_argument('--worker-group');p.add_argument('--assignment');a=p.parse_args()
    if a.worker_group:worker(a.assignment,a.worker_group)
    elif a.preregistration_commit:supervise(a.preregistration_commit)
    else:p.error('Choose exact-remote parent or owned group')
