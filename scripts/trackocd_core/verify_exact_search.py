#!/usr/bin/env python3
"""Compare every first4 irrevocable decision to prior frozen integration.

No Val metrics/targets read, no hyperparameter/model selection. Performance
change is accepted only with identical decisions (including tokens/order).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file
SOURCES=('src/trackocd_core/exact_search.py','src/trackocd_core/exact_replay.py','scripts/trackocd_core/verify_exact_search.py')


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);p.add_argument('--device',choices=('cpu','cuda'),required=True);a=p.parse_args()
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=a.preregistration_commit:raise ValueError('Exact performance source preregistration required')
    for n in SOURCES:
        if subprocess.check_output(['git','show',f'{a.preregistration_commit}:{n}'])!=(ROOT/n).read_bytes():raise ValueError('Source differs')
    gpu=None
    if a.device=='cuda':
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=10)
        choices=sorted([(int(line.split(',')[1]),line.split(',')[0].strip()) for line in rows.splitlines()],reverse=True)
        if choices[0][0]<12288:raise RuntimeError('Need12GiBfree for<=2GiBproof plus8GiBreserve; no foreign interference')
        gpu=choices[0][1];os.environ['CUDA_VISIBLE_DEVICES']=gpu
    import torch
    from scripts.trackocd_core.extract_limited_masa_features import inputs
    from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,evidence_bank,prototypes
    from src.trackocd_core.exact_replay import replay_stream
    from src.trackocd_core.persistent_policy import DecisionMLP
    torch.set_num_threads(1);started=time.monotonic();cfg,_,_=inputs();out=ROOT/cfg['output_directory']
    destination=out/f'exact_search_{a.device}_proof.json'
    if destination.exists():raise ValueError('Preserve actual proof; no overwrite')
    private=out/'integration_sealed_predictions.json';reference=json.loads(private.read_text());manifest=json.loads((out/'integration_manifest.json').read_text())
    train,labels,_=load_train(ROOT);known=json.loads((ROOT/'configs/trackocd_core/roles.json').read_text())['known_ids']
    proto_ids=sorted({r['category_id'] for r in labels if r['partition']=='prototype'})
    freeze=json.loads((ROOT/cfg['model_freeze']).read_text());policy=json.loads((ROOT/'outputs/trackocd_core/core_training/train_first_v1/policy/training_receipt.json').read_text())
    caches=[LimitedMasaVideoCache(out/'shards'/r['npz_filename'],r['video_id']) for r in manifest['records']]
    order=[c.video_id for c in caches]
    variants=[('B0_frame_snapshot_vote','A0_RAW',None),('B1_track_nearest','A0_RAW',None),('B2_track_dpmeans','A0_RAW',None),
        ('A1_SELECTED','A1_SELECTED',None),('A2_EVIDENCE','A2_EVIDENCE',None),('D1_SIMPLE_MLP','A1_SELECTED','D1_SIMPLE_MLP'),
        ('D2_RISK_AWARE','A1_SELECTED','D2_RISK_AWARE'),('FULL','A2_EVIDENCE','D2_RISK_AWARE')]
    proof=[]
    for name,rep,policy_name in variants:
        model,_=frozen_model(ROOT,rep,1027);proto=prototypes(train,labels,proto_ids,model)
        bank={c.video_id:evidence_bank(c,c.rows,model) for c in caches};lookup={c.video_id:c for c in caches}
        def videos(v,cap):
            c=lookup[v]
            for r in sorted(c.rows,key=lambda r:(r['frame_ids'][min(cap,r['observation_count'])-1],r['physical_track_id'])):
                yield r,bank[v][cap][r['key']]
        backend=name if rep=='A0_RAW' else 'B1_track_nearest'
        thresholds=next(c['thresholds'] for c in freeze['representation_thresholds'] if c['representation']==rep and c['seed']==(None if rep=='A0_RAW' else 1027) and c['backend']==backend)
        decision=None;bias=0.
        if policy_name:
            fit=next(f for f in policy['fits'] if (f['representation'],f['seed'],f['model'])==(rep,1027,policy_name))
            cp=fit['selected_checkpoint']['checkpoint'];decision=DecisionMLP().eval().requires_grad_(False)
            if sha256_file(ROOT/cp['path'])!=cp['sha256']:raise ValueError('Frozen policy differs')
            decision.load_state_dict(torch.load(ROOT/cp['path'],weights_only=True,map_location='cpu')['state_dict']);bias=fit['operating_points']['1.0']['wait_bias']
        for cap in (1,2,4,8,16):
            sealed,runtime=replay_stream(videos,order,cap,proto,thresholds,name=backend,decision_model=decision,wait_bias=bias,known_ids=known,device='cuda:0' if gpu else 'cpu')
            events=[{'sequence':e.sequence,'video_id':e.physical_key.video_id,'physical_id':e.physical_key.local_track_id,'observed':e.observed_prefix,
                     'kind':e.kind,'token':e.token,'known_id':e.known_category_id} for e in sealed.events]
            expected=next(r['events'] for r in reference if r['method']==name and r['prefix']==cap)
            if events!=expected:raise AssertionError(f'Performance executor decision mismatch {name}p{cap}; no metrics/retuning allowed')
            proof.append({'method':name,'prefix':cap,'decisions':len(events),'exact_equal':True,'runtime':runtime})
        del bank,model,decision
    result={'status':'PASS_EXACT_DECISION_EQUIVALENCE','device':a.device,'GPU_UUID':gpu,'preregistration_commit':a.preregistration_commit,
        'source_sha256':{n:sha256_file(ROOT/n) for n in SOURCES},'reference_seals_sha256':sha256_file(private),'cases':proof,
        'GT_or_metrics_read':False,'model_or_threshold_changes':False,'search_exhaustive':True,'TAO_Test_access':False,
        'resources':{'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved() if gpu else 0}}
    if gpu and (result['resources']['peak_GPU_reserved_bytes']>2*2**30 or torch.cuda.mem_get_info()[0]<8*2**30):raise RuntimeError('Proof resource bound')
    atomic_json(destination,result);atomic_json(ROOT/f'outputs/trackocd_core/EXACT_SEARCH_{a.device.upper()}_PROOF.json',result)
    print(json.dumps({'status':result['status'],'device':a.device,'cases':len(proof),'resources':result['resources']}),flush=True)


if __name__=='__main__':main()
