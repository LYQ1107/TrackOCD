#!/usr/bin/env python3
"""Frozen-result-only role retrieval/bootstrap audit, never model selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.trackocd_v2.io import atomic_json,sha256_file


def role_retrieval(vectors,routes,labels):
    import numpy as np
    by={r['key']:r for r in labels};rows=[by[r['key']] for r in routes]
    c=np.asarray([r['category_id'] for r in rows]);v=np.asarray([r['video_id'] for r in routes]);roles=np.asarray([r['simulation_role'] for r in rows])
    allowed=v[:,None]!=v[None,:];same=c[:,None]==c[None,:];valid=(same&allowed).any(1)
    order=np.argsort(-np.where(allowed,vectors@vectors.T,-np.inf),axis=1);result={}
    for role in ('known','pseudo_novel'):
        mask=roles==role;records={}
        for k in (1,5,10):
            hits=np.take_along_axis(same&allowed,order[:,:k],axis=1).any(1)
            per={str(category):float(hits[(c==category)&mask&valid].mean()) for category in sorted(set(c[mask])) if ((c==category)&mask&valid).any()}
            records[f'recall_at_{k}_category_macro']=float(np.mean(list(per.values()))) if per else None
            records[f'recall_at_{k}_per_category']=per
        result[role]={'query_tracks':int(mask.sum()),'queries_with_cross_video_positive':int((mask&valid).sum()),
            'unsupported_queries':int((mask&~valid).sum()),'gallery':'all heldout Known and pseudo-Novel except same video',**records}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--preregistration-commit',required=True);args=p.parse_args()
    source='scripts/trackocd_core/analyze_train_representation.py'
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:17890','ls-remote','origin','refs/heads/codex/trackocd-core-training-limited-masa'],text=True,timeout=25).split()[0]
    if remote!=args.preregistration_commit or subprocess.check_output(['git','show',f'{args.preregistration_commit}:{source}'])!=(ROOT/source).read_bytes():raise ValueError('Exact registered analysis source required')
    output=ROOT/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_ROLE_ANALYSIS.json'
    if output.exists():raise ValueError('Preserve analysis; no heldout redesign or retry')
    import numpy as np
    import torch
    from src.trackocd_core.train_first_experiment import load_train,frozen_model,evidence_bank
    from src.trackocd_core.scientific_statistics import paired_video_bootstrap
    torch.set_num_threads(1);cache,labels,known=load_train(ROOT);by={r['key']:r for r in labels}
    routes=[r for r in cache.rows if by[r['key']]['partition']=='heldout_selection']
    heldout_path=ROOT/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT.json';heldout=json.loads(heldout_path.read_text())
    analyses=[];paired=[]
    for representation in ('A0_RAW','A1_SELECTED','A2_EVIDENCE','A2_CAPACITY_CONTROL'):
        for seed in ([None] if representation=='A0_RAW' else [1027,1028,1029]):
            model,lineage=frozen_model(ROOT,representation,seed);bank=evidence_bank(cache,routes,model)
            for prefix in (1,2,4,8,16):
                vectors=np.stack([bank[prefix][r['key']]['embedding'] for r in routes])
                analyses.append({'representation':representation,'seed':seed,'prefix':prefix,'roles':role_retrieval(vectors,routes,labels),'lineage':lineage})
            del model,bank
    for left,right in (('A0_RAW','A1_SELECTED'),('A1_SELECTED','A2_EVIDENCE'),('A2_CAPACITY_CONTROL','A2_EVIDENCE')):
        for seed in (1027,1028,1029):
            for order in heldout['orders']:
                a,b=[next(r for r in heldout['cases'] if (r['representation'],r['backend'],r['seed'],r['order'],r['prefix'])==
                    (name,'B1_track_nearest',None if name=='A0_RAW' else seed,order,16)) for name in (left,right)]
                paired.append({'left':left,'right':right,'seed':seed,'order':order,'prefix':16,
                    'conditional_video_bootstrap':paired_video_bootstrap(a['errors']['per_video_conditional_fixed_mapping_counts'],b['errors']['per_video_conditional_fixed_mapping_counts'])})
    result={'status':'COMPLETE_FROZEN_TRAIN_RESULT_ROLE_DIAGNOSTICS_NOT_NEW_TRIAL',
        'preregistration_commit':args.preregistration_commit,'source_sha256':sha256_file(ROOT/source),'parent_heldout_sha256':sha256_file(heldout_path),
        'analyses':analyses,'paired_p16':paired,'model_optimizer_or_selection':False,'new_threshold_trial':False,'val_or_test_access':False,
        'posthoc_diagnostics_do_not_repair_or_replay_memory':True}
    atomic_json(output,result);print(json.dumps({'status':result['status'],'p16_roles':[r for r in analyses if r['prefix']==16]}),flush=True)


if __name__=='__main__':main()
