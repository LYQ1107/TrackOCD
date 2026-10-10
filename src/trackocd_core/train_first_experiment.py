"""Train-only frozen-model bank and evaluator helpers; no policy GT inputs."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from src.trackocd_core.train_first_cache import TrainFirstCache
from src.trackocd_core.train_first_representation import make_model,prototype_vectors,representation_diagnostics
from src.trackocd_core.evaluation import Target,TrackKey,join_evaluation,evaluate_standard,evaluate_persistent
from src.trackocd_v2.io import sha256_file


def load_train(root):
    from scripts.trackocd_core.extract_train_first import inputs
    split,plan,visual=inputs();cache=TrainFirstCache(root/split['cache_directory'])
    if cache.manifest['selection_plan_sha256']!=sha256_file(root/split['output_directory']/'selection_plan.json'):
        raise ValueError('Split registration changed')
    if cache.manifest['config_sha256']!=sha256_file(root/'configs/trackocd_core/training_split.json'):
        raise ValueError('Feature configuration differs')
    source=pq.read_table(root/split['cache_directory']/'train_labels.parquet').to_pylist()
    registered={r['key']:r for r in plan['rows']};routes={r['key']:r for r in cache.rows}
    if set(r['key'] for r in source)!=set(registered):raise ValueError('Supervision universe differs')
    for r in source:
        if any(r[k]!=registered[r['key']][k] for k in ('category_id','simulation_role','partition')):raise ValueError('Role differs')
    labels=[{**r,'video_id':routes[r['key']]['video_id']} for r in source]
    known=sorted({r['category_id'] for r in labels if r['partition']=='representation_fit'})
    return cache,labels,known


def frozen_model(root,name,seed):
    if name=='A0_RAW':return None,None
    phase='evidence' if name.startswith('A2') else 'representation'
    path=root/f'outputs/trackocd_core/core_training/train_first_v1/{phase}/training_receipt.json'
    receipt=json.loads(path.read_text())
    if receipt['config_sha256']!=sha256_file(root/f'configs/trackocd_core/{"evidence" if phase=="evidence" else "representation"}_training.json'):
        raise ValueError('Completed training configuration differs')
    for source,sha in receipt['source_sha256'].items():
        if sha256_file(root/source)!=sha:raise ValueError('Training source changed after freeze')
    if name=='A1_SELECTED':name=receipt['shared_geometry_family']
    fit=next(f for f in receipt['fits'] if f['model']==name and f['seed']==seed)
    cp=fit['selected_checkpoint']['checkpoint'];p=root/cp['path']
    if p.stat().st_size!=cp['bytes'] or sha256_file(p)!=cp['sha256']:raise ValueError('Checkpoint changed')
    saved=torch.load(p,map_location='cpu',weights_only=True)
    if saved['feature_manifest_sha256']!=receipt['feature_manifest_sha256']:raise ValueError('Checkpoint feature lineage differs')
    model=make_model(name).eval().requires_grad_(False);model.load_state_dict(saved['state_dict'],strict=True)
    return model,{'training_receipt_sha256':sha256_file(path),'checkpoint':cp,'training_model':name,'step':saved['step']}


def evidence_bank(cache,routes,model=None,no_reliability=False):
    """Only visible visual inputs enter frozen forward. Bounded in-memory bank."""
    bank={}
    for cap in (1,2,4,8,16):
        items={}
        with torch.inference_mode():
            for first in range(0,len(routes),64):
                chunk=routes[first:first+64];views=[cache.get_prefix(r['key'],cap) for r in chunk];groups=defaultdict(list)
                for i,v in enumerate(views):groups[len(v.visual)].append((i,v))
                for n,group in groups.items():
                    if model is None:
                        outs=[{'embedding':v.weighted_mean(),'maturity':float(n),
                               'uncertainty':float(np.clip(1-(v.visual@v.weighted_mean()).mean(),0,1)),
                               'frames':v.visual} for _,v in group]
                    else:
                        visual=torch.tensor(np.stack([v.visual for _,v in group]));q=torch.tensor(np.stack([v.quality for _,v in group]))
                        elapsed=torch.tensor(np.stack([v.elapsed_frames for _,v in group]),dtype=torch.float32)
                        if no_reliability:
                            semantic=model.adapter(visual);weights=q.clamp(min=1e-6);weights=weights/weights.sum(1,keepdim=True)
                            embedding=torch.nn.functional.normalize((semantic*weights[...,None]).sum(1),dim=-1)
                            uncertainty=(1-(semantic*embedding[:,None]).sum(-1).mul(weights).sum(1)).clamp(0,1)
                            maturity=1/weights.square().sum(1)
                        else:
                            result=model(visual,q,elapsed);embedding=result['embedding'];uncertainty=result['uncertainty'];maturity=result['effective_maturity']
                        outs=[{'embedding':embedding[j].numpy(),'uncertainty':float(uncertainty[j]),
                               'maturity':float(maturity[j])} for j in range(len(group))]
                    for (i,v),out in zip(group,outs):
                        items[chunk[i]['key']]={**out,'quality':v.quality,'elapsed':float(v.elapsed_frames[-1]),
                            'elapsed_frames':v.elapsed_frames,'actual_prefix':len(v.visual)}
        bank[cap]=items
    return bank


def prototypes(cache,labels,known,model=None):return prototype_vectors(cache,labels,known,model)


def evaluate(sealed,routes,labels):
    by={r['key']:r for r in labels}
    targets=[Target(TrackKey(r['video_id'],str(r['physical_track_id'])),by[r['key']]['category_id'],
        'known' if by[r['key']]['simulation_role']=='known' else 'novel') for r in routes]
    join=join_evaluation(sealed,targets,{t.key:t.key for t in targets})
    standard=evaluate_standard(join);standard.pop('global_anonymous_hungarian_mapping_evaluator_only')
    persistent=evaluate_persistent(join);commits={c.event.physical_key:c.event for c in sealed.commits}
    histories=defaultdict(list);contamination=0;wrong_known=0;novel=0
    target_by={t.key:t for t in targets}
    for c in sealed.commits:
        e=c.event;t=target_by[e.physical_key]
        if e.kind in {'NEW','EXISTING'}:
            old=histories[e.token];contamination+=bool(old) and any(v!=t.category_id for v in old)
            old.append(t.category_id)
    for t in targets:
        if t.role=='novel':novel+=1;wrong_known+=commits.get(t.key) is not None and commits[t.key].kind=='KNOWN'
    extra={'all_novel_wrong_known_count':wrong_known,'all_novel_wrong_known_denominator':novel,
        'all_novel_wrong_known_rate':wrong_known/novel if novel else None,
        'memory_contamination_write_events':contamination,
        'contaminated_states':sum(len(set(v))>1 for v in histories.values()),
        'GT_controlled_identity_join_not_predicted_track_result':True}
    return {'standard':standard,'persistent':persistent,'errors':extra}


def diagnostics(bank,routes,labels,proto,prefix):
    by={r['key']:r for r in labels};rows=[{**by[r['key']],'video_id':r['video_id']} for r in routes]
    vectors=np.stack([bank[prefix][r['key']]['embedding'] for r in routes])
    result=representation_diagnostics(vectors,rows,proto)
    sim=vectors@vectors.T;videos=np.asarray([r['video_id'] for r in routes]);categories=np.asarray([r['category_id'] for r in rows])
    allowed=videos[:,None]!=videos[None,:];same=categories[:,None]==categories[None,:];valid=(same&allowed).any(1)
    order=np.argsort(-np.where(allowed,sim,-np.inf),axis=1)
    for k in (1,5,10):
        hit=np.take_along_axis(same&allowed,order[:,:k],axis=1).any(1)
        supported=[float(hit[(categories==c)&valid].mean()) for c in sorted(set(categories)) if ((categories==c)&valid).any()]
        result[f'cross_video_recall_at_{k}_macro']=float(np.mean(supported)) if supported else None
    return result
