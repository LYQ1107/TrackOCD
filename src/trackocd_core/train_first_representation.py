"""Reuse existing adapter/evidence with balanced legal cross-video batches.

Supervision stays in sampling/loss/metrics, outside model forward. Variable
visible lengths are grouped, not padded or promoted to future observations.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.trackocd_core.representation import CategoryEvidence


class MeanCapacityEvidence(CategoryEvidence):
    """Same extra16833 parameters as temporal scorer; static prefix transform."""
    def __init__(self):
        super().__init__(False)
        self.static = nn.Sequential(nn.Linear(261,32),nn.GELU(),nn.Linear(32,256))
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self, visual, quality, elapsed):
        out=super().forward(visual,quality,elapsed)
        cues=torch.cat((out['embedding'],quality.mean(1,keepdim=True),out['uncertainty'][:,None],
                        out['effective_maturity'][:,None].log1p()/4,
                        elapsed.max(1).values[:,None].clamp(min=0).log1p()/10,
                        quality.std(1,unbiased=False,keepdim=True)),dim=-1)
        out['embedding']=F.normalize(out['embedding']+self.scale*self.static(cues),dim=-1)
        return out


def make_model(name):
    if name=='A2_CAPACITY_CONTROL':return MeanCapacityEvidence()
    if name in {'A1_ADAPTER','A1_GEOMETRY_1','A1_GEOMETRY_5','A2_EVIDENCE'}:
        return CategoryEvidence(name=='A2_EVIDENCE')
    raise ValueError('Unknown registered representation')


class BalancedBatchSampler:
    def __init__(self,rows,seed,categories_per_batch=8):
        self.groups=defaultdict(lambda:defaultdict(list));self.rng=np.random.default_rng(seed+11027)
        for r in rows:self.groups[r['category_id']][r['video_id']].append(r)
        self.categories=sorted(self.groups);self.count=categories_per_batch
        if len(self.categories)<self.count or any(len(vs)<2 for vs in self.groups.values()):
            raise ValueError('Each anchor needs legal cross-video positive and category negatives')
    def draw(self):
        rows=[]
        for c in self.rng.choice(self.categories,self.count,replace=False):
            for v in self.rng.choice(sorted(self.groups[c]),2,replace=False):
                candidates=self.groups[c][v];rows.append(candidates[int(self.rng.integers(len(candidates)))])
        return rows


def forward_views(model,views,device):
    groups=defaultdict(list);embeddings={}
    for i,v in enumerate(views):groups[len(v.visual)].append((i,v))
    for n,items in groups.items():
        arrays=[np.stack([getattr(v,field) for _,v in items]) for field in ('visual','quality','elapsed_frames')]
        result=model(*(torch.tensor(a,device=device,dtype=torch.float32) for a in arrays))
        for j,(i,v) in enumerate(items):embeddings[i]=result['embedding'][j]
    return torch.stack([embeddings[i] for i in range(len(views))])


def encode_rows(cache,rows,prefix,model=None,device='cpu'):
    if not rows:return np.empty((0,768 if model is None else 256),dtype=np.float32)
    vectors=[]
    with torch.inference_mode():
        for first in range(0,len(rows),64):
            views=[cache.get_prefix(r['key'],prefix) for r in rows[first:first+64]]
            vectors.extend(np.stack([v.weighted_mean() for v in views]) if model is None else forward_views(model,views,device).cpu().numpy())
    return np.asarray(vectors,dtype=np.float32)


def prototype_vectors(cache,rows,known_ids,model=None,device='cpu'):
    chosen=[r for r in rows if r['partition']=='prototype' and r['category_id'] in known_ids]
    values=encode_rows(cache,chosen,16,model,device);by=defaultdict(list)
    for r,value in zip(chosen,values):by[r['category_id']].append(value)
    result={}
    for c,vs in by.items():
        mean=np.mean(vs,axis=0);result[c]=mean/max(np.linalg.norm(mean),1e-12)
    if set(result)!=set(known_ids):raise ValueError('Missing registered legal episode Known prototype')
    return result


def representation_diagnostics(vectors,rows,prototypes):
    labels=np.asarray([r['category_id'] for r in rows]);videos=np.asarray([r['video_id'] for r in rows])
    cosine=vectors@vectors.T;allowed=videos[:,None]!=videos[None,:]
    same=labels[:,None]==labels[None,:];valid=(same&allowed).any(1)
    masked=np.where(allowed,cosine,-np.inf);rank1=labels[masked.argmax(1)]==labels
    per_category={str(c):float(rank1[(labels==c)&valid].mean()) for c in sorted(set(labels)) if ((labels==c)&valid).any()}
    similarities=vectors@np.stack(list(prototypes.values())).T;scores=similarities.max(1)
    known=np.asarray([r['simulation_role']=='known' for r in rows]);a,b=scores[known],scores[~known]
    auc=None if not len(a) or not len(b) else float(((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])).mean())
    positive=cosine[same&allowed];negative=cosine[~same&allowed]
    # Spectral effective rank is a representation diagnostic, not category ACC.
    eigen=np.maximum(np.linalg.eigvalsh(cosine.astype(np.float64)),0);prob=eigen/max(eigen.sum(),1e-12);nz=prob[prob>0]
    return {'cross_video_rank1_macro':float(np.mean(list(per_category.values()))) if per_category else None,
            'cross_video_rank1_per_category':per_category,'retrieval_queries_supported':int(valid.sum()),
            'retrieval_queries_without_cross_video_positive':int((~valid).sum()),
            'known_vs_pseudo_novel_auroc':auc,'known_tracks':int(known.sum()),'pseudo_novel_tracks':int((~known).sum()),
            'cross_video_same_category_cosine_mean':float(positive.mean()) if positive.size else None,
            'cross_video_different_category_cosine_mean':float(negative.mean()) if negative.size else None,
            'effective_spectral_rank':float(np.exp(-(nz*np.log(nz)).sum())),
            'prototype_max_score_mean_known':float(a.mean()) if len(a) else None,
            'prototype_max_score_mean_pseudo_novel':float(b.mean()) if len(b) else None,
            'pseudo_novel_wrong_known_at_fixed_point65':float((b>=.65).mean()) if len(b) else None}


def development_metrics(cache,labels,known_ids,model=None,device='cpu'):
    dev=[r for r in labels if r['partition']=='development']
    prototypes=prototype_vectors(cache,labels,known_ids,model,device)
    cases=[]
    for p in (1,2,4,8,16):
        vectors=encode_rows(cache,dev,p,model,device)
        cases.append({'prefix':p,**representation_diagnostics(vectors,dev,prototypes)})
    supported=[r for r in cases if r['cross_video_rank1_macro'] is not None and r['known_vs_pseudo_novel_auroc'] is not None]
    score=float(np.mean([.5*(r['cross_video_rank1_macro']+r['known_vs_pseudo_novel_auroc']) for r in supported])) if len(supported)==5 else None
    return {'partition':'development','cases':cases,'registered_checkpoint_score':score,'final_heldout_opened':False}
