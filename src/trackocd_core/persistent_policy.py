"""Small same-capacity policies with strictly prediction-owned visual memory.

No category/role/GT table is accepted by CandidateMemory or DecisionMLP.
Action supervision and absorbing purity records live in an external trainer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

ACTIONS = ('KNOWN', 'EXISTING', 'NEW', 'WAIT')
FEATURE_COUNT = 17


def unit(value):
    value=np.asarray(value,dtype=np.float32);norm=float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm<=1e-12:raise ValueError('Invalid evidence')
    return value/norm


@dataclass
class VisualMember:
    vector_sum: np.ndarray
    count: int
    quality_sum: float
    @property
    def prototype(self):return unit(self.vector_sum)
    @property
    def dispersion(self):return float(np.clip(1-np.linalg.norm(self.vector_sum)/self.count,0,1))
    @property
    def reliability(self):return self.quality_sum/self.count*(1-self.dispersion)


class CandidateMemory:
    def __init__(self,known_prototypes):
        self.known={int(k):unit(v) for k,v in known_prototypes.items()}
        self.anonymous={};self.next_id=0

    def clear_anonymous(self):
        # Tokens are never recycled, even for reset-per-video ablations.
        self.anonymous.clear()

    def candidates(self,evidence,uncertainty,maturity,quality,elapsed):
        z=unit(evidence)
        known=sorted(((float(z@v),k) for k,v in self.known.items()),key=lambda a:(-a[0],a[1]))
        anonymous=sorted(((float(z@m.prototype),k) for k,m in self.anonymous.items()),key=lambda a:(-a[0],a[1]))
        k1,k2=(known[0][0] if known else -1),(known[1][0] if len(known)>1 else -1)
        a1,a2=(anonymous[0][0] if anonymous else -1),(anonymous[1][0] if len(anonymous)>1 else -1)
        known_id=known[0][1] if known else None;token=anonymous[0][1] if anonymous else None
        member=self.anonymous.get(token)
        q=np.asarray(quality,dtype=np.float32)
        cues=np.asarray([k1,k2,k1-k2,a1,a2,a1-a2,k1-a1,float(uncertainty),
            np.log1p(maturity)/4,float(q.mean()),float(q.std()),np.log1p(max(elapsed,0))/10,
            member.dispersion if member else 0,np.log1p(member.count)/8 if member else 0,
            member.reliability if member else 0,float(bool(known)),float(bool(anonymous))],dtype=np.float32)
        if not np.isfinite(cues).all():raise ValueError('Nonfinite causal candidates')
        return cues,known_id,token

    def apply(self,action,known_id,token,evidence,quality):
        kind=ACTIONS[int(action)];z=unit(evidence);q=float(np.mean(quality))
        if kind=='KNOWN':
            if known_id not in self.known:raise ValueError('Invalid Known ID')
            return {'kind':kind,'category_id':known_id}
        if kind=='EXISTING':
            if token not in self.anonymous:raise ValueError('Future or absent anonymous token')
            m=self.anonymous[token];m.vector_sum=m.vector_sum+z;m.count+=1;m.quality_sum+=q
        elif kind=='NEW':
            token=f'S:{self.next_id}';self.next_id+=1
            self.anonymous[token]=VisualMember(z.copy(),1,q)
        else:return {'kind':'WAIT'}
        return {'kind':kind,'token':token}


class DecisionMLP(nn.Module):
    """D1/D2 share initialization,17 inputs,32 hidden units,708 parameters."""
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(FEATURE_COUNT,32),nn.GELU(),nn.Linear(32,4))
    def forward(self,features):return self.network(features)


def masked_logits(model,features):
    logits=model(features)
    mask=torch.stack((features[...,15]>0,features[...,16]>0,
                      torch.ones_like(features[...,15],dtype=torch.bool),
                      torch.ones_like(features[...,15],dtype=torch.bool)),dim=-1)
    return logits.masked_fill(~mask,-1e9)


def predicted_action(logits,wait_bias=0.,allow_wait=True):
    score=logits.detach().clone();score[...,3]+=wait_bias
    if not allow_wait:score[...,3]=-1e9
    return int(score.argmax(-1).item())


class SupervisionHistory:
    """Trainer/evaluator ONLY; cannot become CandidateMemory input/state."""
    def __init__(self):self.members={};self.last_write={}
    def update(self,prediction,category,sequence):
        if prediction['kind'] in {'NEW','EXISTING'}:
            token=prediction['token'];self.members.setdefault(token,[]).append(category);self.last_write[token]=sequence
    def pure(self,token,category):
        members=self.members.get(token,[])
        return bool(members) and all(c==category for c in members)
    def costs(self,category,is_known,known_id,token,actual_prefix,constants):
        present=any(category in values for values in self.members.values())
        wrong_known=not is_known or known_id!=category
        wrong_existing=is_known or not self.pure(token,category)
        costs=np.asarray([constants['wrong_known'] if wrong_known else 0,
            constants['wrong_existing_pollution'] if wrong_existing else 0,
            constants['known_as_new'] if is_known else constants['wrong_new_fragmentation'] if present else 0,
            constants['wait_base']+constants['wait_per_observation']*actual_prefix],dtype=np.float32)
        # Nominal D1 supervision does not train on magnitude of sequential risk.
        if is_known:target=0 if known_id==category else 3
        else:target=1 if self.pure(token,category) else 3 if present else 2
        polluted=token is not None and len(set(self.members.get(token,[])))>1
        return costs,target,self.last_write.get(token) if polluted else None
