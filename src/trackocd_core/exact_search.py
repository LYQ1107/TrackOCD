"""Performance-only exhaustive search with canonical NumPy score refinement.

No approximate-neighbor index, pruning by labels, new thresholds or changed
updates. FP32 GEMV screens every live centroid; a deliberately conservative
8*d*eps bound retains every potentially winning/tied candidate. Canonical
np.dot then determines scores/ties. Small memories use canonical dots directly.
Recovered baseline source is untouched. Matrix state has no GT/role access.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from src.trackocd_v2.methods.nearest_prototype import NearestPrototype, _normalize
from src.trackocd_v2.methods.dpmeans import OnlineDPMeans
from src.trackocd_core.baselines import FrameSnapshotNearestVoting
from src.trackocd_core.persistent_policy import CandidateMemory,unit


class ExhaustiveMatrix:
    def __init__(self,dimension,device='cpu',capacity=304561):
        self.dimension=dimension;self.device=device;self.capacity=capacity
        self.ids=[];self.lookup={};self.vectors=[];self.refined=0;self.queries=0
        self.matrix=None
        if device!='cpu':
            import torch
            torch.backends.cuda.matmul.allow_tf32=False
            self.matrix=torch.empty((capacity,dimension),dtype=torch.float32,device=device)
        else:self.matrix=np.empty((min(capacity,1024),dimension),dtype=np.float32)

    def clear(self):self.ids.clear();self.lookup.clear();self.vectors.clear()

    def update(self,key,vector):
        z=np.asarray(vector,dtype=np.float32)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.linalg.norm(z)>1.00001:
            raise ValueError('Exact screen requires finite canonical unit centroids')
        if key not in self.lookup:
            if len(self.ids)>=self.capacity:raise RuntimeError('Registered memory capacity exceeded')
            self.lookup[key]=len(self.ids);self.ids.append(key);self.vectors.append(z)
        row=self.lookup[key];self.vectors[row]=z
        if self.device=='cpu':
            if row>=len(self.matrix):
                value=np.empty((min(self.capacity,max(row+1,len(self.matrix)*2)),self.dimension),np.float32)
                value[:row]=self.matrix[:row];self.matrix=value
            self.matrix[row]=z
        else:
            import torch
            self.matrix[row].copy_(torch.from_numpy(z).to(self.device))

    def ranked(self,vector,k=1,tie='insertion',distance=False):
        self.queries+=1;n=len(self.ids)
        if n==0:return []
        if n<64:indices=np.arange(n)
        else:
            if self.device=='cpu':scores=self.matrix[:n]@vector
            else:
                import torch
                with torch.inference_mode():scores=(self.matrix[:n]@torch.from_numpy(vector).to(self.device)).cpu().numpy()
            # Scores from canonical dot and GEMV are unit-vector dot products.
            # Their FP32 accumulation error sum is far below this bound. All
            # near ties are refined, including exact float32 distance ties.
            tolerance=8*self.dimension*np.finfo(np.float32).eps
            cutoff=np.partition(scores,n-min(k,n))[n-min(k,n)]
            indices=np.flatnonzero(scores>=cutoff-tolerance)
        self.refined+=len(indices)
        scores=[(float(1.0-np.dot(vector,self.vectors[i])) if distance else
                 float(np.dot(vector,self.vectors[i])),int(i)) for i in indices]
        sign=1 if distance else -1
        scores.sort(key=lambda p:(sign*p[0],self.ids[p[1]] if tie=='identifier' else p[1]))
        return [(self.ids[i],score) for score,i in scores[:k]]


class ExactNearest(NearestPrototype):
    def __init__(self,*args,device='cpu',capacity=304561,**kwargs):
        super().__init__(*args,**kwargs)
        d=len(next(iter(self.known_prototypes.values())))
        self._search=ExhaustiveMatrix(d,device,capacity)
        self._known={int(k):_normalize(v) for k,v in self.known_prototypes.items()}

    def clear_anonymous(self):self._anonymous.clear();self._search.clear()

    def _update(self,token,vector):
        super()._update(token,vector);self._search.update(token,self._anonymous[token][0])

    def _decision(self,vector):
        z=_normalize(vector)
        known=max(((k,float(np.dot(z,v))) for k,v in self._known.items()),key=lambda a:a[1],default=(None,-1.))
        ranked=self._search.ranked(z);anon=ranked[0] if ranked else (None,-1.)
        if anon[0] is not None and anon[1]>=self.tau_existing and anon[1]>=known[1]:
            return {'kind':'EXISTING','token':anon[0],'score':anon[1]},z
        if known[0] is not None and known[1]>=self.tau_known:
            return {'kind':'KNOWN','token':f'K:{known[0]}','category_id':known[0],'score':known[1]},z
        return {'kind':'NEW','token':f'A:{self._next_id}','score':max(known[1],anon[1])},z

    def step(self,vector):
        action,z=self._decision(vector)
        if action['kind']=='EXISTING':self._update(action['token'],z)
        elif action['kind']=='NEW':
            self._next_id+=1;self._anonymous[action['token']]=(z,1);self._search.update(action['token'],z)
        return action


class ExactFrameVoting(ExactNearest):
    def step_prefix(self,view):
        votes,first=Counter(),{}
        for vector in view.visual:
            action,_=self._decision(vector)
            choice=(action['kind'],action.get('category_id') if action['kind']=='KNOWN' else action.get('token') if action['kind']=='EXISTING' else None)
            votes[choice]+=1;first.setdefault(choice,action)
        choice=max(votes,key=votes.get);action=dict(first[choice]);mean=view.weighted_mean()
        if action['kind']=='EXISTING':self._update(action['token'],mean)
        elif action['kind']=='NEW':
            token=f'A:{self._next_id}';self._next_id+=1;self._anonymous[token]=(mean,1)
            self._search.update(token,mean);action['token']=token
        action['winning_frame_votes']=votes[choice];return action


class ExactDPMeans(OnlineDPMeans):
    def __init__(self,*args,device='cpu',capacity=304561,**kwargs):
        super().__init__(*args,**kwargs)
        self._search=ExhaustiveMatrix(len(next(iter(self.known_prototypes.values()))),device,capacity)
        self._known={int(k):_normalize(v) for k,v in self.known_prototypes.items()}

    def clear_anonymous(self):self._centroids.clear();self._search.clear()

    def step(self,vector):
        z=_normalize(vector);ranked=self._search.ranked(z,distance=True)
        token,distance=ranked[0] if ranked else (None,1.)
        if token is not None and distance<=self.lambda_distance:
            old,n=self._centroids[token];new=_normalize((old*n+z)/(n+1))
            self._centroids[token]=(new,n+1);self._search.update(token,new)
            return {'kind':'EXISTING','token':token,'distance':distance}
        known=min(((k,float(1.0-np.dot(z,v))) for k,v in self._known.items()),key=lambda a:a[1],default=(None,1.))
        if known[0] is not None and known[1]<=self.tau_known:
            return {'kind':'KNOWN','token':f'K:{known[0]}','category_id':known[0],'distance':known[1]}
        token=f'A:{self._next_id}';self._next_id+=1;self._centroids[token]=(z,1);self._search.update(token,z)
        return {'kind':'NEW','token':token,'distance':min(distance,1.)}


class ExactCandidateMemory(CandidateMemory):
    def __init__(self,known_prototypes,device='cpu',capacity=304561):
        super().__init__(known_prototypes)
        self._search=ExhaustiveMatrix(len(next(iter(self.known.values()))),device,capacity)

    def clear_anonymous(self):super().clear_anonymous();self._search.clear()

    def candidates(self,evidence,uncertainty,maturity,quality,elapsed):
        z=unit(evidence)
        known=sorted(((float(z@v),k) for k,v in self.known.items()),key=lambda a:(-a[0],a[1]))
        anonymous=self._search.ranked(z,k=2,tie='identifier')
        k1,k2=(known[0][0] if known else -1),(known[1][0] if len(known)>1 else -1)
        a1,a2=(anonymous[0][1] if anonymous else -1),(anonymous[1][1] if len(anonymous)>1 else -1)
        known_id=known[0][1] if known else None;token=anonymous[0][0] if anonymous else None
        member=self.anonymous.get(token);q=np.asarray(quality,dtype=np.float32)
        cues=np.asarray([k1,k2,k1-k2,a1,a2,a1-a2,k1-a1,float(uncertainty),
            np.log1p(maturity)/4,float(q.mean()),float(q.std()),np.log1p(max(elapsed,0))/10,
            member.dispersion if member else 0,np.log1p(member.count)/8 if member else 0,
            member.reliability if member else 0,float(bool(known)),float(bool(anonymous))],dtype=np.float32)
        if not np.isfinite(cues).all():raise ValueError('Nonfinite causal candidates')
        return cues,known_id,token

    def apply(self,*args,**kwargs):
        action=super().apply(*args,**kwargs)
        if action['kind'] in {'NEW','EXISTING'}:self._search.update(action['token'],self.anonymous[action['token']].prototype)
        return action


def make_exact_baseline(name,prototypes,thresholds,device='cpu'):
    if name=='B2_track_dpmeans':return ExactDPMeans(known_prototypes=prototypes,tau_known=thresholds['dpmeans_known_distance_threshold'],lambda_distance=thresholds['dpmeans_lambda_distance'],device=device)
    cls=ExactFrameVoting if name=='B0_frame_snapshot_vote' else ExactNearest if name=='B1_track_nearest' else None
    if cls is None:raise ValueError('No registered baseline')
    return cls(known_prototypes=prototypes,tau_known=thresholds['known_cosine_threshold'],tau_existing=thresholds['existing_cosine_threshold'],device=device)
