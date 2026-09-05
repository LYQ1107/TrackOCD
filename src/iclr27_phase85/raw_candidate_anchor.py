"""One deterministic causal raw-top-K implementation shared by TRAIN and events."""
from __future__ import annotations
import numpy as np
from typing import Optional

def normalize(v: np.ndarray) -> np.ndarray:
    v=np.asarray(v,np.float32)
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)

def stable_raw_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Return stable descending score indices; ties retain source order."""
    s=np.asarray(scores,np.float32).reshape(-1)
    if s.size==0:return np.empty((0,),np.int64)
    k=max(0,min(int(k),int(s.size)))
    return np.argsort(-s,kind='stable')[:k].astype(np.int64)

def set_context(raw_scores: np.ndarray, original_count: int, source_length: float=0.0,
                source_variance: float=0.0, history_quality: float=0.0) -> np.ndarray:
    """Causal set-level statistics; no labels, IDs, or future rows."""
    s=np.asarray(raw_scores,np.float32).reshape(-1)
    if s.size:
        order=stable_raw_topk(s, min(2,s.size)); best=float(s[order[0]])
        second=float(s[order[1]]) if len(order)>1 else best
        z=s-np.max(s); q=np.exp(np.clip(z,-30,30)); q/=max(float(q.sum()),1e-8)
        ent=float(-(q*np.log(np.maximum(q,1e-8))).sum()/max(np.log(max(len(s),2)),1e-8))
        vals=[best,second,best-second,float(s.mean()),float(s.std()),ent,
              min(float(original_count),256.0)/256.0,min(float(source_length),256.0)/256.0,
              min(max(float(source_variance),0.0),1.0),min(max(float(history_quality),0.0),1.0)]
    else:
        vals=[0.0]*10
    return np.asarray(vals,np.float32)

def candidate_features(candidate_vectors: np.ndarray, source_mean: np.ndarray,
                       source_prototypes: Optional[np.ndarray]=None) -> tuple[np.ndarray,np.ndarray]:
    z=normalize(np.asarray(candidate_vectors,np.float32)); src=normalize(np.asarray(source_mean,np.float32))
    raw=z@src
    prot=np.asarray(source_prototypes if source_prototypes is not None else np.empty((0,z.shape[-1])),np.float32)
    prot=normalize(prot) if prot.size else np.empty((0,z.shape[-1]),np.float32)
    if prot.size:
        pm=z@prot.T; extra=np.stack([raw,pm.max(1),pm.mean(1),pm.min(1)],axis=1)
    else: extra=np.stack([raw,raw,raw,raw],axis=1)
    return extra.astype(np.float32),raw.astype(np.float32)
