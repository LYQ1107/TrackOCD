"""Independent capped-prefix replay, no future tails or target-aware state.

Each p1/p2/p4/p8/p16 experiment is a separate chronological replay. WAIT is
uncommitted at that operating prefix, not a fabricated future-frame controller.
"""
from __future__ import annotations
from collections import defaultdict
import time
import numpy as np
import torch
from src.trackocd_core.baselines import make_baseline,anonymous_memory_size
from src.trackocd_core.features import PrefixView
from src.trackocd_core.evaluation import DecisionEvent,TrackKey,seal_decisions
from src.trackocd_core.persistent_policy import CandidateMemory,masked_logits,predicted_action


def causal_rows(routes,order,prefix):
    by_video=defaultdict(list)
    for r in routes:by_video[r['video_id']].append(r)
    if set(by_video)-set(order):raise ValueError('Stream outside registered order')
    return [r for v in order for r in sorted(by_video[v],
        key=lambda r:(r['frame_ids'][min(prefix,r['observation_count'])-1],r['physical_track_id']))]


def replay(routes,order,prefix,prototypes,thresholds,evidence,name='B1_track_nearest',
           decision_model=None,wait_bias=0.,allow_wait=True,reset_per_video=False,no_memory=False,
           known_ids=None):
    """Evidence maps keys to current-prefix arrays, not supervision rows."""
    model=CandidateMemory(prototypes) if decision_model is not None else make_baseline(name,prototypes,thresholds)
    events=[];counts=[];seconds=0.;previous=None;latencies=[]
    for r in causal_rows(routes,order,prefix):
        v=r['video_id'];n=min(prefix,r['observation_count']);item=evidence[r['key']]
        if (reset_per_video and previous!=v) or no_memory:
            if decision_model is not None:model.clear_anonymous()
            else:
                (model._centroids if name=='B2_track_dpmeans' else model._anonymous).clear()
        previous=v;started=time.perf_counter()
        if decision_model is not None:
            features,k,t=model.candidates(item['embedding'],item['uncertainty'],item['maturity'],item['quality'],item['elapsed'])
            with torch.inference_mode():logits=masked_logits(decision_model,torch.tensor(features))
            action=model.apply(predicted_action(logits,wait_bias,allow_wait),k,t,item['embedding'],item['quality'])
            size=len(model.anonymous)
        elif name=='B0_frame_snapshot_vote':
            # Same current descriptors, pre-track snapshot and one voted update.
            frame=item['frames'];view=PrefixView(frame,np.zeros((len(frame),4),np.float32),item['quality'],item['elapsed_frames'])
            action=model.step_prefix(view);size=anonymous_memory_size(model)
        else:action=model.step(item['embedding']);size=anonymous_memory_size(model)
        seconds+=time.perf_counter()-started
        kind=action['kind'];latencies.append(n if kind!='WAIT' else None)
        events.append(DecisionEvent(len(events),TrackKey(v,str(r['physical_track_id'])),n,kind,
            known_category_id=int(action['category_id']) if kind=='KNOWN' else None,
            token=action['token'] if kind in {'NEW','EXISTING'} else None));counts.append(size)
    sealed=seal_decisions(events,video_order=order,prefix_cap=prefix,known_ids=tuple(prototypes) if known_ids is None else known_ids)
    committed=[n for n in latencies if n is not None]
    return sealed,{'policy_inference_seconds':seconds,'policy_seconds_per_track':seconds/max(len(routes),1),
        'anonymous_memory_final':counts[-1] if counts else 0,'created_states':len(sealed.created_tokens),
        'wait_rate':sum(n is None for n in latencies)/max(len(routes),1),
        'commitment_coverage':len(committed)/max(len(routes),1),'mean_committed_observation_latency':float(np.mean(committed)) if committed else None,
        'wait_censored_at_prefix_not_zero_latency':True,'independent_capped_prefix_replay_not_dense_frame_online':True}
