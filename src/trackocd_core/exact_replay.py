"""Registered exhaustive-search executor, same one-decision capped replay.

Evidence is supplied one video at a time, not all prefixes materialized in RAM.
No GT labels, geometry matches, evaluator state or semantic names accepted.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from src.trackocd_core.exact_search import make_exact_baseline,ExactCandidateMemory
from src.trackocd_core.evaluation import DecisionEvent,TrackKey,seal_decisions
from src.trackocd_core.features import PrefixView
from src.trackocd_core.persistent_policy import masked_logits,predicted_action


def replay_stream(videos,order,prefix,prototypes,thresholds,*,name='B1_track_nearest',
                  decision_model=None,wait_bias=0.,allow_wait=True,
                  reset_per_video=False,no_memory=False,known_ids=None,device='cpu',progress=None):
    """videos(v,cap) -> iterable of (route,visible evidence) in causal order."""
    model=ExactCandidateMemory(prototypes,device=device) if decision_model is not None else make_exact_baseline(name,prototypes,thresholds,device)
    events=[];seconds=0.;committed_count=0;committed_observations=0;size=0
    for video in order:
        if reset_per_video:model.clear_anonymous()
        for r,item in videos(video,prefix):
            if no_memory:model.clear_anonymous()
            started=time.perf_counter()
            if decision_model is not None:
                features,k,t=model.candidates(item['embedding'],item['uncertainty'],item['maturity'],item['quality'],item['elapsed'])
                with torch.inference_mode():logits=masked_logits(decision_model,torch.tensor(features))
                action=model.apply(predicted_action(logits,wait_bias,allow_wait),k,t,item['embedding'],item['quality'])
                size=len(model.anonymous)
            elif name=='B0_frame_snapshot_vote':
                f=item['frames'];view=PrefixView(f,np.zeros((len(f),4),np.float32),item['quality'],item['elapsed_frames'])
                action=model.step_prefix(view);size=len(model._anonymous)
            else:
                action=model.step(item['embedding']);size=len(model._centroids if name=='B2_track_dpmeans' else model._anonymous)
            seconds+=time.perf_counter()-started;kind=action['kind'];n=min(prefix,r['observation_count'])
            if kind!='WAIT':committed_count+=1;committed_observations+=n
            events.append(DecisionEvent(len(events),TrackKey(video,str(r['physical_track_id'])),n,kind,
                known_category_id=int(action['category_id']) if kind=='KNOWN' else None,
                token=action['token'] if kind in {'NEW','EXISTING'} else None))
        if progress:progress(video,len(events),size)
    sealed=seal_decisions(events,video_order=order,prefix_cap=prefix,known_ids=tuple(prototypes) if known_ids is None else known_ids)
    return sealed,{'policy_inference_seconds':seconds,'policy_seconds_per_track':seconds/max(len(events),1),
        'anonymous_memory_final':size,'created_states':len(sealed.created_tokens),
        'wait_rate':1-committed_count/max(len(events),1),'commitment_coverage':committed_count/max(len(events),1),
        'mean_committed_observation_latency':committed_observations/committed_count if committed_count else None,
        'wait_censored_at_prefix_not_zero_latency':True,'independent_capped_prefix_replay_not_dense_frame_online':True,
        'exhaustive_search_no_approximate_neighbors':True,'canonical_dot_refinement':True,
        'search_queries':model._search.queries,'canonical_scores_refined':model._search.refined,'search_device':device}
