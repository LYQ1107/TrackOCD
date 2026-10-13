"""Offline fixed uniform-rank Known calendar, no pseudo/history-based choice.

Caller gates approved policy caches/labels first. Private administrative plans
are frozen BEFORE optimizer work. Model receives only current numeric prefixes;
training plan may use offline Known availability. Perturbation tests must reuse
the frozen plan, not assert causality after reselecting/retraining a schedule.
"""
import hashlib
import numpy as np
from .causal_observation_router import CurrentObservationRouter
from .candidate_training_loss import lawful_label
from .policy_training_data import policy_current_frames


def uniform_ranks(n,k):
    if type(n) is not int or type(k) is not int or not 1<=k<=n:
        raise ValueError('Whole fixed update count required; no partial or duplicated padding')
    return tuple(((2*i+1)*n)//(2*k) for i in range(k))


def freeze_calendar(videos,projection,seed,episodes,budget,guard):
    if type(seed) is not int or seed not in (1027,1028,1029) or type(episodes) is not int or episodes<1:
        raise ValueError('Explicit frozen training seed/episode plan')
    targets={(v,str(t)):label for (v,t),label in projection.targets.items()}
    rng=np.random.default_rng(seed); orders=[]; selected=[]; counts=[]; h=hashlib.sha256()
    for episode in range(episodes):
        order=rng.permutation(len(videos)).tolist(); orders.append(order); available=[]
        for position,ordinal in enumerate(order):
            video,arrays=videos[ordinal]; router=CurrentObservationRouter()
            for frame,rows in policy_current_frames(arrays):
                guard(); batch=router.ingest(frame,rows)
                anchor=False
                for aliases in batch.routing_groups:
                    labels=[lawful_label(targets.get((video,r),('no_retained_semantic_truth',None))) for r in aliases]
                    if labels and all(v is not None for v in labels) and len(set(labels))==1 and labels[0][0]=='Known':
                        anchor=True
                if anchor: available.append((episode,position,frame))
        chosen=tuple(available[i] for i in uniform_ranks(len(available),budget))
        selected.append(chosen); counts.append(len(available))
        for key in chosen: h.update(np.asarray(key,dtype='<i8').tobytes())
    return {'orders':orders,'selected_frames':tuple(selected),'eligible_Known_frame_counts':counts,
        'selected_calendar_sha256':h.hexdigest(),'budget_each':budget,
        'rule':'MIDPOINT_UNIFORM_KNOWN_ELIGIBLE_RANKS_NO_PSEUDO_OR_PREDICTION_HISTORY_SELECTOR',
        'private_admin_plan_not_model_input':True}
