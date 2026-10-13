"""Native baseline encoding after caller's ALL15/source/authority/cache gate.

No files/weights/GT reads here. Raw reference MUST come from frozen original
R reference path, never substitute a numerically different Raw aggregation.
Offline Known prototypes use complete canonical fit5 members only; dev inputs
are numeric prefixes, not labels/video/physical IDs or future track lengths.
"""
import numpy as np
import torch
from .fresh_A_baseline import Fit5BatchSampler,FAMILIES,forward_ragged

CAPS=(1,2,4,8,16,'all_available')


def validate_frozen_model(model):
    if (getattr(model,'_fresh_fit5_baseline_family',None) not in FAMILIES
            or model.training or any(p.requires_grad or p.device.type!='cpu' for p in model.parameters())):
        raise ValueError('Registered frozen native CPU baseline, not fitting/old model')


def encode_native_prefixes(model,tracks,cap,guard):
    validate_frozen_model(model)
    if not ((type(cap) is int and cap in CAPS[:-1]) or (type(cap) is str and cap=='all_available')) or not tracks:
        raise ValueError('Complete observed physical population and one of six frozen snapshots')
    vectors=[];actual=[]
    with torch.inference_mode():
        for first in range(0,len(tracks),128):
            guard();chosen=tracks[first:first+128]
            counts=[t.count() if cap=='all_available' else min(cap,t.count()) for t in chosen]
            if any(type(n) is not int or n<1 for n in counts):raise ValueError('No empty/dropped track')
            prefixes=[t.visible(n) for t,n in zip(chosen,counts)]
            if sum(counts)>65536:raise ValueError('Bounded whole call, not truncating long observations')
            vectors.append(forward_ragged(model,prefixes).numpy().copy());actual.extend(counts);guard()
    embedding=np.concatenate(vectors)
    if embedding.shape!=(len(tracks),256):raise ValueError('Every physical native256 output kept')
    return embedding,actual


def offline_fit5_native_prototypes(model,fit_tracks,guard):
    validate_frozen_model(model);guard()
    # Reuse EXACT qualified canonicalization. This external map is supervision
    # bookkeeping, never forwarded to model. Seed affects draws only (none here).
    sampler=Fit5BatchSampler(fit_tracks,1027);prototypes=[];members=[]
    for slot in range(5):
        groups=sampler._groups[slot]
        tracks=[track for video in sorted(groups) for _,track in groups[video]]
        if not tracks:raise ValueError('All five offline Known groups, no dev fallback')
        values,_=encode_native_prefixes(model,tracks,'all_available',guard)
        mean=np.asarray(values,np.float64).mean(0);norm=np.linalg.norm(mean)
        if not np.isfinite(norm) or norm<=1e-12:raise ValueError('Undefined prototype, no class/row dropping')
        prototypes.append(np.asarray(mean/norm,np.float32));members.append(len(tracks))
    if sum(members)!=sampler.scope['canonical_known_groups']:raise ValueError('All canonical complete fit Known members')
    return np.stack(prototypes),{'scope':sampler.scope,'canonical_members_per_external_fit_slot':members,
        'prototype_source':'complete offline canonical fit5 only, equal member weighting',
        'dev_labels_or_prototypes_or_old_checkpoints_used':False,
        'Raw_reference_requires_original_frozen_R_path':True}
