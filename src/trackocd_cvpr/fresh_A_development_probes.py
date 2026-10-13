"""Qualified one-track numeric probes, not full controller or temporal gain."""
import numpy as np
import torch
from .fit_R_training import FitTrack
from .differentiable_support_bank import VisiblePrefix
from .fresh_A_baseline import forward_ragged
from .fresh_A_development_encoding import encode_native_prefixes,validate_frozen_model


def numeric_probes(model,tracks,guard):
    validate_frozen_model(model)
    selected=next((t for t in tracks if t.count()>=2),None)
    if selected is None:return {'status':'NO_FUTURE_OBSERVATIONS_NOT_PROOF','future_rows_mutated':0}
    first,end=map(int,selected.arrays['offsets'][selected.position:selected.position+2])
    arrays=dict(selected.arrays);arrays['visual']=arrays['visual'].copy()
    original=arrays['visual'][first+1:end].copy()
    arrays['visual'][first+1:end]=0;arrays['visual'][first+1:end,0]=1
    changes=int(np.any(arrays['visual'][first+1:end]!=original,axis=1).sum())
    if not changes:raise ValueError('Future probe is a no-op, not proof')
    changed=FitTrack(selected.video,arrays,selected.position,None)
    renamed=FitTrack(-selected.video-1000,selected.arrays,selected.position,None)
    reference,counts=encode_native_prefixes(model,[selected],1,guard)
    for candidate in (changed,renamed):
        value,actual=encode_native_prefixes(model,[candidate],1,guard)
        if not np.array_equal(value,reference) or actual!=counts:raise ValueError('Actual visible prefix depends on future/routing')
    p=selected.visible(min(4,selected.count()))
    reverse=VisiblePrefix(p.visual.flip(0),p.boxes.flip(0),p.quality.flip(0),p.elapsed)
    timed=VisiblePrefix(p.visual,p.boxes,p.quality,torch.arange(len(p.elapsed),dtype=torch.int64))
    guard()
    with torch.inference_mode():
        base=forward_ragged(model,[p]);ordered=forward_ragged(model,[reverse]);compressed=forward_ragged(model,[timed])
    return {'status':'ACTUAL_ONE_TRACK_PREFIX_PROBES_NOT_FULL_ONLINE_CONTROLLER_PROOF',
        'future_rows_mutated':changes,'actual_first_prefix_future_invariant':True,
        'actual_numeric_prefix_routing_rename_invariant':True,'observations_in_order_probe':len(p.elapsed),
        'order_visual_rows_changed':not torch.equal(p.visual,reverse.visual),
        'compressed_time_values_changed':not torch.equal(p.elapsed,timed.elapsed),
        'order_embedding_l2_change':float((base-ordered).norm()),'time_embedding_l2_change':float((base-compressed).norm()),
        'static_capacity_is_not_claimed_strictly_timeless':True,'temporal_benefit_or_Goal_accepted':False}
