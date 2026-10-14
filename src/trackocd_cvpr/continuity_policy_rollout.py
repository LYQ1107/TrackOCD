"""Complete actual continuity/reuse closed loops; truth is external audit only."""
from collections import Counter
from dataclasses import dataclass,replace
import hashlib
import struct
import numpy as np
from .continuity_reuse_control import ContinuityReuseControl,MODES
from .continuity_origin_audit import ContinuityOriginAudit
from .policy_training_data import policy_current_frames
from .raw_known_membership_control import calibrated_raw_scores,parameter_sha256


@dataclass(frozen=True)
class PrivateContinuityTrace:
    video:int
    result:object
    origins:tuple
    numeric_history_sha256:str
    raw_state_sha256:str


def state_digest(states):
    h=hashlib.sha256()
    for token,s in sorted(states.items()):
        t=token.encode();h.update(struct.pack('<Q',len(t))+t)
        h.update(struct.pack('<QQdd',s.support_groups,s.routed_members,s.mean_quality,s.dispersion))
        h.update(s.vector_sum_float64_bytes+s.status.encode())
    return h.hexdigest()


def trace_numeric(trace):
    return tuple((r.result.events,r.origins,r.numeric_history_sha256,r.raw_state_sha256) for r in trace)


def trace_known_rebound(trace,inverse):
    """Undo ONLY an external Known output binding permutation, never numeric data."""
    rows=[]
    for r in trace:
        events=[]
        for e in r.result.events:
            groups=tuple(tuple(sorted((replace(b,identifier=inverse[b.identifier]) if b.kind=='KNOWN' else b
                for b in group),key=lambda b:str(b.identifier))) for group in e.candidate_bindings_for_external_loss)
            events.append(replace(e,candidate_bindings_for_external_loss=groups,
                category_id=inverse[e.category_id] if e.category_id is not None else None))
        origins=tuple(replace(o,scored_binding=inverse[o.scored_binding]) if o.scored_action=='KNOWN' else o for o in r.origins)
        rows.append((tuple(events),origins,r.numeric_history_sha256,r.raw_state_sha256))
    return tuple(rows)


def policy_stream(videos,labels,known,bindings,cfg,mode,order,guard,*,head,rename_and_reverse=False):
    if mode not in MODES or head.training or any(p.requires_grad for p in head.parameters()):
        raise ValueError('Exact frozen four-arm control, not an optimizer')
    before=parameter_sha256(head);sequence=np.random.default_rng(order).permutation(len(videos)).tolist();routes={}
    if rename_and_reverse:
        for video,a in videos:routes[video]={int(t):'continuity-renamed-'+str(len(a['track_id'])-i-1) for i,t in enumerate(a['track_id'])}
        labels={(v,routes[v][int(r)]):target for (v,r),target in labels.items()}
    control=ContinuityReuseControl(known,lambda f,c:calibrated_raw_scores(f,c,model=head),mode=mode,K=8,
        bank_limits={'max_members':cfg['physical_tracks'],'max_tokens':cfg['observations'],
                     'max_commits':cfg['observations'],'max_payload_bytes':67108864})
    audit=ContinuityOriginAudit(labels,bindings,prefixes=tuple(cfg['prefix_snapshots']),max_tokens=cfg['observations'])
    counts=Counter();private=[];ledger=hashlib.sha256();consumed=0
    for position,ordinal in enumerate(sequence):
        video,arrays=videos[ordinal]
        if position:control.start_next_video()
        for frame,rows in policy_current_frames(arrays,routing_bindings=routes.get(video)):
            if rename_and_reverse:rows=tuple(reversed(rows))
            guard();result=control.ingest(frame,rows);origins=control.continuity_receipts()
            bank,_=control.state_snapshot();commits=bank.commit_history()
            for c in commits[consumed:]:
                # Preserve numeric/provenance ledger but omit opaque renamed route spelling.
                b=repr((c.video_epoch,c.frame_index,c.previous_token,c.token,c.visible_input_sha256)).encode()
                ledger.update(struct.pack('<Q',len(b))+b)
            consumed=len(commits)
            private.append(PrivateContinuityTrace(video,result,origins,ledger.hexdigest(),state_digest(bank.state_evidence())))
            audit.audit_frame(video,result,origins)
            counts['metadata_frames']+=1;counts['empty_frames']+=int(not rows);counts['physical_observations']+=len(rows)
            for aliases,e in zip(result.routing_groups,result.events):counts['canonical_groups']+=1;counts[e.action]+=len(aliases)
    if (counts['physical_observations'],counts['metadata_frames'],counts['empty_frames'])!=(cfg['observations'],cfg['metadata_frames'],cfg['empty_frames']):
        raise ValueError('Complete policy clock, no dropped short/unknown/empty observations')
    if parameter_sha256(head)!=before:raise ValueError('Frozen head changed')
    bank,allocated=control.state_snapshot()
    return ({'mode':mode,'order_seed':order,'full_clock':dict(counts),'origin_aware_audit':audit.finish(),
        'video_ordinal_schedule_sha256':hashlib.sha256(np.asarray(sequence,dtype='<i8').tobytes()).hexdigest(),
        'allocated_tokens':allocated,'actual_commits':len(bank.commit_history()),'optimizer_steps':0,
        'own_continuation_is_scored_reuse_or_semantic_certificate':False,'GT_to_policy_or_memory_writes':False},
        (tuple(private),bank.commit_history(),bank.state_evidence()))
