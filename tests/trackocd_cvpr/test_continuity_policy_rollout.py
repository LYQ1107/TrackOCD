import numpy as np
import pytest
from src.trackocd_cvpr.continuity_policy_rollout import policy_stream,trace_numeric,trace_known_rebound
from src.trackocd_cvpr.continuity_reuse_control import MODES,ContinuityReceipt
from src.trackocd_cvpr.continuity_origin_audit import ContinuityOriginAudit
from src.trackocd_cvpr.atomic_member_control import MemberControlEvent,MemberFrameResult
from src.trackocd_cvpr.arbitration_attribution import attribution_stream
from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit


def fixture():
    def arrays(ids,axes):
        n=3;z=np.zeros((len(ids)*n,768),np.float32)
        for i,axis in enumerate(axes):z[i*n:(i+1)*n,axis]=1
        return {'track_id':np.array(ids),'offsets':np.arange(len(ids)+1)*n,'visual':z,
            'geometry':np.tile([.1,.1,.5,.5],(len(z),1)).astype(np.float32),'quality':np.ones(len(z),np.float32),
            'frame_index':np.tile(np.arange(1,n+1),len(ids)),'clock_frame_index':np.arange(n+1)}
    videos=[(1,arrays([10,20,30],[0,2,3])),(2,arrays([40,50],[0,2]))]
    labels={(1,'10'):('Known',10),(1,'20'):('declared_pseudoNovel',20),(1,'30'):('no_retained_semantic_truth',None),
        (2,'40'):('Known',10),(2,'50'):('declared_pseudoNovel',20)}
    z=np.zeros(768,np.float32);z[0]=1
    return videos,labels,{0:z},{0:10},{'physical_tracks':5,'observations':15,'metadata_frames':8,'empty_frames':2,'prefix_snapshots':[1,2,4,8,16]}


def head():return NamedKnownLogit().eval().requires_grad_(False)


@pytest.mark.parametrize('mode',MODES)
def test_complete_actual_synthetic_closed_loops_keep_unknown_clock_origins_and_own_history(mode):
    a=fixture();r,p=policy_stream(*a,mode,1027,lambda:None,head=head())
    assert r['full_clock']['physical_observations']==15 and r['full_clock']['empty_frames']==2
    audit=r['origin_aware_audit'];last=audit['primary_scored_reuse_prefix_snapshots']['all_available']
    assert last['physical_tracks']==5 and last['Known_queries']==2 and last['declared_pseudo_queries']==2
    assert last['Known_micro_accuracy']==1 and last['certified_reuse_excludes_routing_continuation']
    assert sum(audit['physical_origin_observations'].values())==15
    assert not audit['GT_or_origin_used_to_rewrite_memory_history']
    assert len(p[1])==9 and r['actual_commits']==9
    assert audit['first_commit_latency']['full_track_denominator']==5
    assert audit['first_certified_scored_cross_video_reuse_latency_declared_pseudo']['full_track_denominator']==2


def test_full_raw_exact_original_reference_events_commits_states_not_only_aggregate():
    videos,labels,known,bindings,cfg=fixture()
    targets={key:('Known',0) if role=='Known' else ('policy_pseudo',0) if role=='declared_pseudoNovel' else ('masked',None)
             for key,(role,_) in labels.items()}
    r,p=policy_stream(videos,labels,known,bindings,cfg,'full_raw',1027,lambda:None,head=head())
    old,original=attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=head(),observe=False)
    assert [(x.video,x.result) for x in p[0]]==original[0] and p[1:]==original[1:]
    assert r['full_clock']==old['full_clock'] and r['allocated_tokens']==old['allocated_tokens']


@pytest.mark.parametrize('mode',MODES)
def test_ID_reverse_and_nonvacuous_future_preserve_numeric_origin_state_and_history(mode):
    videos,labels,known,bindings,cfg=fixture();h=head()
    a,p=policy_stream(videos,labels,known,bindings,cfg,mode,1027,lambda:None,head=h)
    b,q=policy_stream(videos,labels,known,bindings,cfg,mode,1027,lambda:None,head=h,rename_and_reverse=True)
    assert a==b and trace_numeric(p[0])==trace_numeric(q[0]) and p[2]==q[2]
    ordinal=int(np.random.default_rng(1027).permutation(2)[0]);v,arr=videos[ordinal]
    changed={k:x.copy() for k,x in arr.items()};future=changed['frame_index']>1
    changed['visual'][future]=0;changed['visual'][future,4]=1
    assert future.any() and np.any(changed['visual'][future]!=arr['visual'][future])
    altered=list(videos);altered[ordinal]=(v,changed)
    _,z=policy_stream(altered,labels,known,bindings,cfg,mode,1027,lambda:None,head=h)
    prior=lambda trace:trace_numeric(tuple(x for x in trace if x.video==v and x.result.frame_index<=1))
    assert prior(p[0]) and prior(p[0])==prior(z[0])


def test_external_label_swap_changes_outcomes_never_numeric_closed_loop_writes():
    videos,labels,known,bindings,cfg=fixture();other=known[0].copy();other[:2]=[0,1];known[1]=other;bindings[1]=11
    h=head();a,p=policy_stream(videos,labels,known,bindings,cfg,'exclude_query_raw_with_continuity',1027,lambda:None,head=h)
    changed={k:('Known',11) if role=='Known' else (role,c) for k,(role,c) in labels.items()}
    b,q=policy_stream(videos,changed,known,bindings,cfg,'exclude_query_raw_with_continuity',1027,lambda:None,head=h)
    assert trace_numeric(p[0])==trace_numeric(q[0]) and p[1:]==q[1:] and a['full_clock']==b['full_clock']
    assert a['origin_aware_audit']!=b['origin_aware_audit']


@pytest.mark.parametrize('mode',MODES)
def test_known_output_binding_shuffle_preserves_numeric_scores_origins_and_dirty_history(mode):
    videos,labels,known,bindings,cfg=fixture();other=known[0].copy();other[:2]=[0,1];known[1]=other;bindings[1]=11
    a,p=policy_stream(videos,labels,known,bindings,cfg,mode,1027,lambda:None,head=head())
    b,q=policy_stream(videos,labels,{1:known[0],0:known[1]},{1:10,0:11},cfg,mode,1027,lambda:None,head=head())
    assert a==b and trace_numeric(p[0])==trace_known_rebound(q[0],{0:1,1:0}) and p[1:]==q[1:]


def event(action,token=None,category=None):return MemberControlEvent('synthetic',b'',b'',(),b'',action,category,token,None)


def test_legacy_certified_identity_never_turns_routing_continuation_into_primary_scored_reuse():
    labels={(1,'known'):('Known',10),(1,'a'):('declared_pseudoNovel',20),(2,'b'):('declared_pseudoNovel',20)}
    audit=ContinuityOriginAudit(labels,{0:10})
    scored=lambda a,b=None:ContinuityReceipt(a,b,'SCORED_CANDIDATE',(),0,0)
    audit.audit_frame(1,MemberFrameResult(1,(event('KNOWN',category=0),event('NEW','S:0')),(('known',),('a',))),
        (scored('KNOWN',0),scored('NEW')))
    audit.audit_frame(2,MemberFrameResult(1,(event('EXISTING','S:0'),),(('b',),)),(scored('EXISTING','S:0'),))
    audit.audit_frame(2,MemberFrameResult(2,(event('EXISTING','S:0'),),(('b',),)),
        (ContinuityReceipt('NEW',None,'ROUTE_CONTINUATION_NOT_CATEGORY_MATCH',('S:0',),2,1),))
    result=audit.finish()
    legacy=result['legacy_identity_dirty_history_includes_continuation_not_primary_scored_match']['prefix_snapshots']['all_available']
    primary=result['primary_scored_reuse_prefix_snapshots']['all_available']
    assert legacy['certified_cross_video_reuse_micro_lower_bound']==.5
    assert primary['certified_cross_video_reuse_micro_lower_bound']==0
    assert primary['routing_only_continuation_declared_pseudo_tracks']==1
    assert primary['all_physical_commit_track_coverage']==1


def test_origin_missing_or_false_binding_and_partial_clock_refuse():
    labels={(1,'a'):('Known',10),(1,'b'):('declared_pseudoNovel',20)};audit=ContinuityOriginAudit(labels,{0:10})
    result=MemberFrameResult(1,(event('KNOWN',category=0),),(('a',),))
    with pytest.raises(ValueError,match='origin receipt'):audit.audit_frame(1,result,())
    with pytest.raises(ValueError,match='binding'):audit.audit_frame(1,result,(ContinuityReceipt('KNOWN',1,'SCORED_CANDIDATE',(),0,0),))
    a=list(fixture());a[-1]={**a[-1],'observations':14}
    with pytest.raises(ValueError):policy_stream(*a,'full_raw',1027,lambda:None,head=head())
