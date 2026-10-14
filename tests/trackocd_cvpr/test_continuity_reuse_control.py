import numpy as np
import pytest
from src.trackocd_cvpr.atomic_member_control import AtomicMemberControl
from src.trackocd_cvpr.causal_observation_router import CurrentObservation
from src.trackocd_cvpr.continuity_reuse_control import ContinuityReuseControl, MODES, excluded_raw_state
from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit, calibrated_raw_scores


def vector(axis):
    x=np.zeros(768,np.float32);x[axis]=1;return x


def row(route,x):return CurrentObservation(route,x,np.array([.1,.1,.5,.5],np.float32),1.)


def scorer():
    head=NamedKnownLogit().eval().requires_grad_(False)
    return lambda f,c:calibrated_raw_scores(f,c,model=head)


@pytest.mark.parametrize('mode',MODES)
def test_actual_closed_loop_factorial_not_same_history_shadow(mode):
    c=ContinuityReuseControl({0:vector(0)},scorer(),mode=mode)
    results=[c.ingest(i,(row('a',vector(2)),)) for i in (1,2,3)]
    actions=[r.events[0].action for r in results]
    assert actions == (['NEW']*3 if mode=='exclude_query_raw' else ['NEW','EXISTING','EXISTING'])
    bank, allocated=c.state_snapshot()
    assert allocated == (3 if mode=='exclude_query_raw' else 1)
    assert len(bank.commit_history())==3
    assert bank.commit_history()[0].token=='S:0'
    receipt=c.continuity_receipts()[0]
    if mode=='exclude_query_raw_with_continuity':
        assert receipt.scored_action=='NEW' and receipt.scored_binding is None
        assert receipt.decision_origin=='ROUTE_CONTINUATION_NOT_CATEGORY_MATCH'
        assert receipt.retained_support_groups==0
        assert all(b.kind!='EXISTING' for bs in results[-1].events[0].candidate_bindings_for_external_loss for b in bs)
    assert not receipt.category_match_or_independent_support_certified


def test_full_raw_arm_exact_original_numeric_events_bank_history_empty_frames():
    z=np.random.default_rng(17).normal(size=768).astype(np.float32);z/=np.linalg.norm(z);z*=np.float32(1.0000002)
    known={0:vector(0),1:z};score=scorer()
    old=AtomicMemberControl(known,score);new=ContinuityReuseControl(known,score,mode='full_raw')
    for frame,rows in ((0,()),(1,(row('a',vector(2)),row('b',vector(0)))),(2,(row('a',vector(3)),))):
        assert old.ingest(frame,rows)==new.ingest(frame,rows)
        a,ai=old.state_snapshot();b,bi=new.state_snapshot()
        assert ai==bi and a.state_evidence()==b.state_evidence() and a.commit_history()==b.commit_history()


def test_known_and_wait_never_become_continuation_and_old_anonymous_support_survives():
    c=ContinuityReuseControl({0:vector(0)},scorer(),mode='exclude_query_raw_with_continuity')
    first=c.ingest(1,(row('a',np.array([.6,.8]+[0.]*766,np.float32)),))
    assert first.events[0].action=='NEW'
    r=c.ingest(2,(row('a',vector(0)),))
    assert r.events[0].action=='KNOWN' and c.continuity_receipts()[0].decision_origin=='SCORED_CANDIDATE'
    bank,_=c.state_snapshot();assert len(bank.commit_history())==1 and bank.state_evidence()['S:0'].support_groups==1
    wait=ContinuityReuseControl({0:vector(0),1:vector(0)},scorer(),mode='exclude_query_raw_with_continuity')
    assert wait.ingest(1,(row('a',vector(2)),)).events[0].action=='NEW'
    assert wait.ingest(2,(row('a',vector(0)),)).events[0].action=='WAIT'
    assert wait.continuity_receipts()[0].scored_action=='WAIT'
    assert wait.continuity_receipts()[0].previous_route_tokens==('S:0',)
    assert len(wait.state_snapshot()[0].commit_history())==1


def test_whole_old_alias_group_exclusion_not_alias_copy_or_member_mean_geometry():
    x=np.zeros(768,np.float32);x[2:4]=[.8,.6]
    c=ContinuityReuseControl({0:vector(0)},scorer(),mode='full_raw')
    c.ingest(1,(row('foreign_a',vector(2)),))
    c.ingest(2,(row('foreign_b',x),))
    foreign=c.state_snapshot()[0].state_evidence()['S:0']
    # weighted_mean normalizes EACH committed prefix. The bank then SUMS
    # distinct prefix members; do not replace that sum with a group mean.
    assert foreign.support_groups==2
    assert np.linalg.norm(np.frombuffer(foreign.vector_sum_float64_bytes,dtype='<f8'))==pytest.approx(np.sqrt(3.6))
    c.start_next_video()
    c.ingest(1,(row('a',x),row('alias',x)))
    c.ingest(2,(row('a',vector(2)),row('alias',vector(2))))
    bank,_=c.state_snapshot();before=bank.state_evidence();history=bank.commit_history()
    states,original,retained=excluded_raw_state(bank,('a',),3)
    assert original==3 and retained==2 and states['S:0'].support_groups==2
    assert states['S:0'].vector_sum_float64_bytes==foreign.vector_sum_float64_bytes
    assert states['S:0'].routed_members==2
    assert bank.state_evidence()==before and bank.commit_history()==history


@pytest.mark.parametrize('mode',MODES)
def test_rename_reverse_and_future_change_preserve_prior_actual_events_receipts(mode):
    def run(names,reverse=False,changed=False):
        c=ContinuityReuseControl({0:vector(0)},scorer(),mode=mode);out=[]
        for frame in (1,2,3):
            rows=[row(names[0],vector(2 if not changed or frame<3 else 4)),row(names[1],vector(3))]
            result=c.ingest(frame,tuple(reversed(rows)) if reverse else rows)
            out.append((result.events,c.continuity_receipts(),c.state_snapshot()[0].state_evidence()))
        return out
    a=run(('a','b'));assert a==run(('909','opaque'),True)
    assert a[:2]==run(('a','b'),changed=True)[:2]


def test_same_route_in_new_video_is_scored_external_reuse_not_old_route_continuation():
    c=ContinuityReuseControl({0:vector(0)},scorer(),mode='exclude_query_raw_with_continuity')
    c.ingest(1,(row('a',vector(2)),));c.start_next_video()
    assert not c.continuity_receipts()
    r=c.ingest(1,(row('a',vector(2)),));receipt=c.continuity_receipts()[0]
    assert r.events[0].action=='EXISTING' and receipt.scored_action=='EXISTING'
    assert receipt.previous_route_tokens==() and receipt.decision_origin=='SCORED_CANDIDATE'


def test_late_score_failure_rolls_back_router_bank_allocator_and_receipts():
    calls=[]
    def bad(f,c):
        calls.append(1);return np.full(len(f)+2,np.nan,np.float32) if len(calls)==2 else scorer()(f,c)
    c=ContinuityReuseControl({0:vector(0)},bad,mode='exclude_query_raw_with_continuity')
    c.ingest(1,(row('a',vector(2)),));calls.clear()
    before,n_before=c.state_snapshot();receipts=c.continuity_receipts()
    with pytest.raises(ValueError,match='One score'):c.ingest(2,(row('a',vector(2)),row('b',vector(3))))
    bank,n=c.state_snapshot()
    assert n==n_before and bank.commit_history()==before.commit_history() and bank.state_evidence()==before.state_evidence()
    assert c.continuity_receipts()==receipts and c._router._last_frame==1


def test_unknown_mode_or_nonpreframe_view_refuses():
    with pytest.raises(ValueError,match='four factorial'):ContinuityReuseControl({},scorer(),mode='selected')
    c=ContinuityReuseControl({0:vector(0)},scorer(),mode='full_raw');c.ingest(1,(row('a',vector(2)),))
    with pytest.raises(ValueError,match='pre-frame'):excluded_raw_state(c.state_snapshot()[0],('a',),1)


def test_source_only_protocol_keeps_all4arms_all5heads_and_exact_completed_prerequisite():
    import json
    from pathlib import Path
    from src.trackocd_v2.io import sha256_file
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/trackocd_cvpr/source_continuity_reuse_factorization_v1.json').read_text())
    assert tuple(cfg['arms'])==MODES and cfg['K']==8 and cfg['new_optimizer_steps']==0
    assert cfg['proposed_frozen_heads']==['initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy']
    assert sha256_file(root/cfg['complete_all5_attribution_comparison'])==cfg['complete_all5_attribution_comparison_sha256']
    assert not cfg['actual_pilot_executor_registered_or_automatically_launched']
    assert not cfg['source_tests_are_real_Train_evidence_or_self_correcting_memory_innovation']
    assert not cfg['full_intervention_gain_joint_R_C_M_Pred_dense_final_or_Goal_accepted']
