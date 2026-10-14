from types import SimpleNamespace
import numpy as np
import pytest
from src.trackocd_cvpr.arbitration_attribution import (
    PredictionSupportObserver, external_label, attribution_stream)
from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit, calibrated_raw_scores
from src.trackocd_cvpr.atomic_member_control import AtomicMemberControl
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter, CurrentObservation


def fixture():
    n = 5; visual = np.zeros((n*3,768),np.float32)
    visual[0,:2] = [.6,.8]; visual[1,:2] = [.9,np.sqrt(1-.9**2)]; visual[2:n,0] = 1
    visual[n:2*n,2] = 1; visual[2*n:,3] = 1
    a = {'track_id':np.array([10,20,30]),'offsets':np.arange(4)*n,'visual':visual,
         'geometry':np.tile([.1,.1,.5,.5],(n*3,1)).astype(np.float32),'quality':np.ones(n*3,np.float32),
         'frame_index':np.tile(np.arange(1,n+1),3),'clock_frame_index':np.arange(n+1)}
    known = np.zeros(768,np.float32); known[0] = 1
    cfg = {'physical_tracks':3,'observations':15,'metadata_frames':6,'empty_frames':1}
    targets = {(1,'10'):('Known',0),(1,'20'):('policy_pseudo',0)}
    return [(1,a)],targets,{0:known},cfg


def head():return NamedKnownLogit().eval().requires_grad_(False)


def test_real_synthetic_observer_preserves_full_events_commits_state_and_counts_unknown():
    args = fixture(); h = head()
    observed,witness = attribution_stream(*args,1027,lambda:None,head=h)
    normal,reference = attribution_stream(*args,1027,lambda:None,head=h,observe=False)
    assert witness == reference and observed['full_clock'] == normal['full_clock']
    assert observed['full_clock']['physical_observations'] == 15 and observed['full_clock']['empty_frames'] == 1
    assert observed['external_strata']['masked']['physical_observations'] == 5
    k = observed['external_strata']['Known']
    assert k['positive_named_gate_bypassed_groups'] >= 1
    assert k['Existing_provenance_partition_only_own_groups'] >= 1
    assert k['same_history_remove_own_correct_Known_groups'] >= 1
    assert observed['same_history_one_decision_shadow_not_counterfactual_rollout_gain']
    assert not observed['GT_or_shadow_decisions_used_for_any_policy_or_memory_write']


def test_near_unit_nonaxis_Known_prototypes_use_exact_constructor_candidate_and_score_bytes():
    videos,targets,known,cfg = fixture()
    value = np.random.default_rng(17).normal(size=768).astype(np.float32)
    value /= np.linalg.norm(value); value *= np.float32(1.0000002)
    known[1] = value
    measured,private = attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=head())
    normal,reference = attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=head(),observe=False)
    assert private == reference and measured['full_clock'] == normal['full_clock']


def test_changed_external_labels_do_not_change_actual_events_support_or_shadow_numeric_process():
    videos,targets,known,cfg = fixture(); h = head()
    a,pa = attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=h)
    changed = dict(targets); changed[(1,'10')] = ('policy_pseudo',1)
    b,pb = attribution_stream(videos,changed,known,cfg,1027,lambda:None,head=h)
    assert pa == pb and a['full_clock'] == b['full_clock']
    assert a['external_strata'] != b['external_strata']


def test_ID_reverse_delivery_and_actual_future_perturbation_preserve_required_numeric_witness():
    videos,targets,known,cfg = fixture(); h = head()
    a,pa = attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=h)
    b,pb = attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=h,rename_and_reverse=True)
    assert a == b and [r.events for _,r in pa[0]] == [r.events for _,r in pb[0]] and pa[2] == pb[2]
    changed = {k:v.copy() for k,v in videos[0][1].items()}
    future = changed['frame_index'] > 1; changed['visual'][future] = 0; changed['visual'][future,4] = 1
    _,pc = attribution_stream([(1,changed)],targets,known,cfg,1027,lambda:None,head=h)
    earlier = lambda rows:[r.events for _,r in rows if r.frame_index <= 1]
    assert any(earlier(pa[0])) and earlier(pa[0]) == earlier(pc[0])


def test_alias_exclusion_preserves_other_alias_group_dedup_and_never_mutates_witness():
    x = np.zeros(768,np.float32); x[2] = 1; known = np.zeros(768,np.float32); known[0] = 1
    rows = tuple(CurrentObservation(r,x,np.array([.1,.1,.5,.5],np.float32),1.) for r in ('a','b'))
    model = head(); c = AtomicMemberControl({0:known},lambda f,q:calibrated_raw_scores(f,q,model=model))
    batch = CurrentObservationRouter().ingest(1,rows); result = c.ingest(1,rows)
    observer = PredictionSupportObserver(2,2,10); observer.observe(batch,result)
    original = observer.evidence(); token = result.events[0].token
    assert original[token].support_groups == 1 and original[token].routed_members == 2
    one = observer.evidence(exclude_aliases=('a',))[token]
    assert one.support_groups == one.routed_members == 1 and one.prototype().tolist() == original[token].prototype().tolist()
    assert observer.evidence(exclude_aliases=('a','b'))[token].status == 'EMPTY'
    assert observer.evidence() == original and observer.commits == 2
    assert observer.membership(token,('a',)) == {'contains_own':True,'only_own':False,'has_other_same_video':True,'has_other_video':False}
    observer.next_video()
    assert observer.membership(token,('a',))['has_other_video'] and not observer.membership(token,('a',))['contains_own']


def test_unknown_conflicting_group_not_a_Known_or_Novel_negative_and_no_head_training():
    targets = {(1,'a'):('Known',0),(1,'b'):('policy_pseudo',0)}
    assert external_label(targets,1,('a','b')) is None and external_label(targets,1,('missing',)) is None
    with pytest.raises(ValueError,match='Frozen'):attribution_stream(*fixture(),1027,lambda:None,head=NamedKnownLogit())


@pytest.mark.parametrize('key,value,message',[('observations',14,'Member/token/history/payload bound'),('metadata_frames',5,'Full policy clock')])
def test_denominator_relaxation_does_not_turn_partial_clock_into_success(key,value,message):
    videos,targets,known,cfg = fixture(); cfg[key] = value
    with pytest.raises(ValueError,match=message):
        attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=head())
