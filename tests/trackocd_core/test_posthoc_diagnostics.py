import numpy as np
from src.trackocd_core.evaluation import DecisionEvent,TrackKey,Target,seal_decisions,join_evaluation
from src.trackocd_core.posthoc_diagnostics import evaluate_with_flags,paired_corrections


def test_unknown_members_and_mixed_tokens_never_hungarian_ct():
    keys=[TrackKey(v,str(i)) for i,v in enumerate([1,1,2,2,3])]
    targets=[Target(keys[0],99,'novel'),Target(keys[2],99,'novel'),Target(keys[3],100,'novel'),Target(keys[4],99,'novel')]
    events=[DecisionEvent(i,k,1,'NEW' if i==0 else 'EXISTING',token='A:0') for i,k in enumerate(keys)]
    sealed=seal_decisions(events,video_order=[1,2,3],prefix_cap=1,known_ids=[])
    join=join_evaluation(sealed,targets,{k:None if i==1 else k for i,k in enumerate(keys)})
    d,f=evaluate_with_flags(join)
    assert d['persistent']['commit_ct_correct']==0 and not f['CT_correct'].any()
    assert d['errors']['unmatched_anonymous_write_events']==1 and d['errors']['contaminated_states']==1
    assert d['standard']['new_correct']==3 and d['standard']['new_denominator']==4


def test_error_corrections_report_losses_as_well_as_wins():
    a={'video_id':np.array([1,2,3]),'local_id':np.array(['1','2','3']),'reuse_opportunity':np.array([False,True,True]),
       'novel':np.array([False,True,True]),'standard_correct':np.array([True,False,True]),'CT_correct':np.array([False,False,True])}
    b={**a,'standard_correct':np.array([False,True,True]),'CT_correct':np.array([False,True,False])}
    d=paired_corrections(a,b)
    assert d['standard_correct_all']['errors_corrected']==1 and d['standard_correct_all']['new_errors_introduced']==1
    assert d['CT_correct_novel']['net_correct_change']==0 and d['CT_correct_novel']['denominator']==2
