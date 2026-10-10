import copy
import numpy as np
import pytest
import torch
from src.trackocd_core.persistent_policy import CandidateMemory,DecisionMLP,SupervisionHistory,masked_logits,predicted_action
from src.trackocd_core.train_first_replay import causal_rows


def test_prediction_memory_has_no_gt_and_wait_never_writes():
    m=CandidateMemory({3:np.array([1.,0.])});a=np.array([0.,1.]);before=copy.deepcopy(m.__dict__)
    m.apply(3,3,None,a,[1.]);assert m.next_id==before['next_id'] and not m.anonymous
    new=m.apply(2,3,None,a,[1.]);assert new['token']=='S:0'
    m.apply(1,3,new['token'],np.array([1.,0.]),[1.]);assert m.anonymous['S:0'].count==2
    assert m.anonymous['S:0'].dispersion>0
    assert not any(k in m.__dict__ for k in ('category','labels','gt','roles'))
    with pytest.raises(ValueError):m.apply(1,3,'future',a,[1.])
    m.clear_anonymous();assert m.apply(2,3,None,a,[1.])['token']=='S:1'


def test_mlp_capacity_masks_and_same_init():
    torch.manual_seed(1027);a=DecisionMLP();torch.manual_seed(1027);b=DecisionMLP()
    assert sum(p.numel() for p in a.parameters())==708
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    cues=torch.zeros(17);z=masked_logits(a,cues)
    assert z[0].item()==-1e9 and z[1].item()==-1e9
    assert predicted_action(z,100)==3 and predicted_action(z,100,False)==2


def test_absorbing_supervision_pollution_not_gt_repair():
    h=SupervisionHistory();h.update({'kind':'NEW','token':'S:0'},10,0)
    h.update({'kind':'EXISTING','token':'S:0'},11,1)
    h.update({'kind':'EXISTING','token':'S:0'},10,2)
    assert not h.pure('S:0',10) and h.members['S:0']==[10,11,10]
    costs,target,credit=h.costs(10,False,1,'S:0',4,{'wrong_known':8,'wrong_existing_pollution':10,
        'known_as_new':4,'wrong_new_fragmentation':3,'wait_base':.5,'wait_per_observation':.05})
    assert costs[1]==10 and costs[2]==3 and costs[3]>0 and target==3 and credit==2


def test_paired_video_bootstrap_is_conditional_and_never_changes_mapping():
    from src.trackocd_core.scientific_statistics import paired_video_bootstrap
    a=[{'video_id':v,'known_gt':2,'novel_gt':2,'old_correct':1,'new_correct':1,'reuse_opportunities':1,'correct_ct':0} for v in (1,2,3)]
    b=[{**r,'correct_ct':1} for r in a]
    result=paired_video_bootstrap(a,b)
    assert result['metrics']['correct_commit_ct']['right_minus_left']==1
    assert result['metrics']['correct_commit_ct']['conditional_video_bootstrap_95pct']==[1,1]
    assert result['conditional_on_frozen_global_mapping_and_predicted_stream_state']
    assert not result['independent_video_or_category_observations_claimed']


def test_short_track_chronological_replay_uses_observed_frame():
    rows=[{'key':'a','video_id':1,'physical_track_id':2,'observation_count':1,'frame_ids':[7]},
          {'key':'b','video_id':1,'physical_track_id':1,'observation_count':2,'frame_ids':[1,8]}]
    assert [r['key'] for r in causal_rows(rows,[1],16)]==['a','b']
    assert [r['key'] for r in causal_rows(rows,[1],1)]==['b','a']


def test_actual_causal_bank_and_all_wait_coverage_penalty():
    from pathlib import Path
    from src.trackocd_core.train_first_experiment import load_train,evidence_bank,prototypes,evaluate
    from src.trackocd_core.train_first_replay import replay
    root=Path(__file__).resolve().parents[2];cache,labels,known=load_train(root);by={r['key']:r for r in labels}
    rows=[r for r in cache.rows if by[r['key']]['partition']=='policy_train'][:8]
    bank=evidence_bank(cache,rows);proto=prototypes(cache,labels,known)
    model=DecisionMLP().eval();order=sorted({r['video_id'] for r in rows})
    ledger,runtime=replay(rows,order,16,proto,{},bank[16],decision_model=model,wait_bias=100)
    scores=evaluate(ledger,rows,labels)
    assert runtime['wait_rate']==1 and runtime['created_states']==0
    assert scores['standard']['all_denominator']==len(rows) and scores['standard']['all_acc']==0
    assert scores['standard']['non_wait_coverage']==0 and scores['standard']['unresolved_count']==len(rows)
    for cap in (1,2,4,8,16):
        for r in rows:assert bank[cap][r['key']]['actual_prefix']==min(cap,r['observation_count'])


def test_real_predicted_rollout_risk_gradients_not_gt_memory():
    from torch.nn import functional as F
    m=CandidateMemory({1:np.array([1.,0.])});h=SupervisionHistory();model=DecisionMLP()
    probabilities=[];actions=[];losses=[]
    constants={'wrong_known':8,'wrong_existing_pollution':10,'known_as_new':4,'wrong_new_fragmentation':3,'wait_base':.5,'wait_per_observation':.05}
    # Forced wrong actions are a test fixture; real trainer uses predicted_action.
    for i,(category,z,forced) in enumerate(((10,[0.,1.],2),(11,[.1,1.],1),(10,[0.,1.],1))):
        features,k,t=m.candidates(z,.1,2,[1.],3);logits=masked_logits(model,torch.tensor(features))
        costs,target,credit=h.costs(category,False,k,t,2,constants)
        probability=logits.softmax(-1);loss=F.cross_entropy(logits[None],torch.tensor([target]))+(probability*torch.tensor(costs)).sum()
        if credit is not None:loss=loss+.25*probabilities[credit][actions[credit]]
        prediction=m.apply(forced,k,t,z,[1.]);h.update(prediction,category,i)
        probabilities.append(probability);actions.append(forced);losses.append(loss)
    torch.stack(losses).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(m.anonymous)==1 and m.anonymous['S:0'].count==3
    assert not h.pure('S:0',10) and h.members['S:0']==[10,11,10]
