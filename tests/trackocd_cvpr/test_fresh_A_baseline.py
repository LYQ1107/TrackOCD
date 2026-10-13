from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.differentiable_support_bank import VisiblePrefix
from src.trackocd_cvpr.fresh_A_baseline import (
    FAMILIES,fresh_model,forward_ragged,supervised_batch_loss,Fit5BatchSampler,cross_space_Gram_drift)


def prefix(n,axis):
    visual=torch.zeros(n,768); visual[:,axis]=1
    for i in range(n): visual[i,(axis+i+7)%768]=.2*(i+1)
    visual=visual/visual.norm(dim=1,keepdim=True)
    return VisiblePrefix(visual,torch.tensor([[0.,0.,1.,1.]]).repeat(n,1),torch.ones(n),torch.arange(n))


def tracks():
    rows=[]
    for slot in range(5):
        for video in (slot*2,slot*2+1):
            p=prefix(slot+1,slot)
            rows.append(SimpleNamespace(loss_target=slot,video=video,position=0,
                arrays={'offsets':np.array([0,slot+1]),'frame_index':np.arange(slot+1)},
                count=lambda n=slot+1:n,visible=lambda n,p=p:VisiblePrefix(p.visual[:n],p.boxes[:n],p.quality[:n],p.elapsed[:n])))
    rows.append(SimpleNamespace(loss_target=None))
    return rows


def test_fresh_seed_adapter_init_exact_matched_all_families_independent_seeds_and_capacity():
    models=[fresh_model(f,1027) for f in FAMILIES]
    assert [sum(p.numel() for p in m.parameters()) for m in models]==[525056,525056,525056,541889,541889]
    for key,value in models[0].adapter.state_dict().items():
        assert all(torch.equal(value,m.adapter.state_dict()[key]) for m in models[1:])
    assert not torch.equal(models[0].adapter.network[0].weight,fresh_model(FAMILIES[0],1028).adapter.network[0].weight)
    with pytest.raises(ValueError): fresh_model(FAMILIES[0],0)


@pytest.mark.parametrize('family',FAMILIES)
def test_actual_ragged_nonpadding_forward_and_external_loss_have_nonzero_gradients(family):
    model=fresh_model(family,1027)
    views=[prefix((c%3)+1,c) for c in range(5) for _ in range(2)]
    fit=torch.arange(5).repeat_interleave(2); videos=torch.arange(10); groups=torch.arange(10)
    loss,embedding=supervised_batch_loss(model,views,fit,videos,groups)
    assert embedding.shape==(10,256) and torch.isfinite(loss)
    assert torch.allclose(embedding,torch.stack([forward_ragged(model,[v])[0] for v in views]),atol=1e-6)
    loss.backward(); assert model.adapter.network[0].weight.grad.norm()>0
    if family=='A2_EVIDENCE': assert model.reliability[0].weight.grad.norm()>0
    if family=='A2_CAPACITY_CONTROL': assert model.static[0].weight.grad.norm()>0


def test_external_mask_full_physical_accounting_aliases_and_matched_seed_batches():
    a=tracks(); a.append(a[0]); left=Fit5BatchSampler(a,1027); right=Fit5BatchSampler(list(reversed(a)),1027)
    assert left.scope==right.scope
    assert left.scope['physical_tracks']==12 and left.scope['masked_no_fit5_target_tracks']==1
    assert left.scope['canonical_known_groups']==10 and left.scope['exact_same_video_Raw_aliases_deduplicated']==1
    for step in range(10):
        x,y=left.draw(step),right.draw(step)
        assert (x.fit_slots,x.videos,x.canonical_groups,x.observed_counts)==(y.fit_slots,y.videos,y.canonical_groups,y.observed_counts)
        assert all(n<=t.count() for n,t in zip(x.observed_counts,x.tracks))
        assert all(x.videos[2*c]!=x.videos[2*c+1] for c in range(5))


def test_no_new_label_or_fake_cross_video_positive_no_empty_nonfinite_prefixes():
    a=tracks(); a[0].loss_target=5
    with pytest.raises(ValueError): Fit5BatchSampler(a,1027)
    with pytest.raises(ValueError): Fit5BatchSampler(tracks()[1:],1027)
    with pytest.raises(ValueError): forward_ragged(fresh_model(FAMILIES[0],1027),[])
    p=prefix(1,0); p.visual[0,0]=float('nan')
    with pytest.raises(ValueError): forward_ragged(fresh_model(FAMILIES[0],1027),[p])


def test_exact_alias_known_conflict_and_skipped_or_repeated_update_plans_refused():
    a=tracks(); original=a[0]; conflicting=SimpleNamespace(**vars(original)); conflicting.loss_target=1
    with pytest.raises(ValueError): Fit5BatchSampler([*a,conflicting],1027)
    sampler=Fit5BatchSampler(tracks(),1027)
    with pytest.raises(ValueError): sampler.draw(1)
    sampler.draw(0)
    with pytest.raises(ValueError): sampler.draw(0)


def test_frozen_protocol_points_to_existing_current_data_not_old_15class_loaders():
    root=Path(__file__).resolve().parents[2]
    cfg=json.loads((root/'configs/trackocd_cvpr/fresh_A_fit5_baselines_v1.json').read_text())
    assert (root/cfg['data_config']).is_file() and (root/cfg['frozen_protocol']).is_file()
    assert cfg['training_seeds']==[1027,1028,1029] and cfg['steps_per_family_and_seed']==200
    assert cfg['checkpoint_selection']=='Fixed200, never dev/final/Val/Test selection'
    assert not cfg['historical_old15_A1_A2_weights_labels_loader_or_selected_checkpoint_reuse']
    assert not cfg['policy_dev_final_Val_Test_targets_in_training_loss'] and not cfg['automatic_training_or_Goal_acceptance']


def test_cross_space_geometry_is_pairwise_Gram_not_invalid_768_256_pointwise_cosine():
    raw=np.eye(3,768); adapted=np.eye(3,256); out=cross_space_Gram_drift(raw,adapted)
    assert out['full_pairwise_Gram_MSE']==0 and out['pointwise_Raw_cosine_not_defined_across_768D_and_256D']
    with pytest.raises(ValueError): cross_space_Gram_drift(raw,adapted[:2])


def test_original_A1_geometry_controls_preserved_not_weak_selected_baseline():
    models=[fresh_model(f,1027) for f in ('A1_ADAPTER','A1_GEOMETRY_1','A1_GEOMETRY_5')]
    views=[prefix((c%3)+1,c) for c in range(5) for _ in range(2)]
    fit=torch.arange(5).repeat_interleave(2); videos=torch.arange(10); groups=torch.arange(10)
    losses=[supervised_batch_loss(m,views,fit,videos,groups)[0] for m in models]
    assert losses[0]<losses[1]<losses[2]
    assert torch.allclose(losses[2]-losses[0],5*(losses[1]-losses[0]),atol=1e-6)


def test_singleton_temporal_gradient_is_legitimately_zero_not_a_failed_optimizer():
    model=fresh_model('A2_EVIDENCE',1027); views=[prefix(1,c) for c in range(5) for _ in range(2)]
    loss,_=supervised_batch_loss(model,views,torch.arange(5).repeat_interleave(2),torch.arange(10),torch.arange(10))
    loss.backward()
    assert model.adapter.network[0].weight.grad.norm()>0
    assert model.reliability[0].weight.grad.norm()<1e-8


def test_zero_adapted_resultant_refused_without_dropping_physical_row():
    model=fresh_model('A1_ADAPTER',1027)
    with torch.no_grad():
        for value in model.parameters(): value.zero_()
    with pytest.raises(ValueError): forward_ragged(model,[prefix(1,0)])
