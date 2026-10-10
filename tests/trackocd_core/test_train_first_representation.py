import numpy as np
import torch
from src.trackocd_core.features import prefix_view
from src.trackocd_core.representation import cross_video_category_loss
from src.trackocd_core.train_first_representation import BalancedBatchSampler,forward_views,make_model,representation_diagnostics


def test_temporal_and_static_controls_have_exact_equal_capacity_and_same_adapter_init():
    hashes=[]
    for name in ('A1_ADAPTER','A2_EVIDENCE','A2_CAPACITY_CONTROL'):
        torch.manual_seed(1027);model=make_model(name)
        hashes.append([v.clone() for v in model.adapter.state_dict().values()])
        assert sum(p.numel() for p in model.parameters())==(525056 if name=='A1_ADAPTER' else 541889)
    assert all(torch.equal(a,b) for values in hashes[1:] for a,b in zip(hashes[0],values))


def test_each_sampler_anchor_has_cross_video_positive_and_negative_and_repeats_identically():
    rows=[{'category_id':c,'video_id':v,'physical_track_id':c*10+v} for c in range(8) for v in range(3)]
    a=BalancedBatchSampler(rows,1027);b=BalancedBatchSampler(rows,1027)
    for step in range(5):
        batch=a.draw();assert batch==b.draw()
        cats=torch.tensor([r['category_id'] for r in batch]);videos=torch.tensor([r['video_id'] for r in batch]);ids=torch.tensor([r['physical_track_id'] for r in batch])
        emb=torch.nn.functional.normalize(torch.randn(16,256),dim=-1)
        assert torch.isfinite(cross_video_category_loss(emb,cats,videos,ids))
        assert ((cats[:,None]==cats[None,:])&(videos[:,None]!=videos[None,:])).sum().item()==16


def test_variable_visible_lengths_not_padded_and_gradient_flows_for_every_model():
    rng=np.random.default_rng(8)
    views=[prefix_view(rng.normal(size=(n,768)),np.ones((n,4)),np.ones(n),np.arange(n),n) for n in (1,3,8)]
    for name in ('A1_ADAPTER','A2_EVIDENCE','A2_CAPACITY_CONTROL'):
        model=make_model(name);embedding=forward_views(model,views,'cpu')
        assert embedding.shape==(3,256) and torch.isfinite(embedding).all()
        embedding[:,0].sum().backward()
        assert all(p.grad is not None for p in model.parameters())


def test_temporal_and_capacity_prefix_ignore_future_tail():
    torch.manual_seed(7);visual=torch.randn(1,8,768);quality=torch.ones(1,8);elapsed=torch.arange(8)[None].float()
    for name in ('A2_EVIDENCE','A2_CAPACITY_CONTROL'):
        model=make_model(name).eval()
        original=model(visual[:,:2],quality[:,:2],elapsed[:,:2])['embedding']
        visual[:,2:]=999
        assert torch.equal(original,model(visual[:,:2],quality[:,:2],elapsed[:,:2])['embedding'])


def test_diagnostics_exclude_same_video_gallery_and_detect_known_collapse():
    rows=[{'category_id':c,'video_id':v,'simulation_role':'known' if c==1 else 'pseudo_novel'} for c in (1,2) for v in (1,2)]
    value=np.asarray([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],dtype=np.float32)
    result=representation_diagnostics(value,rows,{1:np.asarray([1.,0.])})
    assert result['cross_video_rank1_macro']==1 and result['known_vs_pseudo_novel_auroc']==1
    collapsed=representation_diagnostics(np.ones((4,2),dtype=np.float32)/np.sqrt(2),rows,{1:np.ones(2)/np.sqrt(2)})
    assert collapsed['known_vs_pseudo_novel_auroc']==.5
    assert collapsed['pseudo_novel_wrong_known_at_fixed_point65']==1


def test_training_registration_full_cache_and_known_allowlist_before_optimizer():
    import inspect
    from scripts.trackocd_core import train_core_representation
    source=inspect.getsource(train_core_representation.main)
    boundary=source.index('optimizer=torch.optim.AdamW')
    for required in ('remote!=args.preregistration_commit','TrainFirstCache(cache_root)',
                     "cache.manifest['selection_plan_sha256']",'legal_known',"'Registered category/video partition changed'"):
        assert source.index(required)<boundary
    assert "'final_heldout_evaluated':False" in source
    assert "'policy_or_heldout_features_used_for_fit':False" in source
