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
def test_actual_main_representation_three_seeds_and_complete_matched_fits():
    import json
    from pathlib import Path
    from src.trackocd_v2.io import sha256_file
    root=Path(__file__).resolve().parents[2]
    receipt=json.loads((root/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_RESULT.json').read_text())
    assert receipt['fit_tracks']==1305 and len(receipt['fit_classes'])==15
    assert receipt['phase']=='representation' and len(receipt['fits'])==9
    assert not receipt['val_or_test_access'] and not receipt['final_heldout_evaluated']
    assert not receipt['DINO_detector_tracker_trained'] and not receipt['historical_outputs_overwritten']
    for seed in (1027,1028,1029):
        fits=[f for f in receipt['fits'] if f['seed']==seed]
        assert len(fits)==3 and len({f['adapter_initial_state_sha256'] for f in fits})==1
        assert len({f['batch_identity_prefix_sha256'] for f in fits})==1
        for f in fits:
            assert f['steps']==1000 and len(f['trace'])==1000
            assert f['positive_pairs']==16000 and f['negative_pairs']==224000
            assert 12<f['fit_tracks_actually_seen']<=1305
            assert [c['step'] for c in f['checkpoints']]==[250,500,1000]
            for c in f['checkpoints']:
                cp=c['checkpoint'];assert (root/cp['path']).stat().st_size==cp['bytes']
                assert sha256_file(root/cp['path'])==cp['sha256']
                assert c['development']['final_heldout_opened'] is False
    assert len({v['model'] for v in receipt['selected_geometry'].values()})==1


def test_actual_evidence_capacity_and_all_inputs_match_selected_A1():
    import json
    from pathlib import Path
    from src.trackocd_v2.io import sha256_file
    root=Path(__file__).resolve().parents[2]
    a=json.loads((root/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_RESULT.json').read_text())
    b=json.loads((root/'outputs/trackocd_core/TRAIN_FIRST_EVIDENCE_RESULT.json').read_text())
    assert b['phase']=='evidence' and len(b['fits'])==6 and b['matched_A1_selected_checkpoint_steps']
    assert not b['final_heldout_evaluated'] and not b['val_or_test_access']
    for f in b['fits']:
        reference=next(r for r in a['fits'] if (r['model'],r['seed'])==(a['shared_geometry_family'],f['seed']))
        assert f['adapter_initial_state_sha256']==reference['adapter_initial_state_sha256']
        assert f['batch_identity_prefix_sha256']==reference['batch_identity_prefix_sha256']
        assert f['selected_checkpoint']['step']==reference['selected_checkpoint']['step']
        assert f['parameters']==541889 and f['steps']==1000 and len(f['trace'])==1000
        assert f['positive_pairs']==reference['positive_pairs'] and f['negative_pairs']==reference['negative_pairs']
        for checkpoint in f['checkpoints']:
            cp=checkpoint['checkpoint'];assert sha256_file(root/cp['path'])==cp['sha256']


def test_actual_full_representation_heldout_same_denominators_and_all_cases():
    import json
    from pathlib import Path
    from src.trackocd_v2.io import sha256_file
    root=Path(__file__).resolve().parents[2]
    r=json.loads((root/'outputs/trackocd_core/TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT.json').read_text())
    assert len(r['cases'])==420 and r['known_gt']==207 and r['pseudo_novel_gt']==16
    assert not r['GT_or_Hungarian_in_prediction_memory'] and not r['val_or_test_access']
    assert r['all_seeds_orders_prefixes_kept'] and not r['heldout_used_to_redesign_method']
    assert r['config_sha256']==sha256_file(root/'configs/trackocd_core/gt_main_evaluation.json')
    assert len(r['calibrations'])==21 and all(len(c['all25trials'])==25 and not c['heldout_used'] for c in r['calibrations'])
    opportunities={o:set() for o in r['orders']}
    for c in r['cases']:
        assert c['standard']['old_denominator']==207 and c['standard']['new_denominator']==16
        assert c['standard']['all_denominator']==223 and c['persistent']['effective_commit_coverage']==1
        assert not c['persistent']['posthoc_hungarian_used']
        opportunities[c['order']].add(c['persistent']['fixed_gt_cross_video_reuse_opportunities'])
        assert sum(v['correct_ct'] for v in c['errors']['per_video_conditional_fixed_mapping_counts'])==c['persistent']['commit_ct_correct']
    assert all(len(v)==1 for v in opportunities.values())
    ledger=r['private_ledger'];assert sha256_file(root/ledger['path'])==ledger['sha256']
