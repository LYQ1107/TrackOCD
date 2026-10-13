import copy
import importlib.util
import json
from pathlib import Path
import pytest
import torch
from src.trackocd_cvpr.fresh_A_training import fit_baseline,validate_protocol,verify_matched_pilot,checkpoint_state_equal
from src.trackocd_cvpr.fresh_A_baseline import FAMILIES,fresh_model
from scripts.trackocd_cvpr.pilot_fresh_A_optimizer import CONFIG,SOURCES,validate_execution,output_bytes
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest

ROOT=Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT/'configs/trackocd_cvpr/fresh_A_fit5_baselines_v1.json').read_text())


def toy_tracks():
    spec=importlib.util.spec_from_file_location('baseline_fixture',Path(__file__).with_name('test_fresh_A_baseline.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.tracks()


def test_all_five_synthetic_optimizer_pilots_match_actual_initialization_calendar_and_reopen(tmp_path):
    # Synthetic plumbing, NOT a real fit receipt, data read or Novel learning.
    rows=[]; checkpoints=[]
    for family in FAMILIES:
        def save(model,optimizer,step,metadata):
            p=tmp_path/(family+'.pt')
            torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                        'step':step,**metadata},p)
            restored=torch.load(p,weights_only=True)
            verifier=fresh_model(family,1027); verifier.load_state_dict(restored['model_state'])
            assert digest(verifier)==digest(model)
            assert len(restored['optimizer_state']['state'])==len(list(model.parameters()))
            assert checkpoint_state_equal(restored['optimizer_state'],optimizer.state_dict())
            checkpoints.append(restored['step'])
        model,optimizer,row=fit_baseline(toy_tracks(),config(),family,1027,'pilot',lambda:None,lambda r,c:None,save)
        assert row['completed_optimizer_steps']==10 and row['model_parameter_sha256_before']!=digest(model)
        assert row['adapter_loss_gradient_norm_peak']>0 and row['training_GPU_hours']==0
        assert row['batched_individual_numerically_qualified_steps']==10
        assert row['scope']['physical_tracks']==11 and row['scope']['masked_no_fit5_target_tracks']==1
        assert row['actually_sampled_canonical_groups']==10 and len(optimizer.state)==len(list(model.parameters()))
        rows.append(row)
    assert checkpoints==[10]*5
    assert verify_matched_pilot(rows)['actual_private_batch_calendar_identical']
    for field,value in [('batch_calendar_sha256','bad'),('adapter_initialization_sha256','bad'),('completed_optimizer_steps',9)]:
        bad=copy.deepcopy(rows); bad[0][field]=value
        with pytest.raises(ValueError): verify_matched_pilot(bad)
    with pytest.raises(ValueError): verify_matched_pilot(rows[:4])


def test_fixed200_synthetic_is_independent_initialization_not_pilot_warmstart_and_saves_all_fixed_steps():
    saved=[]; progressed=[]
    model,_,row=fit_baseline(toy_tracks(),config(),'A1_ADAPTER',1028,'Fixed200',lambda:None,
        lambda r,c:progressed.append(r['step']),lambda m,o,s,metadata:saved.append(s))
    assert saved==[20,50,100,200] and progressed==list(range(1,201))
    assert row['model_parameter_sha256_before']==digest(fresh_model('A1_ADAPTER',1028))
    assert row['completed_optimizer_steps']==200 and digest(model)!=row['model_parameter_sha256_before']
    with pytest.raises(ValueError): fit_baseline(toy_tracks(),config(),'A1_ADAPTER',1028,'pilot',lambda:None,None,None)


@pytest.mark.parametrize('field,value',[('training_seeds',[1027]),('steps_per_family_and_seed',10),
    ('learning_rate',.003),('historical_old15_A1_A2_weights_labels_loader_or_selected_checkpoint_reuse',True),
    ('policy_dev_final_Val_Test_targets_in_training_loss',True)])
def test_frozen_training_protocol_does_not_silently_relax_or_expand(field,value):
    cfg=config(); cfg[field]=value
    with pytest.raises(ValueError): validate_protocol(cfg)


@pytest.mark.parametrize('field,value',[('execution_device','CUDA'),('CPU_compute_workers',2),
    ('wall_seconds_ceiling',6000),('full_scientific_fit_automatic_start',True),
    ('output_directory','outputs/other')])
def test_pilot_execution_scope_refuses_mutation(field,value):
    cfg=json.loads((ROOT/CONFIG).read_text()); cfg[field]=value
    with pytest.raises(ValueError): validate_execution(cfg,{'status':'USER_APPROVED_FREEZE_THEN_BOUNDED_CACHE_AND_TRAIN'})


def test_source_registration_closure_and_recursive_partial_output_accounting(tmp_path):
    for p in SOURCES: assert (ROOT/p).is_file(),p
    for p in ('src/trackocd_core/representation.py','src/trackocd_core/train_first_representation.py',
              'src/trackocd_cvpr/fresh_A_training.py',CONFIG): assert p in SOURCES
    (tmp_path/'checkpoint.pt.partial').write_bytes(b'abc')
    (tmp_path/'nested').mkdir(); (tmp_path/'nested'/'receipt').write_bytes(b'ab')
    assert output_bytes(tmp_path)==5
    (tmp_path/'link').symlink_to(tmp_path/'checkpoint.pt.partial')
    with pytest.raises(ValueError): output_bytes(tmp_path)


def test_checkpoint_exact_optimizer_values_not_only_state_count_or_model_hash():
    a={'state':{0:{'step':torch.tensor(1.),'exp_avg':torch.tensor([.1,.2])}},'param_groups':[{'lr':.0003}]}
    b=copy.deepcopy(a); assert checkpoint_state_equal(a,b)
    b['state'][0]['exp_avg'][0]=.3; assert not checkpoint_state_equal(a,b)
    b=copy.deepcopy(a); b['param_groups'][0]['lr']=.003; assert not checkpoint_state_equal(a,b)


@pytest.mark.parametrize('failure',['remote','source'])
def test_real_worker_gate_failure_precedes_private_loader_and_namespace(monkeypatch,tmp_path,failure):
    from types import SimpleNamespace
    import scripts.trackocd_cvpr.pilot_fresh_A_optimizer as driver
    import src.trackocd_cvpr.fit_R_training as fit_data
    cfg=json.loads((ROOT/CONFIG).read_text()); out=tmp_path/cfg['output_directory']
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver.sys,'pycache_prefix',str(out/'_fresh_no_write_cache'))
    monkeypatch.setattr(driver.sys,'dont_write_bytecode',True)
    def forbidden(*args,**kwargs): raise AssertionError('Private loader must not run')
    monkeypatch.setattr(fit_data,'load_completed_fit_tracks',forbidden)
    if failure=='remote':
        def unavailable(*args,**kwargs): raise RuntimeError('remote unavailable')
        monkeypatch.setattr(driver,'verify_registration',unavailable)
        with pytest.raises(RuntimeError,match='remote unavailable'):
            driver.worker(SimpleNamespace(preregistration_commit='a'*40,git_proxy=''),cfg,{'branch':'research/cvpr-trackocd-autonomous-v2'})
    else:
        monkeypatch.setattr(driver,'verify_registration',lambda *a,**k:{'sha':'a'*40})
        monkeypatch.setattr(driver,'SOURCES',('dummy_source.py',))
        (tmp_path/'dummy_source.py').write_bytes(b'current')
        monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**k:b'different')
        with pytest.raises(ValueError,match='Exact registered'):
            driver.worker(SimpleNamespace(preregistration_commit='a'*40,git_proxy=''),cfg,{'branch':'research/cvpr-trackocd-autonomous-v2'})
    assert not out.exists()
