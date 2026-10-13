import copy
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
import scripts.trackocd_cvpr.fit_fresh_A_fixed200 as driver
from src.trackocd_cvpr.fresh_A_formal_fit import (
    CONFIG,PREFIX,configuration,verify_pilot,active_research_workers,cohort_sources,verified_cohort)

ROOT=Path(__file__).resolve().parents[2]


def test_actual_five_family_pilot_qualified_without_private_weights_read():
    cfg=configuration(ROOT,'A1_ADAPTER',1027);proof=verify_pilot(ROOT,cfg)
    assert proof['all_five_actual_qualified'] and not proof['pilot_weights_read']
    assert cfg['output_directory']==PREFIX+'_A1_ADAPTER_s1027'
    assert cfg['optimizer_steps']==200 and not cfg['pilot_or_historical_checkpoint_warmstart']
    assert len(cohort_sources())==45 and len(set(cohort_sources()))==45
    # Actual formal inputs are still absent, never infer matrix from pilot.
    with pytest.raises(ValueError,match='All15'):verified_cohort(ROOT)


@pytest.mark.parametrize('field,value',[('optimizer_steps',10),('training_seeds',[1027]),
    ('checkpoint_selection','best dev'),('pilot_or_historical_checkpoint_warmstart',True),
    ('parallel_fit_with_live_owned_M_eval_or_other_fresh_A_fit',True),('wall_seconds_ceiling',6000)])
def test_config_no_scope_checkpoint_seed_or_guard_relaxation(tmp_path,field,value):
    cfg=json.loads((ROOT/CONFIG).read_text());cfg[field]=value
    p=tmp_path/CONFIG;p.parent.mkdir(parents=True);p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):configuration(tmp_path,'A1_ADAPTER',1027)


def test_live_kernel_cmdline_not_old_receipt_prevents_parallel_fit(tmp_path):
    def process(pid,words,state='S'):
        p=tmp_path/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(words)+b'\0')
        (p/'stat').write_text(f'{pid} (python) {state} 1 2')
    own_script=str(ROOT/driver.SCRIPT).encode()
    process(10,[b'python',own_script])
    process(11,[b'python',str(ROOT/'scripts/trackocd_cvpr/evaluate_M_development_RAM_retry1.py').encode(),b'--worker'])
    process(12,[b'python',own_script],state='Z')
    process(13,[b'python',b'/other_project/evaluate_M_development.py',b'--worker'])
    assert active_research_workers(ROOT,proc_root=tmp_path,own_pid=10)==(11,)


def test_supervisor_busy_guard_before_network_or_private_spawn(monkeypatch):
    cfg=configuration(ROOT,'A2_EVIDENCE',1027)
    monkeypatch.setattr(driver,'active_research_workers',lambda root:(123,))
    def forbidden(*a,**k):raise AssertionError('No network, new child or private input while active M')
    monkeypatch.setattr(driver,'source_gate',forbidden);monkeypatch.setattr(driver.subprocess,'Popen',forbidden)
    with pytest.raises(RuntimeError,match='active'):
        driver.supervise(SimpleNamespace(family='A2_EVIDENCE',seed=1027),cfg,{})


def test_worker_failed_live_gate_does_not_create_output_or_read_private_data(monkeypatch,tmp_path):
    cfg=configuration(ROOT,'A1_GEOMETRY_1',1028);out=tmp_path/cfg['output_directory']
    monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(driver.sys,'dont_write_bytecode',True)
    monkeypatch.setattr(driver.sys,'pycache_prefix',str(out/'_fresh_no_write_cache'))
    def unavailable(*a,**k):raise RuntimeError('fresh gate failed')
    monkeypatch.setattr(driver,'source_gate',unavailable)
    with pytest.raises(RuntimeError,match='fresh gate failed'):driver.worker(SimpleNamespace(),cfg,{})
    assert not out.exists()


def test_all_sources_exist_and_complete_actual_pilot_receipts_gated():
    for path in driver.SOURCES:assert (ROOT/path).is_file(),path
    assert len(driver.SOURCES)==len(set(driver.SOURCES))
    assert any(path.endswith('CPU_visibility_retry1/result.json') for path in driver.SOURCES)
    assert CONFIG in driver.SOURCES and 'src/trackocd_cvpr/fresh_A_formal_fit.py' in driver.SOURCES


def synthetic_cohort(root,monkeypatch):
    # Fabricated receipt structure tests ONLY; never actual seeded fits/evidence.
    from src.trackocd_cvpr.fresh_A_baseline import FAMILIES
    source='qualified_dummy_source.py';(root/source).write_bytes(b'synthetic source')
    monkeypatch.setattr(driver,'SOURCES',(source,))
    sha=lambda value:hashlib.sha256(value.encode()).hexdigest()
    for family in FAMILIES:
        for seed in (1027,1028,1029):
            out=root/f'{PREFIX}_{family}_s{seed}';out.mkdir(parents=True)
            proof={'sha':'a'*40,'method':'fresh_git_ls_remote','ref':'refs/heads/research/cvpr-trackocd-autonomous-v2'}
            launch={'preregistration_commit':'a'*40,'live_remote_registration_proof':proof,
                    'source_sha256':{source:sha('synthetic source')}}
            trial={'family':family,'seed':seed,'stage':'Fixed200','completed_optimizer_steps':200,
                'parameter_count':541889 if family.startswith('A2_') else 525056,
                'model_parameter_sha256_before':sha('before'+family+str(seed)),
                'model_parameter_sha256_after':sha('after'+family+str(seed)),
                'adapter_initialization_sha256':sha('adapter'+str(seed)),
                'batch_calendar_sha256':sha('calendar'+str(seed)),'training_GPU_hours':0,
                'loss_curve':[{'step':i} for i in range(1,201)],
                'scope':{'physical_tracks':1401,'eligible_known_physical_tracks':1158,
                         'masked_no_fit5_target_tracks':243,'supervised_classes':5}}
            result={'status':'COMPLETED_ONE_FRESH_FIXED200_FIT_NOT_ALL15_DEV_OR_GOAL_ACCEPTANCE',
                'trial':trial,'live_remote_registration_proof':proof,'pilot_or_historical_checkpoint_warmstart':False,
                'dev_final_Val_Test_inputs_used':False,'formal15fits_dev_or_Goal_accepted':False,
                'resources':{'CUDA_initialized':False,'training_GPU_hours':0},
                'checkpoint_selection':'Fixed200, never dev/final/Val/Test selection',
                'private_checkpoints':[{'step':i,'bytes':100,'sha256':'b'*64,'reopen_exact':True} for i in (20,50,100,200)]}
            terminal={'successful_owned_execution':True,'owned_worker_exit_code':0,'guard_stop_reason':None,
                'full_result_present':True,'last_persisted_completed_optimizer_steps':200,'training_GPU_hours':0,
                'CPU_compute_workers':1,'peak_identified_native_driver_helpers':1,'peak_sampled_worker_threads':2}
            for path,obj in ((out/'launch.json',launch),(out/'result.json',result),
                             (out.with_name(out.name+'_supervisor_attempt_1.json'),terminal)):
                path.write_text(json.dumps(obj))


def test_synthetic_complete_cohort_and_missing_seed_refused_not_actual_fits(tmp_path,monkeypatch):
    synthetic_cohort(tmp_path,monkeypatch);cohort=verified_cohort(tmp_path)
    assert len(cohort['trials'])==15 and not cohort['full_dev_or_Goal_accepted']
    assert not cohort['private_endpoint_hashes_checked']
    p=tmp_path/(PREFIX+'_A2_EVIDENCE_s1029_supervisor_attempt_1.json')
    t=json.loads(p.read_text());t['owned_worker_exit_code']=-9;t['successful_owned_execution']=False;p.write_text(json.dumps(t))
    with pytest.raises(ValueError):verified_cohort(tmp_path)


@pytest.mark.parametrize('field,value',[('batch_calendar_sha256','0'*64),('adapter_initialization_sha256','0'*64),
    ('completed_optimizer_steps',10),('parameter_count',50113)])
def test_synthetic_same_seed_actual_parity_not_merely_seed_name(tmp_path,monkeypatch,field,value):
    synthetic_cohort(tmp_path,monkeypatch)
    p=tmp_path/(PREFIX+'_A2_EVIDENCE_s1028/result.json');d=json.loads(p.read_text());d['trial'][field]=value;p.write_text(json.dumps(d))
    with pytest.raises(ValueError):verified_cohort(tmp_path)


def test_synthetic_partial_checkpoint_source_or_current_frozen_dependency_refused(tmp_path,monkeypatch):
    synthetic_cohort(tmp_path,monkeypatch)
    p=tmp_path/(PREFIX+'_A2_CAPACITY_CONTROL_s1029/result.json');d=json.loads(p.read_text())
    d['private_checkpoints']=d['private_checkpoints'][:-1];p.write_text(json.dumps(d))
    with pytest.raises(ValueError):verified_cohort(tmp_path)
    synthetic_cohort_replacement=copy.deepcopy(d);synthetic_cohort_replacement['private_checkpoints'].append({'step':200,'bytes':100,'sha256':'b'*64,'reopen_exact':True})
    p.write_text(json.dumps(synthetic_cohort_replacement));(tmp_path/'qualified_dummy_source.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='dependency changed'):verified_cohort(tmp_path)
