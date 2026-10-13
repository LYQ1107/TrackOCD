import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.raw_membership_formal_protocol import configuration,CONFIG,EXPECTED
from src.trackocd_cvpr.raw_known_membership_control import calendar_matched_fit
from src.trackocd_cvpr.spread_known_calendar import freeze_calendar
from tests.trackocd_cvpr.test_raw_known_membership_controls import fixture

ROOT=Path(__file__).resolve().parents[2]


def test_actual_pilot_public_prerequisite_four_new_namespaces_no_fake_fullpolicy_seeds():
    names={configuration(ROOT,'calendar_matched',s)['output_directory'] for s in (1027,1028,1029)}
    names.add(configuration(ROOT,'full_policy',None)['output_directory']); assert len(names)==4
    for control,seed in (('full_policy',1027),('calendar_matched',None),('calendar_matched',True)):
        with pytest.raises(ValueError,match='Exact four fresh'):
            configuration(ROOT,control,seed)


@pytest.mark.parametrize('field,value', [('calendar_optimizer_steps',201),('old_or_pilot_weight_warmstart',True),
    ('full_policy_independent_seed_runs',True),('dev_selected_model_checkpoint_threshold_or_control',True)])
def test_formal_scope_budget_selection_and_warmstart_relaxation_fail_before_private_input(tmp_path,field,value):
    p=tmp_path/CONFIG;p.parent.mkdir(parents=True);p.write_text(json.dumps({**EXPECTED,field:value}))
    with pytest.raises(ValueError,match='Exact four fresh'):
        configuration(tmp_path,'calendar_matched',1027)


def test_actual_synthetic_fixed200_preserves_four_full_clocks_and_all_fixed_snapshots():
    inputs=fixture();plan=freeze_calendar(*inputs[:2],1029,4,50,lambda:None);saved=[]
    _,_,r,_=calendar_matched_fit(*inputs,1029,frozen_calendar=plan,episodes=4,updates_per_episode=50,
        checkpoint_steps=[20,50,100,200],guard=lambda:None,progress=lambda row:None,
        checkpoint=lambda model,opt,step,rng:saved.append(step))
    assert saved==[20,50,100,200] and r['optimizer_steps']==200 and len(r['episodes'])==4
    assert r['actual_shared_calendar_sha256']==plan['selected_calendar_sha256']
    assert all((e['physical_observations'],e['metadata_frames'],e['empty_frames'],e['masked_groups'])==(150,51,1,50) for e in r['episodes'])


@pytest.mark.parametrize('control,seed',[('calendar_matched',1028),('full_policy',None)])
def test_formal_supervisor_explicit_choice_and_missing_result_is_not_completed(tmp_path,monkeypatch,control,seed):
    import scripts.trackocd_cvpr.fit_raw_known_membership_formal as driver
    commands=[]
    class Child:
        def poll(self):return 0
        def wait(self,timeout):return 0
    monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setattr(driver,'busy',lambda:False)
    monkeypatch.setattr(driver,'source_gate',lambda *args:None)
    monkeypatch.setattr(driver.pilot.prior.runtime,'resources_ok',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'Popen',lambda cmd,**kwargs:(commands.append(cmd),Child())[1])
    args=SimpleNamespace(control=control,seed=seed,preregistration_commit='f'*40,git_proxy='loopback')
    cfg={'output_directory':'formal','rss_bytes_ceiling':1,'output_bytes_ceiling':9999}
    assert driver.supervise(args,cfg,{'system_ram_reserve_fraction':0.})==1
    cmd=commands[0];assert cmd[cmd.index('--control')+1]==control and '--worker' in cmd
    if seed is None:assert '--seed' not in cmd
    else:assert cmd[cmd.index('--seed')+1]==str(seed)
    r=json.loads((tmp_path/'formal_supervisor_attempt_1.json').read_text())
    assert not r['complete_result_present'] and not r['successful_owned_execution']


def test_formal_dependency_closure_preserves_actual_qualified_pilot_numeric_paths():
    from scripts.trackocd_cvpr.fit_raw_known_membership_formal import SOURCES
    from scripts.trackocd_cvpr.pilot_raw_known_membership_controls import SOURCES as PILOT
    from src.trackocd_cvpr.raw_membership_pilot_qualification import HASHES
    assert set(PILOT)<=set(SOURCES) and set(HASHES)<=set(SOURCES)
    assert all((ROOT/p).is_file() for p in SOURCES)
