import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.trackocd_cvpr.raw_known_membership_control import NamedKnownLogit

ROOT = Path(__file__).resolve().parents[2]


def test_real_pilot_source_closure_and_fixed_cfg_do_not_enable_formal_dev_or_oldweights():
    from scripts.trackocd_cvpr.pilot_raw_known_membership_controls import SOURCES,configuration,OUT
    from scripts.trackocd_cvpr.evaluate_spread_C_development import SOURCES as OLD
    cfg = configuration(ROOT)
    assert set(OLD) <= set(SOURCES) and all((ROOT/p).is_file() for p in SOURCES)
    assert cfg['pilot']['calendar_updates'] == cfg['pilot']['full_policy_maxiter'] == 50
    assert cfg['pilot']['full_policy_maxfun'] == 250 and cfg['pilot']['formal_warmstart'] is False
    assert cfg['maximum_actual_pilot_attempts'] == 1 and cfg['output_directory'] == OUT
    assert not cfg['automatic_private_load_training_development_or_expansion']


def test_actual_synthetic_private_checkpoint_reopen_distinguishes_AdamW_RNG_from_solver_record(tmp_path):
    from scripts.trackocd_cvpr.pilot_raw_known_membership_controls import save_own_checkpoint
    model = NamedKnownLogit(); optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    loss = model(.8)**2; loss.backward(); optimizer.step()
    rng = {'bit_generator':'PCG64','state':{'state':123,'inc':456},'has_uint32':0,'uinteger':0}
    a = save_own_checkpoint(tmp_path,'calendar_matched',model,lambda:None,optimizer=optimizer,step=50,rng=rng)
    b = save_own_checkpoint(tmp_path,'full_policy',model,lambda:None,step=3,solver={'solver_success':True,'iterations':3})
    assert a['reopen_exact'] and a['AdamW_and_RNG_reopened']
    assert b['reopen_exact'] and not b['AdamW_and_RNG_reopened'] and not b['solver_internal_LBFGS_resume_state_claimed']
    assert (tmp_path/'private_calendar_matched_pilot.pt').stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError,match='Never overwrite'):
        save_own_checkpoint(tmp_path,'calendar_matched',model,lambda:None,optimizer=optimizer,step=50,rng=rng)


def test_existing_output_and_live_worker_refuse_before_source_or_Popen(tmp_path,monkeypatch):
    import scripts.trackocd_cvpr.pilot_raw_known_membership_controls as driver
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    (tmp_path/driver.OUT).mkdir(parents=True)
    monkeypatch.setattr(driver,'source_gate',lambda *a:pytest.fail('Should refuse before source gate'))
    with pytest.raises(ValueError,match='One actual pilot'):
        driver.supervise(None,{}, {})
    # A new synthetic root, not removal of any original output/asset.
    other = tmp_path/'fresh'; other.mkdir(); monkeypatch.setattr(driver,'ROOT',other)
    monkeypatch.setattr(driver,'busy',lambda:True)
    with pytest.raises(RuntimeError,match='Observe live'):
        driver.supervise(None,{}, {})


def test_supervisor_explicit_worker_and_missing_result_is_not_completed(tmp_path,monkeypatch):
    import scripts.trackocd_cvpr.pilot_raw_known_membership_controls as driver
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self,timeout): return 0
    monkeypatch.setattr(driver,'ROOT',tmp_path); monkeypatch.setattr(driver,'busy',lambda:False)
    monkeypatch.setattr(driver,'source_gate',lambda *a:None)
    monkeypatch.setattr(driver.prior.runtime,'resources_ok',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'Popen',lambda cmd,**kw:(commands.append(cmd),Child())[1])
    args = SimpleNamespace(preregistration_commit='f'*40,git_proxy='loopback')
    assert driver.supervise(args,{'rss_bytes_ceiling':1,'output_bytes_ceiling':99999},{'system_ram_reserve_fraction':0.}) == 1
    cmd = commands[0]
    assert '--worker' in cmd and '--preregistration-commit' in cmd and '--seed' not in cmd
    r = json.loads((tmp_path/driver.OUT).with_name(Path(driver.OUT).name+'_supervisor_attempt_1.json').read_text())
    assert not r['complete_result_present'] and not r['successful_owned_execution']
