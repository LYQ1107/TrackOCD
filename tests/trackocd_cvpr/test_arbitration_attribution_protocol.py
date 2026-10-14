import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.arbitration_attribution_protocol import configuration,CONFIG,EXPECTED

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all4_public_complete_window_gates_one_readonly_all5_policy_pilot_without_weights():
    cfg = configuration(ROOT)
    assert cfg['models'] == ['initial_Raw','calendar1027','calendar1028','calendar1029','ONE_fullpolicy']
    assert (cfg['physical_tracks'],cfg['observations'],cfg['metadata_frames'],cfg['empty_frames']) == (54,1268,374,1)
    assert cfg['optimizer_steps'] == 0 and cfg['all13_private_snapshot_fingerprints_before_policy_input']
    assert 'optimizer' not in cfg and 'learning_rate' not in cfg and 'schedule' not in cfg and 'model' not in cfg
    assert cfg['full_normal_reference_event_commit_state_parity_per_model']
    assert cfg['full_ID_reverse_and_nonvacuous_future_prefix_probes_per_model']


@pytest.mark.parametrize('key,value',[
    ('models',['ONE_fullpolicy']),('observations',100),('physical_tracks',44),('optimizer_steps',1),
    ('shadow_exclusion_writes_or_retroactive_GT_history_edit',True),('dev_final_Val_Test_original_pixels_or_new_nonKnown_GT_read',True),
    ('labels_provided_to_scorer_observer_or_support_exclusion',True),('wall_seconds_ceiling',6000),
    ('full_normal_reference_event_commit_state_parity_per_model',False),
    ('full_ID_reverse_and_nonvacuous_future_prefix_probes_per_model',False),('maximum_actual_pilot_attempts',2),
])
def test_protocol_relaxations_refuse_before_any_private_endpoint_or_label(tmp_path,key,value):
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({**EXPECTED,key:value}))
    with pytest.raises(ValueError,match='Exact finite all5'):configuration(tmp_path)


def test_missing_actual_window_is_not_authority_to_load_private_weights(tmp_path):
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps(EXPECTED))
    with pytest.raises(FileNotFoundError):configuration(tmp_path)


def test_owned_supervisor_requires_complete_result_and_preserves_prior_attempt(tmp_path,monkeypatch):
    import scripts.trackocd_cvpr.pilot_arbitration_attribution as driver
    commands = []
    class Child:
        def poll(self):return 0
        def wait(self,timeout):return 0
    monkeypatch.setattr(driver,'ROOT',tmp_path); monkeypatch.setattr(driver,'busy',lambda:False)
    monkeypatch.setattr(driver,'source_gate',lambda *a:None)
    monkeypatch.setattr(driver.runtime,'resources_ok',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'Popen',lambda cmd,**kw:(commands.append(cmd),Child())[1])
    args = SimpleNamespace(preregistration_commit='f'*40,git_proxy='loopback')
    cfg = {'output_directory':'new_pilot','rss_bytes_ceiling':1,'output_bytes_ceiling':999}
    assert driver.supervise(args,cfg,{'system_ram_reserve_fraction':0.}) == 1
    assert '--worker' in commands[0] and commands[0][commands[0].index('--preregistration-commit')+1] == 'f'*40
    t = json.loads((tmp_path/'new_pilot_supervisor_attempt_1.json').read_text())
    assert not t['successful_owned_execution'] and not t['complete_result_present']
    with pytest.raises(ValueError,match='Preserve every prior'):driver.supervise(args,cfg,{'system_ram_reserve_fraction':0.})


def test_complete_original_input_source_closure_and_nonworker_persistent_command():
    from scripts.trackocd_cvpr.pilot_arbitration_attribution import SOURCES
    from scripts.trackocd_cvpr.evaluate_raw_membership_development import SOURCES as PRIOR
    from scripts.trackocd_cvpr.launch_owned_arbitration_attribution import supervisor_command
    assert set(PRIOR) <= set(SOURCES) and all((ROOT/p).is_file() for p in SOURCES)
    cmd = supervisor_command(SimpleNamespace(preregistration_commit='f'*40,git_proxy='loopback'))
    assert '--worker' not in cmd and '--control' not in cmd and '--seed' not in cmd
