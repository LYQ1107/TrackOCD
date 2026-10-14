import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.arbitration_development_adapter import development_targets,development_stream
from src.trackocd_cvpr.arbitration_development_protocol import configuration,CONFIG,EXPECTED
from src.trackocd_cvpr.arbitration_attribution import attribution_stream
from tests.trackocd_cvpr.test_arbitration_attribution import fixture,head

ROOT = Path(__file__).resolve().parents[2]


def test_external_dev_role_binding_adapter_preserves_all_masked_and_semantic_values_not_training():
    labels = {(1,'a'):('Known',100),(1,'b'):('declared_pseudoNovel',200),
              (1,'c'):('outside_eligible_Known',None),(1,'d'):('no_retained_semantic_truth',None)}
    t = development_targets(labels,{100:0})
    assert t == {(1,'a'):('Known',0),(1,'b'):('policy_pseudo',200),(1,'c'):('masked',None),(1,'d'):('masked',None)}
    assert len(t) == len(labels)


@pytest.mark.parametrize('label', [('Known',999),('declared_pseudoNovel',100),('outside_eligible_Known',123),('invented_Novel',44)])
def test_missing_or_new_semantic_binding_or_negative_masks_refuse(label):
    with pytest.raises(ValueError):development_targets({(1,'a'):label},{100:0})


def test_unmodified_observer_actual_events_commits_state_same_with_dev_external_bucket_spelling():
    videos,targets,known,cfg = fixture(); labels = {(1,'10'):('Known',100),(1,'20'):('declared_pseudoNovel',200)}
    h = head(); normal,np = attribution_stream(videos,targets,known,cfg,1027,lambda:None,head=h)
    dev,dp = development_stream(videos,labels,known,{100:0},cfg,1027,lambda:None,head=h)
    assert np == dp and normal == dev
    renamed,rp = development_stream(videos,labels,known,{100:0},cfg,1027,lambda:None,head=h,rename_and_reverse=True)
    assert renamed == dev and [r.events for _,r in rp[0]] == [r.events for _,r in dp[0]] and rp[2] == dp[2]


def test_genuine_policy_pilot_qualifies_full_all5_dev_source_not_private_access_or_selection():
    for model in EXPECTED['models']:
        cfg = configuration(ROOT,model)
        assert (cfg['physical_tracks'],cfg['observations'],cfg['metadata_frames'],cfg['empty_frames']) == (472,8753,2562,116)
        assert cfg['order_seeds'] == [1027,1028,1029] and cfg['optimizer_steps'] == 0
        assert cfg['full_normal_reference_event_commit_state_parity_for_EACH_order']
        assert cfg['full_ID_reverse_and_nonvacuous_future_prefix_probes']
        assert 'optimizer' not in cfg and 'learning_rate' not in cfg


@pytest.mark.parametrize('key,value',[
    ('models',['ONE_fullpolicy']),('order_seeds',[1027]),('physical_tracks',421),('observations',100),
    ('metadata_frames',100),('optimizer_steps',1),('wall_seconds_ceiling',6000),
    ('maximum_actual_attempts_per_model',2),('unmodified_qualified_policy_observer_numeric_source',False),
    ('full_normal_reference_event_commit_state_parity_for_EACH_order',False),
    ('full_ID_reverse_and_nonvacuous_future_prefix_probes',False),
    ('shadow_exclusion_writes_or_retroactive_GT_history_edit',True),
    ('external_dev_kind_bucket_remap_is_training_target_or_new_semantic_value',True),
    ('final_Val_Test_original_pixels_or_new_nonKnown_GT_read',True),
])
def test_subset_guard_probe_selection_or_new_role_relaxation_refuses_before_private_input(tmp_path,key,value):
    p = tmp_path/CONFIG; p.parent.mkdir(parents=True); p.write_text(json.dumps({**EXPECTED,key:value}))
    with pytest.raises(ValueError,match='Exact all5'):configuration(tmp_path,'initial_Raw')


def test_supervisor_forwards_explicit_model_and_never_accepts_missing_result(tmp_path,monkeypatch):
    import scripts.trackocd_cvpr.evaluate_arbitration_development as driver
    commands = []
    class Child:
        def poll(self):return 0
        def wait(self,timeout):return 0
    monkeypatch.setattr(driver,'ROOT',tmp_path); monkeypatch.setattr(driver,'busy',lambda:False)
    monkeypatch.setattr(driver,'source_gate',lambda *a:None); monkeypatch.setattr(driver.runtime,'resources_ok',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'Popen',lambda cmd,**kw:(commands.append(cmd),Child())[1])
    cfg = {'output_directory':'new_dev','rss_bytes_ceiling':1,'output_bytes_ceiling':999}
    args = SimpleNamespace(model='calendar1028',preregistration_commit='f'*40,git_proxy='loopback')
    assert driver.supervise(args,cfg,{'system_ram_reserve_fraction':0.}) == 1
    assert commands[0][commands[0].index('--model')+1] == 'calendar1028'
    t = json.loads((tmp_path/'new_dev_supervisor_attempt_1.json').read_text())
    assert not t['complete_result_present'] and not t['successful_owned_execution']
    with pytest.raises(ValueError,match='Preserve all previous'):driver.supervise(args,cfg,{'system_ram_reserve_fraction':0.})


def test_original_qualified_observer_source_closure_and_boot_command_remain_intact():
    from scripts.trackocd_cvpr.evaluate_arbitration_development import SOURCES
    from scripts.trackocd_cvpr.pilot_arbitration_attribution import SOURCES as PILOT
    from scripts.trackocd_cvpr.launch_owned_arbitration_development import supervisor_command
    assert set(PILOT) <= set(SOURCES) and all((ROOT/p).is_file() for p in SOURCES)
    cmd = supervisor_command(SimpleNamespace(model='ONE_fullpolicy',preregistration_commit='f'*40,git_proxy='loopback'))
    assert cmd[cmd.index('--model')+1] == 'ONE_fullpolicy' and '--worker' not in cmd and '--seed' not in cmd
