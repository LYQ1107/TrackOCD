import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr import continuity_policy_protocol as protocol
from scripts.trackocd_cvpr import pilot_continuity_reuse_policy as worker
from scripts.trackocd_cvpr.launch_owned_continuity_policy import supervisor_command

ROOT=Path(__file__).resolve().parents[2]


def no_private_stat(monkeypatch):
    original=Path.stat
    def guard(p,*a,**k):
        if p.suffix in ('.pt','.pth','.npz'):raise AssertionError('Prospective public gate touched private input')
        return original(p,*a,**k)
    monkeypatch.setattr(Path,'stat',guard)


def test_actual_public_all5_prerequisite_and_complete_protocol_without_private_data(monkeypatch):
    no_private_stat(monkeypatch)
    cfg=protocol.configuration(ROOT,'initial_Raw')
    assert cfg['physical_tracks']==54 and cfg['observations']==1268 and cfg['metadata_frames']==374
    assert cfg['arms']==list(protocol.MODES) and cfg['order_seeds']==[1027,1028,1029]
    assert cfg['all13_private_snapshot_fingerprints_before_policy_input']
    assert cfg['origin_aware_primary_reuse_excludes_routing_continuation']
    assert not cfg['GT_origin_or_future_truth_to_policy_or_memory_repair']
    assert len(worker.SOURCES)==len(set(worker.SOURCES)) and all((ROOT/p).is_file() for p in worker.SOURCES)


def test_registered_source_and_actual_human_gate_bind_all229_paths_without_private_reads(monkeypatch):
    cfg=protocol.configuration(ROOT,'initial_Raw');calls=[]
    monkeypatch.setattr(worker,'configuration',lambda root,model:cfg)
    monkeypatch.setattr(worker,'verify_registration',lambda *a,**k:{'sha':'registered'})
    def mirror(command,**kwargs):
        assert command[:2]==['git','show'];p=command[2].split(':',1)[1];calls.append(p);return (ROOT/p).read_bytes()
    monkeypatch.setattr(worker.subprocess,'check_output',mirror);no_private_stat(monkeypatch)
    assert worker.source_gate(SimpleNamespace(model='initial_Raw',preregistration_commit='registered',git_proxy=''),cfg,{'branch':'research'})=={'sha':'registered'}
    assert calls==list(worker.SOURCES)


@pytest.mark.parametrize('key,value',[
    ('models',['initial_Raw']),('arms',['full_raw']),('order_seeds',[1027]),('physical_tracks',44),
    ('observations',997),('metadata_frames',373),('empty_frames',0),('K',1),('optimizer_steps',1),
    ('rss_bytes_ceiling',4294967296),('wall_seconds_ceiling',3001),
    ('origin_aware_primary_reuse_excludes_routing_continuation',False),
    ('GT_origin_or_future_truth_to_policy_or_memory_repair',True),
    ('dev_final_Val_Test_pixels_or_new_nonKnown_GT_read',True),
])
def test_mutated_subset_scope_guard_or_truth_policy_refuses_before_prerequisite_or_private_read(tmp_path,monkeypatch,key,value):
    cfg=copy.deepcopy(protocol.EXPECTED);cfg[key]=value;p=tmp_path/protocol.CONFIG;p.parent.mkdir(parents=True);p.write_text(json.dumps(cfg))
    no_private_stat(monkeypatch)
    with pytest.raises(ValueError,match='Exact all5'):protocol.configuration(tmp_path,'initial_Raw')


@pytest.mark.parametrize('key,value',[
    ('status','not_approved'),('video_scope',408),('allowed_training_label_roles',['all']),
    ('new_nonKnown_semantic_GT_or_Novel_vocabulary_authorized',True),
    ('unlabeled_objects_background_or_inferred_Novel_labels',True),
    ('old_51_heldout_Val_Test_DEVplus_Q1_public_new_sealed_and_other_videos_allowed',True),
])
def test_actual_human_scope_gate_refuses_mutation_before_private_followup(tmp_path,monkeypatch,key,value):
    cfg=protocol.configuration(ROOT,'initial_Raw');a=json.loads((ROOT/cfg['approval']).read_text());a[key]=value
    p=tmp_path/cfg['approval'];p.parent.mkdir(parents=True);p.write_text(json.dumps(a))
    monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(worker,'SOURCES',())
    monkeypatch.setattr(worker,'configuration',lambda root,model:cfg)
    monkeypatch.setattr(worker,'verify_registration',lambda *a,**k:{'sha':'registered'})
    no_private_stat(monkeypatch)
    with pytest.raises(ValueError,match='Actual human'):
        worker.source_gate(SimpleNamespace(model='initial_Raw',preregistration_commit='registered',git_proxy=''),cfg,{'branch':'research'})


def test_launcher_forwards_exact_one_model_and_no_worker_or_automatic_next():
    args=SimpleNamespace(model='calendar1028',preregistration_commit='registered',git_proxy='loopback')
    command=supervisor_command(args)
    assert command[command.index('--model')+1]=='calendar1028'
    assert '--worker' not in command and command[command.index('--preregistration-commit')+1]=='registered'
    with pytest.raises(SystemExit):worker.parser().parse_args(['--preregistration-commit','x','--model','selected'])
