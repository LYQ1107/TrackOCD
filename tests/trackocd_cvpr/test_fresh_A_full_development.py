import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import scripts.trackocd_cvpr.evaluate_fresh_A_development as driver
from src.trackocd_cvpr.fresh_A_baseline import fresh_model
from src.trackocd_cvpr.fresh_A_development_probes import numeric_probes

ROOT=Path(__file__).resolve().parents[2]


def test_all15_absent_refuses_before_remote_or_private_checkpoint(monkeypatch,tmp_path):
    cfg=driver.configuration(ROOT)
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    def forbidden(*a,**k):raise AssertionError('No network or private reads on incomplete15 cohort')
    monkeypatch.setattr(driver,'verify_registration',forbidden)
    with pytest.raises(ValueError,match='All15'):
        driver.source_gate(SimpleNamespace(preregistration_commit='a'*40,git_proxy=''),cfg,{'branch':'research/cvpr-trackocd-autonomous-v2'})


@pytest.mark.parametrize('key,value',[('training_seeds',[1027]),('physical_tracks',100),
    ('prefixes',[1,16]),('optimizer_steps',1),('select_model_or_checkpoint',True),('wall_seconds_ceiling',6000),
    ('final_Val_Test_new_nonKnown_GT_vocabulary_or_original_pixels_allowed',True)])
def test_frozen_full_development_cannot_silently_reduce_or_expand(tmp_path,key,value):
    cfg=json.loads((ROOT/driver.CONFIG).read_text());cfg[key]=value
    p=tmp_path/driver.CONFIG;p.parent.mkdir(parents=True);p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):driver.configuration(tmp_path)


def test_worker_failed_source_gate_precedes_private_endpoint_hash_and_output(monkeypatch,tmp_path):
    cfg=driver.configuration(ROOT);out=tmp_path/cfg['output_directory']
    monkeypatch.setattr(driver,'ROOT',tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(driver.sys,'dont_write_bytecode',True)
    monkeypatch.setattr(driver.sys,'pycache_prefix',str(out/'_fresh_no_write_cache'))
    def unavailable(*a,**k):raise RuntimeError('fresh remote/source gate failed')
    def private_forbidden(*a,**k):raise AssertionError('No private endpoints on failed live gate')
    monkeypatch.setattr(driver,'source_gate',unavailable);monkeypatch.setattr(driver,'verified_cohort',private_forbidden)
    with pytest.raises(RuntimeError,match='fresh remote/source'):driver.worker(SimpleNamespace(),cfg,{})
    assert not out.exists()


@pytest.mark.parametrize('family',['A1_ADAPTER','A2_EVIDENCE','A2_CAPACITY_CONTROL'])
def test_native_actual_changed_toy_future_prefix_invariance_not_temporal_science(family):
    spec=importlib.util.spec_from_file_location('fit_fixture',Path(__file__).with_name('test_fit_R_training.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[module.track(10,0,17),module.track(11,0,19)]
    model=fresh_model(family,1027).eval().requires_grad_(False)
    proof=numeric_probes(model,rows,lambda:None)
    assert proof['future_rows_mutated']>0 and proof['actual_first_prefix_future_invariant']
    assert proof['actual_numeric_prefix_routing_rename_invariant'] and proof['compressed_time_values_changed']
    assert proof['static_capacity_is_not_claimed_strictly_timeless']
    assert not proof['temporal_benefit_or_Goal_accepted']


def test_sources_complete_no_private_cohort_files_in_source_registration():
    for path in driver.SOURCES:assert (ROOT/path).is_file(),path
    assert len(driver.SOURCES)==len(set(driver.SOURCES))
    assert not any('private_step200_checkpoint' in p for p in driver.SOURCES)
    assert 'src/trackocd_cvpr/fresh_A_development_probes.py' in driver.SOURCES
    assert driver.configuration(ROOT)['prefixes']==[1,2,4,8,16,'all_available']
