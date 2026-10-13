from pathlib import Path
from types import SimpleNamespace
import pytest
import scripts.trackocd_cvpr.pilot_paired_known_rejection as driver

ROOT=Path(__file__).resolve().parents[2]


def test_frozen_pilot_config_and_registered_dependency_closure():
    cfg=driver.configuration(ROOT)
    assert cfg['pilot']['optimizer_steps_each']==50 and cfg['output_directory']==driver.OUT
    assert not cfg['automatic_training_launch']
    assert len(driver.SOURCES)==len(set(driver.SOURCES))
    for path in driver.SOURCES: assert (ROOT/path).is_file(),path
    assert not any('private_' in path for path in driver.SOURCES)


def test_busy_guard_precedes_source_network_or_worker_spawn(monkeypatch,tmp_path):
    cfg=driver.configuration(ROOT); monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'busy',lambda:True)
    def forbidden(*a,**k): raise AssertionError('No network or child while busy')
    monkeypatch.setattr(driver,'source_gate',forbidden); monkeypatch.setattr(driver.subprocess,'Popen',forbidden)
    with pytest.raises(RuntimeError,match='live'): driver.supervise(SimpleNamespace(),cfg,{})


def test_failed_worker_live_gate_before_output_or_private_load(monkeypatch,tmp_path):
    cfg=driver.configuration(ROOT); out=tmp_path/driver.OUT
    monkeypatch.setattr(driver,'ROOT',tmp_path); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(driver.sys,'dont_write_bytecode',True)
    monkeypatch.setattr(driver.sys,'pycache_prefix',str(out/'_fresh_no_write_cache'))
    def fail(*a,**k): raise RuntimeError('fresh source gate failed')
    monkeypatch.setattr(driver,'source_gate',fail)
    with pytest.raises(RuntimeError,match='source gate'): driver.worker(SimpleNamespace(),cfg,{})
    assert not out.exists()


def test_execution_is_pilot_only_not_historical_weight_or_auto_formal_loader():
    source=(ROOT/driver.SCRIPT).read_text()
    assert 'episodes=1,updates_per_episode=50' in source
    assert 'weights_only=True' in source and 'checkpoint_state_equal' in source
    assert 'No network after registration' in source
    assert 'pilot_informative_for_potential_later_formal_registration' in source
