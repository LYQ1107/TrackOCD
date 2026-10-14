from pathlib import Path
from types import SimpleNamespace

from src.trackocd_cvpr.M_endpoint_source_compatibility import CURRENT_SOURCES
from scripts.trackocd_cvpr.current_source_M_endpoint_refresh import configuration

ROOT = Path(__file__).resolve().parents[2]


def test_current_source_refresh_has_new_namespace_and_frozen_protocol():
    args = SimpleNamespace(variant='relative_quarantine', seed=1027)
    cfg = configuration(args)
    assert cfg['output_directory'].endswith(
        'M_GT_window_current_source_v1_relative_quarantine_s1027')
    assert cfg['optimizer_steps'] == 200
    assert cfg['trainable_parameters_each'] == 161
    assert cfg['scientific_M_trial'] is True
    assert cfg['full_scientific_fit_automatic_start'] is False
    assert all((ROOT / path).is_file() for path in CURRENT_SOURCES)


def test_refresh_worker_is_not_an_automatic_development_launcher():
    source = (ROOT / 'scripts/trackocd_cvpr/current_source_M_endpoint_refresh.py').read_text()
    assert 'original.worker(args, cfg, safety)' in source
    assert 'automatic_full_M_development_or_Goal_acceptance' in source
    assert 'M_GT_window_current_source_v1' in source
