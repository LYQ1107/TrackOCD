import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr.evaluate_M_development import configuration
from src.trackocd_cvpr.M_all_model_fingerprint_development_profile import all_model_configuration
from src.trackocd_cvpr.M_development_RAM_recovery import recovery_configuration, RETRY

ROOT = Path(__file__).resolve().parents[2]


def test_actual_terminal_RAM_stop_recovers_namespace_only_no_limits_or_data_change():
    args = SimpleNamespace(model='relative_quarantine',seed=1027)
    old = all_model_configuration(ROOT,configuration(ROOT,args),args)
    new = recovery_configuration(ROOT,old,args)
    assert new['output_directory'] == RETRY
    assert {k:v for k,v in old.items() if k!='output_directory'} == {k:v for k,v in new.items() if k!='output_directory'}
    assert new['wall_seconds_ceiling']==3000 and new['rss_bytes_ceiling']==3221225472


@pytest.mark.parametrize('model,seed', [('relative_weighted',1027),('relative_quarantine',1028),('pooled_weighted',1029)])
def test_other_trials_cannot_use_this_recorded_RAM_recovery(model,seed):
    args = SimpleNamespace(model=model,seed=seed)
    old = all_model_configuration(ROOT,configuration(ROOT,args),args)
    with pytest.raises(ValueError): recovery_configuration(ROOT,old,args)


def test_no_prior_namespace_or_partial_readback_or_unbounded_retry_loop():
    source=(ROOT/'scripts/trackocd_cvpr/evaluate_M_development_RAM_retry1.py').read_text()
    assert 'return base.main()' in source and 'recovery_configuration(' in source
    assert 'while ' not in source and 'kill(' not in source and 'sleep(' not in source
