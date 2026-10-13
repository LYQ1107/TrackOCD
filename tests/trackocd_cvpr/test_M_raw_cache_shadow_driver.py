import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr import shadow_M_raw_cache as driver

ROOT = Path(__file__).resolve().parents[2]


def test_missing_source_cohort_stops_before_network_or_private_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    def forbidden(*a, **kw): raise AssertionError('must fail before network or private inputs')
    monkeypatch.setattr(driver, 'verify_registration', forbidden)
    with pytest.raises(FileNotFoundError): driver.source_gate(SimpleNamespace(), {}, {})
    assert not list(tmp_path.iterdir())


def test_full_policy_shadow_registered_scope_is_all9_and_two_zero_fit_not_actual_dev_gain():
    cfg = json.loads((ROOT / driver.CONFIG).read_text())
    assert cfg['training_seeds'] == [1027, 1028, 1029] and cfg['order_seed'] == 1027
    assert cfg['models'] == ['Mean_latest', 'StateEMA_latest', 'relative_quarantine', 'relative_weighted', 'pooled_weighted']
    assert not cfg['full72development_parity_speedup_purity_or_Goal_acceptance']
    assert not cfg['old_failure_overwritten_or_guards_relaxed']
    assert len(driver.SOURCES) == len(set(driver.SOURCES))
    for path in driver.SOURCES: assert (ROOT / path).is_file()
    assert 'src/trackocd_cvpr/qualified_raw_cache_scope.py' in driver.SOURCES


def test_trace_hash_includes_private_event_and_ledger_contents_not_a_count_only_check():
    a = [('synthetic', b'actual-score-and-prototype-bytes')]
    assert driver.trace_digest(a) == driver.trace_digest(a)
    assert driver.trace_digest(a) != driver.trace_digest([('synthetic', b'other-score-and-prototype-bytes')])
