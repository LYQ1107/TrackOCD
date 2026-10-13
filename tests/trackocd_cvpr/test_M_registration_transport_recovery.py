import copy
import json
from pathlib import Path
import pytest
from src.trackocd_cvpr import memory_transport_recovery as recovery
from src.trackocd_cvpr.M_development_cohort import cohort_sources

ROOT = Path(__file__).resolve().parents[2]


def config():
    return {'scientific_M_trial': True, 'variant': 'pooled_weighted', 'seed': 1028,
        'output_directory': recovery.ORIGINAL, 'optimizer_steps': 200,
        'unchanged_loss_and_guards': {'window': 'original-fixed', 'wall': 3000}}


def test_actual_failed_terminal_original_source_and_window_bound_no_private_assets():
    record = recovery.verify_original_preinput_failure(ROOT)
    assert record['original_tool_terminal_exit_code'] == 1
    assert not (ROOT / recovery.ORIGINAL).exists()
    terminal = json.loads((ROOT / recovery.FAILED_TERMINAL).read_text())
    assert terminal['owned_worker_exit_code'] == 1 and not terminal['successful_owned_execution']
    assert terminal['last_persisted_completed_optimizer_steps'] == 0


def test_only_namespace_changes_never_resumes_or_overwrites():
    cfg = config(); previous = copy.deepcopy(cfg)
    result, sources = recovery.recovery_configuration(ROOT, cfg, 'transport_retry1')
    assert cfg == previous and result['output_directory'] == recovery.RETRY
    assert {k: v for k, v in result.items() if k != 'output_directory'} == {k: v for k, v in previous.items() if k != 'output_directory'}
    assert sources == recovery.RECOVERY_SOURCES
    assert recovery.recovery_configuration(ROOT, cfg, None) == (cfg, ())
    for field, value in (('seed', 1029), ('variant', 'relative_weighted'), ('scientific_M_trial', False)):
        altered = dict(cfg); altered[field] = value
        with pytest.raises(ValueError, match='exact fixed'): recovery.recovery_configuration(ROOT, altered, 'transport_retry1')


def test_cohort_has_exactly_one_recovery_not_failed_attempt_instead_of_missing_seed():
    paths = cohort_sources()
    assert len(paths) == len(set(paths)) == 27
    assert sum(recovery.RETRY in p for p in paths) == 3
    assert recovery.FAILED_TERMINAL not in paths
    assert recovery.FAILED_TERMINAL in recovery.RECOVERY_SOURCES
    assert sum('s1029' in p for p in paths) == 9
