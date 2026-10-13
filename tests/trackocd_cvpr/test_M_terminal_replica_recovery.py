import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.M_terminal_replica_recovery import ORIGINAL, REPLICA, verify_original, verify_replica_result
from src.trackocd_cvpr.memory_transport_recovery import recovery_configuration
from scripts.trackocd_cvpr.launch_owned_M_fit import supervisor_command

ROOT = Path(__file__).resolve().parents[2]


def test_unobserved_exit_preserved_no_fake_normal_terminal_and_namespace_only():
    original = verify_original(ROOT)
    assert original['summary']['optimizer_steps'] == 200
    cfg = {'scientific_M_trial': True, 'variant': 'relative_weighted', 'seed': 1029,
           'output_directory': ORIGINAL, 'unchanged': {'steps': 200, 'seed': 1029, 'loss': 'original'}}
    out, sources = recovery_configuration(ROOT, cfg, 'terminal_replica1')
    assert out['output_directory'] == REPLICA
    assert {k: v for k, v in out.items() if k != 'output_directory'} == {k: v for k, v in cfg.items() if k != 'output_directory'}
    assert not any('private_' in path for path in sources)


def test_any_better_or_worse_metric_parameter_history_optimizer_RNG_endpoint_refuses_replica():
    old = verify_original(ROOT)
    assert verify_replica_result(ROOT, old)['entire_fixed_summary_endpoint_optimizer_RNG_and_private_history_exact']
    for field in ('summary', 'endpoint', 'private_prediction_history'):
        altered = copy.deepcopy(old)
        if field == 'summary': altered[field]['model_parameter_sha256_after'] = 'different-even-if-better'
        else: altered[field]['sha256'] = 'different-even-if-better'
        with pytest.raises(ValueError, match='difference'): verify_replica_result(ROOT, altered)


def test_persistent_launch_keeps_original_supervised_worker_recovery_flags_and_no_fake_worker_mode():
    args = SimpleNamespace(variant='relative_weighted', seed=1029, recovery_attempt='terminal_replica1',
                          preregistration_commit='public-test-only', git_proxy='')
    cmd = supervisor_command(args)
    assert '--worker' not in cmd and '--config' in cmd
    assert cmd[-2:] == ['--recovery-attempt', 'terminal_replica1']
