import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.trackocd_cvpr.evaluate_C_development import CONFIG, recovery_configuration
from scripts.trackocd_cvpr.launch_owned_C_development import supervisor_command

ROOT = Path(__file__).resolve().parents[2]


def test_transport_retry_changes_only_namespace_not_science_or_guards():
    original = json.loads((ROOT / CONFIG).read_text())
    args = SimpleNamespace(recovery_attempt='transport_retry1', model='Raw_Top1', seed=1027)
    retry, evidence = recovery_configuration(ROOT, original, args)
    assert {k: v for k, v in original.items() if k != 'output_prefix'} == {
        k: v for k, v in retry.items() if k != 'output_prefix'}
    assert retry['output_prefix'] != original['output_prefix']
    assert original['wall_seconds_ceiling'] == retry['wall_seconds_ceiling'] == 3000
    assert original['optimizer_steps'] == retry['optimizer_steps'] == 0
    assert len(evidence) == 4
    unchanged, evidence = recovery_configuration(ROOT, original, SimpleNamespace(recovery_attempt=None))
    assert unchanged == original and evidence == ()


@pytest.mark.parametrize('model,seed,attempt', [('Raw_Top8', 1027, 'transport_retry1'),
    ('Raw_Top1', 1028, 'transport_retry1'), ('Top1_conditioned', 1027, 'transport_retry1'),
    ('Raw_Top1', 1027, 'another_retry')])
def test_recovery_is_not_trial_replacement_or_model_selection(model, seed, attempt):
    with pytest.raises(ValueError):
        recovery_configuration(ROOT, json.loads((ROOT / CONFIG).read_text()),
                               SimpleNamespace(model=model, seed=seed, recovery_attempt=attempt))


def test_persistent_supervisor_command_carries_exact_recovery_and_registration():
    args = SimpleNamespace(model='Raw_Top1', seed=1027, preregistration_commit='a' * 40,
                           git_proxy='http://127.0.0.1:17890', recovery_attempt='transport_retry1')
    command = supervisor_command(args)
    assert '--worker' not in command  # launch guarded supervisor, never unguarded worker
    assert command[command.index('--recovery-attempt') + 1] == args.recovery_attempt
    assert command[command.index('--preregistration-commit') + 1] == args.preregistration_commit
    args.recovery_attempt = None
    assert '--recovery-attempt' not in supervisor_command(args)


def test_interrupted_attempt_is_not_promoted_to_a_terminal_success():
    path = ROOT / 'outputs/trackocd_cvpr/cvpr3/C_GT_causal_dev_v1_Raw_Top1_interrupted_observation.json'
    evidence = json.loads(path.read_text())
    assert evidence['execution_tool_exit_code'] == 143 and evidence['owned_worker_exit_code'] is None
    assert not evidence['complete_result_present'] and not evidence['normal_supervisor_terminal_receipt_present']
    assert not evidence['termination_source_or_guard_reason_proven']


def test_guarded_supervisor_forwards_recovery_to_fresh_worker(tmp_path, monkeypatch):
    from scripts.trackocd_cvpr import qualify_policy_training_input as script
    commands = []
    class Child:
        def poll(self): return 0
        def wait(self, timeout): return 0
    monkeypatch.setattr(script, 'ROOT', tmp_path)
    monkeypatch.setattr(script, 'resources_ok', lambda *args: None)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda cmd, **kw: (commands.append(cmd), Child())[1])
    args = SimpleNamespace(model='Raw_Top1', seed=1027, preregistration_commit='source',
                           git_proxy='', recovery_attempt='transport_retry1')
    assert script.supervise(args, {'output_directory': 'new_attempt', 'output_bytes_ceiling': 9999}, {},
                            worker_script='scripts/trackocd_cvpr/evaluate_C_development.py') == 0
    assert commands[0][commands[0].index('--recovery-attempt') + 1] == 'transport_retry1'
