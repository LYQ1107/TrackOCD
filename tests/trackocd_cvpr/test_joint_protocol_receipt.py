"""Actual public integer proof/resources; private category witnesses not read."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v1'


def test_actual_joint_registration_and_immutable_source_bindings():
    result = json.loads((OUT / 'result.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == 'a0b7cc192ece6266171da9b1c05cd6ebb2c0e500e0bcf56d6fb72dbe91e20399'
    assert hashlib.sha256((OUT / 'launch.json').read_bytes()).hexdigest() == '2845eb79ddbedd8fb98f5dc17a84d1359b6979cb850691ae3aeae8725cd213c2'
    assert result['preregistration_commit'] == launch['preregistration_commit'] == '159732dd2ce2eac03ebd443ffa8bb9ef98035f63'
    assert launch['fresh_source_no_existing_bytecode_cache'] is True
    assert launch['owned_launch_attempt'] == 2
    for name, digest in launch['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)).hexdigest() == digest
    assert result['private_output'] == {'bytes': 2146, 'sha256': '66347f5fafedbb3cb6f6afe9f152ac0c35865687c45aea944879df5dd209c11f'}


def test_actual_proof_refutes_balanced_family_not_unbalanced_or_whole_dataset():
    result = json.loads((OUT / 'result.json').read_text())
    structural = result['structural_result']
    proof = structural['integer_necessary_conditions']
    assert proof['class_pool'] == sum(proof['forced_component_sizes_sorted']) == 23
    assert proof['forced_equality_edges'] == 15
    assert proof['forced_component_sizes_sorted'] == [16, 1, 1, 1, 1, 1, 1, 1]
    assert proof['balanced_group_max_classes'] == 6
    assert proof['balanced_family_integer_capacity_contradiction'] is True
    assert proof['unbalanced_final_capacity_ceiling_with_dev1_policy2_fit2'] == 18
    assert proof['nonempty_four_role_unbalanced_capacity_contradiction'] is False
    assert structural['solver_executed'] is False
    assert structural['feasible_joint_assignment_verified'] is False
    assert structural['new_protocol_frozen'] is False
    assert structural['complete_visual_category_absence_proven'] is False
    assert result['census_reconciliation'] == {'tracks': 1962, 'observations': 38680, 'videos': 429, 'positive_Known_classes': 48}


def test_actual_failed_gate_and_one_owned_successful_worker_no_process_replay():
    first = OUT.with_name(OUT.name + '_supervisor_attempt_1.json')
    second = OUT.with_name(OUT.name + '_supervisor_attempt_2.json')
    assert hashlib.sha256(first.read_bytes()).hexdigest() == 'f940381de7df45ddf5b9f9c2b4ca902e965ec24cda2c40670da689db949b80d8'
    assert hashlib.sha256(second.read_bytes()).hexdigest() == '93de094f1b72ff9f9dd73401468c46d1dba4f438dab9dd348254332517449d81'
    one, two = json.loads(first.read_text()), json.loads(second.read_text())
    assert one['owned_worker_exit_code'] == 1 and one['successful_owned_execution'] is False
    assert two['owned_worker_exit_code'] == 0 and two['successful_owned_execution'] is True
    for receipt in (one, two):
        assert receipt['only_new_owned_Popen_worker_managed'] is True
        assert receipt['guard_stop_reason'] is None
        assert receipt['peak_sampled_owned_worker_RSS_bytes'] < 268435456
        assert receipt['peak_sampled_worker_threads'] == 1
        assert receipt['wall_seconds_including_imports_network_and_solver'] < 120
        assert receipt['training_GPU_hours'] == 0
    result = json.loads((OUT / 'result.json').read_text())
    assert result['original_annotation_old_target_tables_names_Novel_Val_Test_pixels_features_models_read'] is False
    assert result['optimization_training_steps'] == result['resources']['training_GPU_hours'] == 0
