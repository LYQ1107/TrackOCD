"""Actual containment receipt, not fresh annotation or scientific execution."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/safety/metadata_containment_audit_v2'


def test_actual_containment_receipt_bound_to_registered_source_and_incident():
    result = json.loads((OUT / 'result.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '55dff1469bee381d512b9723fa463f9bece55d3ff90119cfa194d26ec418e67b'
    assert hashlib.sha256((OUT / 'launch.json').read_bytes()).hexdigest() == '519e8df2f368fb5bef83cec2c98a12e201b8381d96207febd4df0b9f7e0a6318'
    assert result['preregistration_commit'] == launch['preregistration_commit'] == '8222946d80dd161700850f692fa004d13abce9a9'
    assert result['source_sha256'] == launch['source_sha256']
    for name, digest in launch['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    assert result['incident_sha256'] == '4df93a3ec16b0ee7b46701fee112a77d19d95a0994ab7845528bcaae03050bf8'


def test_actual_plan_containment_does_not_clear_prior_access_or_historical_runtime():
    result = json.loads((OUT / 'result.json').read_text())
    assert result['sanitized_plan']['videos'] == 57
    assert result['sanitized_plan']['images'] == 2065
    assert result['sanitized_plan']['nested_GT_fields_in_sanitized_plan'] is False
    assert result['scientific_access_clearance_granted'] is False
    assert result['prior_GT_metadata_access_undone'] is result['all_historical_runtime_leakage_absent_proven'] is False
    assert result['annotation_target_labels_images_features_models_Val_Test_read'] is False
    assert result['optimization_steps'] == result['resources']['training_GPU_hours'] == 0
    assert len(result['recorded_physical_sources']) == 8
    assert all(row['recorded_source_file_binding_matches_git'] and
               not row['historical_loaded_runtime_module_identity_proven'] for row in result['recorded_physical_sources'])
    assert 0 < result['resources']['guarded_process_RSS_bytes'] < 268435456
    assert result['resources']['resource_ru_maxrss_bytes_including_preexec_history'] == 847216640


def test_preread_network_and_resource_failures_not_erased_by_v2_pass():
    for name in ('metadata_containment_network_gate_failure_v1.json', 'metadata_containment_resource_gate_failure_v1.json'):
        failed = json.loads((ROOT / 'outputs/trackocd_cvpr/safety' / name).read_text())
        assert failed['registered_commit'] == '9812b5c229eb536c14ac00c9a25e063885b84400'
        assert failed['status'].startswith('FAILED_')
        assert failed['exit_code'] == 1
        assert failed['launch_or_result_created_by_failed_attempt'] is False
        assert failed['scientific_execution_or_private_plan_read_in_failed_attempt'] is False
