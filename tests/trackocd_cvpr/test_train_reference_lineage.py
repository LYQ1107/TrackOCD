"""Preserved source-file measurements are not runtime/census certificates."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/train_reference_lineage_v1'


def test_original_hash_measurement_preserved_with_explicit_wording_erratum():
    path = OUT / 'result.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == 'c69d675df34a6e23d4ec17389abbf5fbac91b0c46dba67bce9e5410e4cfcffca'
    result = json.loads(path.read_text())
    erratum = json.loads((OUT / 'interpretation_erratum.json').read_text())
    assert erratum['original_result_sha256'] == digest
    assert erratum['original_measured_source_hash_checks_invalidated'] is False
    assert erratum['original_result_modified_or_deleted'] is False
    assert 'all eight' in erratum['correction']
    assert sum(row['actual_original_plan_source_hash_recorded'] for row in result['source_checks']) == 6
    assert result['private_plan_rows_or_summary_JSON_decoded'] is result['role_ID_arrays_JSON_decoded'] is False
    assert result['annotation_file_opened_in_this_audit'] is result['new_or_protected_GT_labels_decoded'] is False
    assert result['fresh_protocol_or_complete_category_absence_certified'] is False


def test_original_registration_archived_byte_binding_not_current_source_assumption():
    launch = json.loads((OUT / 'launch.json').read_text())
    assert launch['preregistration_commit'] == '4ca9d42fea942ff0dfbbfb000bb68fde8d1df887'
    for name, digest in launch['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest


def test_corrected_real_rerun_reports_all_loaded_runtime_identities_unproven():
    path = OUT.parent / 'train_reference_lineage_v2/result.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '69a11cfd565de6738c9d70cc8b54c1dd3836dd373f40b4755aec2b002689e50f'
    result = json.loads(path.read_text())
    assert result['preregistration_commit'] == 'd2af08b40eca03bc29e90aedea1a69b613bc7dfa'
    assert sum(row['original_plan_hash_matches_registered_git_bytes'] for row in result['source_checks']) == 6
    assert all(row['historical_loaded_runtime_module_identity_proven'] is False for row in result['source_checks'])
    assert all('original_runtime_byte_identity_proven_from_recorded_plan' not in row for row in result['source_checks'])
    assert result['private_plan_rows_or_summary_JSON_decoded'] is result['role_ID_arrays_JSON_decoded'] is False
    assert result['fresh_protocol_or_complete_category_absence_certified'] is False
    assert result['optimization_steps'] == result['resources']['training_GPU_hours'] == 0
