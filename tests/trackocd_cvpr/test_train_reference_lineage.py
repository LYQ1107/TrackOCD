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
