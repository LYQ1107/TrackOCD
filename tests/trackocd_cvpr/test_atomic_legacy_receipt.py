"""Published receipt checks, separate from executed interface mutation tests."""
import hashlib
import json
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr0'


def test_failed_fixture_receipt_is_preserved_not_relabelled_pass():
    path = OUT / 'atomic_legacy_smoke_v1/failure.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'cf38c01744436d54b9a604e59bb66d71a8b8faf9bb28c7b0a80d04607eb8f0a9'
    failed = json.loads(path.read_text())
    assert failed['complete'] is False and failed['exception_type'] == 'ValueError'
    assert not (path.parent / 'result.json').exists()


def test_manifest_failure_is_not_misreported_as_a_running_worker():
    manifest = json.loads((OUT / 'experiment_manifest.json').read_text())
    stages = {row['id']: row for row in manifest['stages']}
    failed = stages['atomic_legacy_smoke_v1']
    assert failed['state'] == 'FAILED_ACTUAL_PRESERVED_NO_COMPLETED_RECEIPT'
    assert failed['actual_failure_receipt']['complete'] is False
    assert stages['atomic_legacy_smoke_v2']['state'] == 'COMPLETE_SUBSTEP_NOT_GOAL_COMPLETE'
    assert manifest['status'] == 'IN_PROGRESS' and manifest['training_GPU_hours_used'] == 0


def test_tiny_result_matches_actual_preregistration_and_limits():
    result_path = OUT / 'atomic_legacy_smoke_v2/result.json'
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == '65fe010fc91eb7bbc79c27ce99893b6bcb5743a3e1e395f0d37ce19f104c3c89'
    result = json.loads(result_path.read_text())
    launch = json.loads((result_path.parent / 'launch.json').read_text())
    assert result['preregistration_commit'] == launch['preregistration_commit'] == '033081c62706bae99f6af4dde637d49db4c5307d'
    assert result['source_sha256'] == launch['source_sha256']
    for name, digest in launch['source_sha256'].items():
        original = subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == digest
    assert result['selected_cached_frames'] == 32 < result['total_cached_video_frames'] == 40
    assert result['selected_observations'] == 1101
    assert result['resources']['wall_seconds'] < 120
    assert result['resources']['peak_RSS_bytes'] < 2 * 1024 ** 3
    assert result['resources']['training_GPU_hours'] == result['optimization_steps'] == 0
    assert result['GT_or_category_labels_decoded'] is result['Val_or_Test_access'] is False
    assert result['new_feature_extraction'] is result['trained_checkpoint_parity'] is result['dense_frontend_acceptance'] is False
    assert result['historical_physical_ID_controller_invariance_remains_FAIL'] is True


@pytest.mark.parametrize('policy', ('always_new', 'existing_if_available_else_new', 'always_wait'))
def test_registered_fixed_policies_are_not_a_learned_method_result(policy):
    result = json.loads((OUT / 'atomic_legacy_smoke_v2/result.json').read_text())
    case = next(x for x in result['cases'] if x['fixed_diagnostic_policy'] == policy)
    assert case['renumber_and_reverse_equal'] is True
    assert case['future_negation_changed_later_not_prior'] is True
    assert case['future_append_retained_all_prior_outputs'] is True
    assert case['baseline']['exact_tie_groups'] == 0
    assert sum(case['baseline']['actions'].values()) == 1101
    assert 'synthetic policies' in result['interpretation']
