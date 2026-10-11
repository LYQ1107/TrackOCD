"""Measured necessary-only feasibility regressions, not protocol certification."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/train_fold_feasibility_v1'


def test_actual_fixed_role_feasibility_exact_registration_and_all_folds_preserved():
    path = OUT / 'result.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ca642ab8504a00fb6682e348bbec74bfc1ba4bf176f7ce3a1af2946e12cb3163'
    result = json.loads(path.read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert result['preregistration_commit'] == launch['preregistration_commit'] == 'ca05ef02223846e9edfa02f7b9de0d3f988830bd'
    for name, digest in launch['source_sha256'].items():
        content = subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == result['source_sha256'][name] == digest
    assert [row['necessary_support_deficits'] for row in result['summary']['folds']] == [0, 9, 9, 0]
    assert result['summary']['class_or_seed_selected_from_metric_or_support_results'] is False
    assert result['metadata_scope']['old_final_heldout_labels_decoded'] is False
    assert result['model_or_new_annotation_or_feature_execution'] is result['Val_or_Test_access'] is False
    assert result['resources']['training_GPU_hours'] == result['optimization_steps'] == 0


def test_two_necessary_pass_folds_not_misrepresented_as_lawful_global_splits():
    summary = json.loads((OUT / 'result.json').read_text())['summary']
    assert summary['new_training_protocol_frozen'] is False
    assert summary['all_source_categories_final_pseudo_once'] is True
    for fold in summary['folds']:
        assert fold['global_video_disjoint_assignment_proven'] is False
        assert fold['complete_category_absence_in_video_proven'] is False
    assert sum(fold['necessary_support_only_PASS'] for fold in summary['folds']) == 2
