"""Actual public protocol aggregates only, not private category/video IDs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/known_census_protocol_feasibility_v1'


def test_actual_fixed_audit_receipt_and_registered_sources():
    result = json.loads((OUT / 'result.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == '2d3873990058f7c1853f391609953620a30a56d5477725e3e817fcc5c1a78128'
    assert hashlib.sha256((OUT / 'launch.json').read_bytes()).hexdigest() == '59727344abf074ac399cd2f0d06fd5e7b46c2d995dbc297d2b600fa302435cea'
    assert result['preregistration_commit'] == launch['preregistration_commit'] == '5d8684f7cebd26f255c297f2c85efa0a6aad0428'
    assert result['source_sha256'] == launch['source_sha256']
    for name, digest in launch['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)).hexdigest() == digest
    assert result['private_output'] == {'bytes': 179723, 'sha256': '94521869748c9cf1c92a23fc99d693824cd5899860e05d7a66f15f945c81219b'}


def test_all_eight_actual_failures_preserved_not_called_global_impossibility():
    result = json.loads((OUT / 'result.json').read_text())
    assert result['census_reconciliation'] == {'tracks': 1962, 'observations': 38680, 'videos': 429, 'positive_Known_classes': 48}
    variants = result['hypotheses']
    assert [(h['eligibility_video_support'], h['class_pool']) for h in variants] == [(0, 48), (3, 23)]
    assert [[f['individual_necessary_support_deficits'] for f in h['folds']] for h in variants] == [[27, 37, 36, 26], [5, 15, 14, 3]]
    assert [[f['matched_slots'] for f in h['folds']] for h in variants] == [[73, 34, 39, 79], [48, 19, 20, 50]]
    for h in variants:
        assert h['all_folds_retained_no_successful_fold_or_metric_selection'] is True
        assert h['new_protocol_frozen'] is False
        assert len(h['folds']) == 4
        for f in h['folds']:
            assert f['globally_video_disjoint_sufficient_witness_found'] is False
            assert f['failure_proves_general_assignment_impossible'] is False
            assert f['complete_visual_category_absence_proven'] is False
            assert sum(row['classes'] for row in f['roles'].values()) == h['class_pool']
            deficits = sum(sum(n < row['minimum_distinct_videos_per_class'] for n in row['candidate_support_counts_sorted'])
                           for row in f['roles'].values())
            assert deficits == f['individual_necessary_support_deficits'] > 0


def test_actual_source_scope_and_resources_no_training_or_final_acceptance():
    result = json.loads((OUT / 'result.json').read_text())
    assert result['annotation_old_target_tables_names_Novel_Val_Test_pixels_features_models_read'] is False
    assert result['labels_in_forward_or_live_memory'] is False
    assert result['aggregate_support_previously_observed_not_blind_preregistration'] is True
    assert result['optimization_steps'] == result['resources']['training_GPU_hours'] == 0
    assert result['resources']['wall_seconds'] < 120
    assert result['resources']['guarded_process_RSS_bytes'] < 268435456
    assert result['resources']['peak_sampled_smaps_rollup_Rss_bytes_through_result_snapshot'] < 268435456
