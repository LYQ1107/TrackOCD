"""Real admin evidence; no private video IDs or target GT decoded by tests."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/safety/train_partition_scope_v1'


def test_actual_admin_projection_receipt_and_source_binding():
    result = json.loads((OUT / 'result.json').read_text())
    launch = json.loads((OUT / 'launch.json').read_text())
    assert hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest() == 'eaa932dcacbf70167e77f5e4c6ca9132b49e4a29c1642f7c7bf942ac4df65408'
    assert hashlib.sha256((OUT / 'launch.json').read_bytes()).hexdigest() == '155c373754482187205ac637a1aea75f051289d98c5774f7b8cb71f74c08993f'
    assert result['preregistration_commit'] == launch['preregistration_commit'] == 'd4badaa591997bfc8d2be3f375feae36213c3a50'
    assert result['source_sha256'] == launch['source_sha256']
    for name, digest in launch['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git', 'show', f"{launch['preregistration_commit']}:{name}"], cwd=ROOT)).hexdigest() == digest


def test_actual_video_disjointness_is_not_GT_label_or_protocol_clearance():
    result = json.loads((OUT / 'result.json').read_text())
    scope = result['partition_scope']
    assert scope['selected_rows'] == 2166 and scope['selected_videos'] == 480
    assert {part: row['videos'] for part, row in scope['partitions'].items()} == {
        'prototype': 39, 'representation_fit': 280, 'development': 57, 'policy_train': 53, 'heldout_selection': 51}
    assert scope['old_selected_plan_global_video_disjointness_verified'] is True
    assert scope['category_absence_or_complete_annotation_census_proven'] is False
    assert scope['new_protocol_or_scientific_access_cleared'] is False
    assert result['row_value_fields_JSON_decoded'] == ['video_id', 'partition']
    assert result['category_role_observation_summary_values_JSON_decoded'] is False
    assert result['annotation_target_label_tables_roles_images_features_models_Val_Test_read'] is False
    assert result['scientific_access_clearance_granted'] is False
    assert result['optimization_steps'] == result['resources']['training_GPU_hours'] == 0
    assert result['private_exclusion_index']['sha256'] == 'ee0d0ea48688506bbba51579eaf5b6db1b901fff4d4ce478e77281a6725fda5d'


def test_resource_erratum_preserves_actual_measurements_and_original_result():
    result = json.loads((OUT / 'result.json').read_text())
    erratum = json.loads((OUT / 'resource_counter_erratum.json').read_text())
    assert erratum['original_result_not_overwritten'] is True
    assert erratum['original_result_sha256'] == hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest()
    assert erratum['guard_sample_peak_bytes_before_result'] == result['resources']['peak_sampled_smaps_rollup_Rss_bytes']
    assert erratum['reported_result_snapshot_sample_bytes'] == result['resources']['sampled_smaps_rollup_Rss_bytes']
    assert erratum['maximum_of_recorded_samples_bytes'] >= erratum['guard_sample_peak_bytes_before_result']
    assert erratum['scope_or_video_partition_result_affected'] is False


def test_scope_proposal_cannot_authorize_itself_or_silently_expand_video_set():
    proposal = json.loads((ROOT / 'configs/trackocd_cvpr/clean_train_gt_scope_proposal.json').read_text())
    assert proposal['status'] == 'PENDING_USER_DIRECTION_AFTER_METADATA_ACCESS_INCIDENT'
    assert proposal['new_GT_read_authorized_by_this_proposal'] is False
    assert proposal['training_or_cache_execution_authorized_by_this_proposal'] is False
    assert proposal['allowlisted_videos'] == 429 and proposal['excluded_old_heldout_videos'] == 51
    assert proposal['all_other_video_IDs_default_deny'] is True
    assert proposal['proposed_if_confirmed']['nested_video_GT_arrays_decoded'] is False
    assert proposal['proposed_if_confirmed']['full_visual_category_absence_certified_from_positive_annotations'] is False
