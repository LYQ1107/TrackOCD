"""Public audit metadata checks, not execution or scientific acceptance tests."""
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def public_audit(name):
    return json.loads((ROOT / 'outputs/trackocd_cvpr/cvpr1' / name).read_text())


def test_masa_review_is_pinned_but_not_claimed_replication():
    review = public_audit('masa_source_audit.json')
    assert review['repository_commit'] == 'c5472b9c7615f35abdf1188cb1a0c5408fe50d66'
    assert review['official_repository_verified']
    assert review['whole_project_license_read']
    assert review['actual_model_loss_training_evaluation_source_read']
    assert not review['additional_to_user_supplied_ten']
    assert not review['source_execution_or_training']
    assert not review['TrackOCD_new_method_or_replication_claimed']
    assert not review['twelve_work_two_extra_acceptance_complete']
    sources = review['actual_files_read']
    assert len(sources) == 17
    assert len({entry['path'] for entry in sources}) == len(sources)
    assert all(re.fullmatch('[0-9a-f]{40}', entry['git_blob_sha1']) for entry in sources)
    assert all(entry['entire_file_read'] for entry in sources)
    dependencies = review['dependency_source_scope']
    assert len(dependencies) == 3
    assert all(re.fullmatch('[0-9a-f]{64}', entry['sha256']) for entry in dependencies)


def test_review_keeps_instance_and_retrospective_output_boundaries():
    review = public_audit('masa_source_audit.json')
    assert 'physical-instance' in review['original_task']
    assert 'cross-video semantic anonymous category' in review['model_outputs']
    assert 'majority vote' in review['online_or_offline']
    assert 'not actual GT/ID/future mutation proof' in review['causal_or_noncausal']
    qualifications = '\n'.join(review['source_config_qualifications'])
    assert 'four-column boxes' in qualifications
    assert 'not a foreground calibration certificate' in qualifications
    assert 'with_cats' in qualifications
    assert 'before update/TTL eviction' in qualifications
    assert 'Actual ID/future/GT mutation tests and downstream atomic category policy/memory' in review['requires_before_qualified_import']


def test_conditional_storage_permission_does_not_claim_cleanup():
    review = json.loads((ROOT / 'outputs/trackocd_cvpr/storage_review_20261012.json').read_text())
    latest = review['followup_20261011T2127Z']
    assert latest['filesystem_available_bytes'] > review['minimum_free_bytes']
    assert latest['deleted_paths'] == []
    assert latest['reclaimed_bytes'] == 0
    assert not latest['processes_intervened']
    assert not latest['large_launch_allowed']


def test_ovtr_static_source_scope_is_not_legal_frontend_acceptance():
    review = public_audit('ovtr_source_audit.json')
    assert review['repository_commit'] == '500e72c19bf5f7f8717546911a5639fdc26bfee5'
    assert review['official_repository_verified']
    assert review['actual_model_loss_training_evaluation_source_read']
    assert not review['source_execution_or_training']
    assert not review['TrackOCD_new_method_or_replication_claimed']
    assert not review['twelve_work_two_extra_acceptance_complete']
    assert len(review['actual_files_read']) == 16
    assert sum(entry['entire_file_read'] for entry in review['actual_files_read']) == 12
    for entry in review['actual_files_read']:
        assert re.fullmatch('[0-9a-f]{40}', entry['git_blob_sha1'])
        if not entry['entire_file_read']:
            assert entry['line_ranges_read']
    assert 'not referenced' in review['causal_or_noncausal']
    assert 'not an actual GT/future/ID mutation certificate' in review['causal_or_noncausal']
    assert 'mutation checks' in review['requires_before_qualified_import'][2]


def test_exact_archive_cleanup_retains_complete_input_receipts():
    review = json.loads((ROOT / 'outputs/trackocd_cvpr/coco_download_cleanup_20261012.json').read_text())
    targets = review['verification']['targets']
    assert [target['verified_images'] for target in targets] == [118287, 5000]
    assert review['deleted_paths'] == [target['archive_path'] for target in targets]
    assert sum(target['archive_bytes'] for target in targets) == review['removed_payload_bytes']
    assert all(isinstance(target['archive_signature'][3], str) for target in targets)
    assert review['reclaimed_bytes'] == review['filesystem_available_after_bytes'] - review['filesystem_available_before_bytes']
    assert not review['annotations_read']
    assert not review['external_processes_intervened']
    assert len(review['retained']) == 3


def test_vb_cgcd_gt_alignment_and_empty_guide_are_not_hidden():
    review = public_audit('vb_cgcd_source_audit.json')
    assert review['repository_commit'] == '7fe9501b425b5775588d0a06d60c817ee1f04663'
    assert review['official_repository_verified']
    assert review['actual_model_loss_training_evaluation_source_read']
    assert not review['whole_project_license_read']
    assert 'GT' in review['model_inputs']
    assert 'empty guide' in review['losses']
    assert 'whole-session' in review['online_or_offline']
    assert len(review['actual_files_read']) == 9
    for entry in review['actual_files_read']:
        assert entry['entire_file_read']
        assert re.fullmatch('[0-9a-f]{40}', entry['git_blob_sha1'])
    assert not review['source_execution_or_training']
    assert not review['twelve_work_two_extra_acceptance_complete']


def test_hypcd_license_does_not_certify_causal_category_memory():
    review = public_audit('hypcd_source_audit.json')
    assert review['repository_commit'] == 'ffc895b7ff38be8502bb7e3bd851f1d2a55dff73'
    assert review['official_repository_verified']
    assert review['whole_project_license_read']
    assert review['additional_to_user_supplied_ten']
    assert review['actual_model_loss_training_evaluation_source_read']
    assert len(review['actual_files_read']) == 14
    assert all(entry['entire_file_read'] for entry in review['actual_files_read'])
    assert 'Hungarian' in review['evaluation']
    assert 'no dynamic birth' in review['state_memory']
    assert not review['source_execution_or_training']
    assert not review['TrackOCD_new_method_or_replication_claimed']
    assert not review['twelve_work_two_extra_acceptance_complete']


def test_paper_followups_preserve_static_and_historical_boundaries():
    talon = public_audit('talon_source_audit.json')
    ltc = public_audit('ltc_source_audit.json')
    for review in (talon, ltc):
        assert review['historical_initial_status'] == 'PARTIAL_SOURCE_REVIEW_NOT_CVPR1_COMPLETE'
        assert review['venue_independently_verified']
        assert review['actual_model_loss_training_evaluation_source_read']
        assert not review['whole_project_license_read']
        assert not review['source_execution_or_training']
        assert not review['twelve_work_two_extra_acceptance_complete']
    assert 'sign' in '\n'.join(talon['paper_source_followup'])
    assert 'CVPR2026F' in ltc['accepted_paper_url']


def test_coverage_counts_scoped_cores_not_availability_or_replication():
    index = public_audit('review_coverage_index.json')
    rows = index['entries']
    assert len({row['audit_file'] for row in rows}) == len(rows)
    counted = [row for row in rows if row['actual_core_scope_review_countable']]
    assert len(counted) == index['scoped_actual_core_reviews'] == 12
    assert sum(row['licensed_extra_for_inspection_substep'] for row in rows) == 3
    for row in rows:
        review = json.loads((ROOT / row['audit_file']).read_text())
        assert review['repository_commit'] == row['repository_commit']
        assert len(review['actual_files_read']) == row['actual_path_scopes_read']
        assert not row['deployment_or_scientific_replication_certified']
    excluded = {Path(row['audit_file']).name for row in rows if not row['actual_core_scope_review_countable']}
    assert excluded == {'assignment_hash_source_audit.json', 'vc_cgcd_source_audit.json',
                        'tract_source_availability_audit.json', 'dydm_paper_source_availability_audit.json'}
    assert index['additional_availability_only'][0]['tree_files'] == ['README.md']
    assert not index['additional_availability_only'][0]['actual_core_scope_review_countable']
    assert not index['cvpr1_stage_acceptance_complete']
    assert not index['scientific_goal_complete']
