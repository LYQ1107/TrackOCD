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
