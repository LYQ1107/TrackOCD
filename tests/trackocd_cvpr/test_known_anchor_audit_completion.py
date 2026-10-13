from pathlib import Path
import json
import pytest
import scripts.trackocd_cvpr.qualify_known_anchor_audit_completion as driver

ROOT=Path(__file__).resolve().parents[2]


def test_actual_one_exit0_full_counter_exception_not_normal_supervisor():
    counts=driver.validate(ROOT)
    assert counts['Known_eligible_frames']==333 and counts['cooccurring_Known_pseudo_frames']==174
    assert counts['pseudo_groups_on_all_Known_anchor_frames']==176
    assert counts['selected_pseudo_groups']==0
    assert json.loads((ROOT/driver.TERMINAL).read_text())['successful_owned_execution'] is False


def test_exact_immutable_terminal_binding_cannot_be_rewritten_as_normal(tmp_path):
    for path in driver.HASHES:
        p=tmp_path/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/path).read_bytes())
    p=tmp_path/driver.TERMINAL; t=json.loads(p.read_text()); t['successful_owned_execution']=True
    p.write_text(json.dumps(t))
    with pytest.raises(ValueError,match='Exact one'): driver.validate(tmp_path)


def test_actual_separate_qualification_preserves_failed_original_and_no_new_fit():
    from src.trackocd_v2.io import sha256_file
    path=ROOT/driver.OUTPUT
    assert sha256_file(path)=='932d3dbd1d3d6b5cdc1f58aec0cb08e5fc6ba56a0939665237a94c7defa41db2'
    record=json.loads(path.read_text())
    assert record['effective_complete_worker_qualification']
    assert not record['original_supervisor_successful_owned_execution']
    assert not record['original_terminal_rewritten'] and not record['counter_failure_cause_proven']
    assert record['no_dataset_GT_weights_or_private_history_read']
    assert not record['new_training_model_or_calendar_selection']
    assert record['live_remote_registration_proof']['sha']=='f00cfbc3451c410550cddb636a734926f8552bb8'
    assert record['artifact_sha256']==driver.HASHES and record['scope_counts']==driver.validate(ROOT)
