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
