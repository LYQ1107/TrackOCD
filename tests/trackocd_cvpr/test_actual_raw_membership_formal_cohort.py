import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.raw_membership_formal_cohort import verified_cohort,cohort_sources,TRIALS,directory

ROOT=Path(__file__).resolve().parents[2]


def test_actual_all4_formal_controls_no_pilot_warmstart_no_three_fullpolicy_seeds_or_dev_claim():
    r=verified_cohort(ROOT)
    assert r['all4_actual_fresh_formal_controls_verified'] and len(r['trials'])==4
    assert len(cohort_sources())==12
    assert r['actual_three_distinct_matched_C_calendars_verified']
    assert r['same_deterministic_initialization_and_scalar_table']
    assert not r['fullpolicy_is_three_training_seeds'] and not r['private_snapshot_bytes_checked']
    assert not r['transfer_calibration_control_winner_or_Goal_accepted']
    assert sum(x['actual_checkpoint_count'] for x in r['trials'])==13


def test_missing_full_cohort_refuses_before_any_private_snapshot_or_dev_label(tmp_path):
    with pytest.raises(ValueError,match='ALL4 actual normal'):
        verified_cohort(tmp_path,check_private_snapshots=True)


@pytest.mark.parametrize('change',['warmstart','budget','solver_failed'])
def test_partial_or_false_complete_formal_public_proofs_refuse_private_followup(tmp_path,change,monkeypatch):
    for path in cohort_sources():
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/path).read_bytes())
    # Calendar lineage is public prerequisite evidence, not a private model.
    # Include it so a late solver mutation reaches its own rejection gate.
    for seed in (1027,1028,1029):
        path=f'outputs/trackocd_cvpr/cvpr3/paired_spread_Known_rejection_Fixed200_v1_s{seed}/result.json'
        p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/path).read_bytes())
    selected=('full_policy',None) if change=='solver_failed' else TRIALS[0]
    p=tmp_path/(directory(*selected)+'/result.json');r=json.loads(p.read_text())
    if change=='warmstart':r['old_or_pilot_weight_warmstart']=True
    elif change=='budget':r['summary']['optimizer_steps']=199
    else:r['solver_converged']=False
    p.write_text(json.dumps(r))
    original_stat=Path.stat
    def forbid_private_stat(path,*args,**kwargs):
        if path.suffix=='.pt':raise AssertionError('Invalid public cohort touched private snapshot')
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',forbid_private_stat)
    # No training source/private endpoints are copied: failed public proof is
    # rejected before the source/file/followup input access can be reached.
    expected='bounded converged' if change=='solver_failed' else ('All200updates' if change=='budget' else 'bounded fresh control')
    with pytest.raises(ValueError,match=expected):verified_cohort(tmp_path,check_private_snapshots=True)
