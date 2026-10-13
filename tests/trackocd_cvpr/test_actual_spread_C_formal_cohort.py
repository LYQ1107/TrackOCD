import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.spread_C_complete_fit_cohort import verified_cohort, cohort_sources

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all6_fresh_paired_fixed200_public_proof_not_dev_or_calibration_gain():
    proof = verified_cohort(ROOT)
    assert proof['all6_actual_fresh_fixed200_fits'] and len(proof['trials']) == 6
    assert not proof['private_endpoint_hashes_checked']
    assert not proof['development_calibration_Pred_joint_or_Goal_accepted']
    assert len(cohort_sources()) == 9
    assert {r['seed'] for r in proof['trials']} == {1027, 1028, 1029}
    assert all(r['actual_extra_gradient_steps'] == 0 for r in proof['trials'] if r['arm'] == 'baseline')


def test_missing_or_partial_seed_receipts_do_not_qualify_full6(tmp_path):
    with pytest.raises(ValueError, match='All three normal complete'):
        verified_cohort(tmp_path)


@pytest.mark.parametrize('field', ['short_budget', 'warmstart', 'failed_terminal'])
def test_public_receipt_mutations_fail_before_any_private_endpoint_or_dev_input(tmp_path, field):
    for path in cohort_sources():
        p = tmp_path/path; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT/path).read_bytes())
    path = next(p for p in cohort_sources() if p.endswith('/result.json'))
    if field == 'failed_terminal':
        path = next(p for p in cohort_sources() if p.endswith('_supervisor_attempt_1.json'))
    p = tmp_path/path; r = json.loads(p.read_text())
    if field == 'short_budget':
        r['summary']['optimizer_steps_each'] = 199
    elif field == 'warmstart':
        r['pilot_or_historical_checkpoint_warmstart'] = True
    else:
        r['successful_owned_execution'] = False
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='Actual fresh complete'):
        verified_cohort(tmp_path, check_private_endpoints=True)
