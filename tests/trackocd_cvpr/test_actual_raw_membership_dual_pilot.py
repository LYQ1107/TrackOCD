import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.raw_membership_pilot_qualification import verify_actual_pilot,HASHES,NS

ROOT = Path(__file__).resolve().parents[2]


def test_actual_normal_dual_pilot_exposure_solver_checkpoint_and_source_not_quality_acceptance():
    p = verify_actual_pilot(ROOT)
    assert p['actual_normal_dual_pilot_prerequisites_checked']
    assert not p['private_data_or_weights_read'] and not p['calibration_quality_or_deployment_posterior_claimed']
    r = json.loads((ROOT/(NS+'/result.json')).read_text())
    d = r['resubstitution_diagnostics_on_same_training_samples']
    assert d['full_policy']['class_macro_NLL'] < d['initial']['class_macro_NLL']
    # Calendar NLL/Brier improve while weighted ECE worsens: no blanket quality claim.
    assert d['calendar_matched']['class_macro_NLL'] < d['initial']['class_macro_NLL']
    assert d['calendar_matched']['fixed_10_bin_class_macro_ECE'] > d['initial']['fixed_10_bin_class_macro_ECE']


def test_original_public_receipt_mutation_cannot_fake_solver_success_or_permission(tmp_path):
    for path in HASHES:
        p = tmp_path/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/path).read_bytes())
    p = tmp_path/(NS+'/result.json'); r = json.loads(p.read_text())
    r['full_policy_control']['solver_success'] = False; p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Exact original'):
        verify_actual_pilot(tmp_path)
