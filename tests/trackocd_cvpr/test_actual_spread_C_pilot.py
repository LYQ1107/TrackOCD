from pathlib import Path
import json
import pytest
from src.trackocd_cvpr.spread_C_pilot_qualification import verify_informative_pilot,HASHES,NS

ROOT=Path(__file__).resolve().parents[2]


def test_actual_normal_informative_spread_pilot_not_discovery_or_formal_acceptance():
    proof=verify_informative_pilot(ROOT)
    assert proof['actual_informative_pilot_checked']
    assert not proof['private_weights_read_or_warmstart'] and not proof['formal_dev_calibration_or_Goal_accepted']
    r=json.loads((ROOT/(NS+'/result.json')).read_text()); s=r['summary']
    for arm in s['arms']:
        rows=s['loss_curves'][arm]
        assert sum(r['base_family_groups'].get('Known',0) for r in rows)==119
        assert sum(r['base_family_groups'].get('policy_pseudo',0) for r in rows)==15
        assert s['episodes'][arm][0]['loss_opportunities']['available_rejection_groups_on_updated_frames']==26
        assert s['episodes'][arm][0]['physical_actions']=={'KNOWN':1094,'NEW':174}
    # Same aggregate actions in both arms despite different trained weights.


def test_exact_informative_receipt_cannot_be_replaced_with_no_gradient_claim(tmp_path):
    for path in HASHES:
        p=tmp_path/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes((ROOT/path).read_bytes())
    p=tmp_path/(NS+'/result.json'); r=json.loads(p.read_text()); r['actual_nonzero_rejection_gradient']=False
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Exact one'): verify_informative_pilot(tmp_path)
