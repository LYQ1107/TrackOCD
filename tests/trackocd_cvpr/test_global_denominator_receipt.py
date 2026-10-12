"""Complete retained-Known counts; explicitly not all-object ground truth."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v9_denominators'


def test_actual_retained_Known_denominators_and_full_GT_boundary():
    result = json.loads((OUT / 'result.json').read_text())
    public = result['structural_result']
    assert public['status'] == 'VERIFIED_RETAINED_KNOWN_DENOMINATORS_FULL_GT_STILL_UNPROVEN'
    roles = public['role_denominators']
    assert [roles[r]['all_retained_Known']['tracks'] for r in
        ('fit_known', 'policy_train_pseudo', 'development_pseudo', 'independent_final_pseudo')] == [1178, 44, 423, 234]
    final = roles['independent_final_pseudo']
    assert final['fit_Known']['tracks'] == 135 and final['own_role_pseudo']['tracks'] == 93
    assert final['outside_eligible_Known']['tracks'] == 6
    assert all(role['other_role_pseudo_conflict']['tracks'] == 0 for role in roles.values())
    assert public['unassigned_retained_Known'] == {'tracks': 83, 'observations': 1561}
    assert public['reconciled_Known_tracks'] == 1962 and public['reconciled_Known_observations'] == 38680
    assert not public['full_GT_denominator_proven'] and not public['new_protocol_frozen']
    assert not public['nonKnown_GT_for_assigned_videos_retained']
    assert public['Known_rows_are_not_all_physical_ground_truth']
    assert result['optimization_training_steps'] == 0
    assert result['live_remote_registration_proof']['sha'] == '2e3dd439599f58a62f9f738bb9e479248feb5a27'
    receipt = json.loads(OUT.with_name(OUT.name + '_supervisor_attempt_1.json').read_text())
    assert receipt['owned_worker_exit_code'] == 0 and receipt['guard_stop_reason'] is None
    assert receipt['peak_sampled_worker_threads'] == 1 and receipt['peak_sampled_owned_worker_RSS_bytes'] <= 268435456
