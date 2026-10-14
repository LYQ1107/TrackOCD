from pathlib import Path

from src.trackocd_cvpr.joint_rc_policy_receipts import verify_joint_rc_policy_receipt


ROOT = Path(__file__).resolve().parents[2]


def test_actual_joint_rc_policy_pilot_public_receipt_is_bounded_and_not_goal_acceptance():
    result = verify_joint_rc_policy_receipt(ROOT)
    assert result['optimizer_steps'] == 2
    assert result['videos'] == 2
    assert result['metadata_frames'] == 79
    assert result['physical_observations'] == 215
