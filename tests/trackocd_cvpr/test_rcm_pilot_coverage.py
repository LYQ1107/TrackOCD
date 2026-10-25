from pathlib import Path

from scripts.trackocd_cvpr.audit_rcm_pilot_coverage import build_audit


def test_rcm_pilot_does_not_claim_memory_supervision_without_cross_video_support():
    audit = build_audit(Path(__file__).resolve().parents[2])

    assert audit["status"] == "FAIL_RCM_PILOT_M_SUPPORT_COVERAGE_ZERO"
    assert audit["coverage"]["optimizer_steps"] == 2
    assert audit["coverage"]["m_loss_steps"] == 0
    assert audit["coverage"]["positive_other_video_groups"] == 0
    assert audit["boundary"]["m_branch_scientifically_exercised"] is False
    assert audit["boundary"]["goal_a_b_c_accepted"] is False
