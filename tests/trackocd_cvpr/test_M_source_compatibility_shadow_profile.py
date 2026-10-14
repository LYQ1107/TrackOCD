from pathlib import Path

from src.trackocd_cvpr.M_source_compatibility_shadow_profile import verify_source_compatibility_shadow

ROOT = Path(__file__).resolve().parents[2]


def test_actual_all11_source_compatibility_shadow_is_bound_and_not_M_qualification():
    result = verify_source_compatibility_shadow(ROOT)
    assert result['actual_all11_current_source_deployment_trace_compatibility_checked']
    assert result['current_source_training_provenance_established'] is False
    assert result['M_qualification_or_goal_acceptance'] is False
