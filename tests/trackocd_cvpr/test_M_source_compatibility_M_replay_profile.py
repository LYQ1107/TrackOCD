from pathlib import Path

from src.trackocd_cvpr.M_source_compatibility_M_replay_profile import (
    verify_source_compatibility_M_replay,
)

ROOT = Path(__file__).resolve().parents[2]


def test_full_M_source_compatibility_replay_is_complete_but_not_qualified():
    result = verify_source_compatibility_M_replay(ROOT)
    assert result['actual_full_M_source_compatibility_replay_complete']
    assert result['current_source_training_provenance_established'] is False
    assert result['M_qualification_or_goal_acceptance'] is False
    assert result['launch_scope_is_inherited_original_worker_label']
