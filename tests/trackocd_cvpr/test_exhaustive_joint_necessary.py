import pytest
from src.trackocd_cvpr.joint_known_protocol import exhaustive_component_necessary


def isolated():
    return {3 * c + i: {c} for c in range(8) for i in range(3)}


def test_complete_placements_positive_support_is_not_a_verified_video_witness():
    result, _ = exhaustive_component_necessary(isolated())
    assert result['enumerated_component_placements'] == result['expected_placements'] == 16384
    assert result['necessary_support_survivors'] == result['legal_class_partitions'] > 0
    assert result['status'] == 'INCONCLUSIVE_NECESSARY_SUPPORT_SURVIVORS'
    assert not result['feasible_joint_assignment_verified'] and not result['new_protocol_frozen']
    assert result['first_rejecting_fold_partition_counts'] == [0] * 4
    assert not result['solver_executed']


def test_complete_forced_component_capacity_negative_not_global_claim():
    result, _ = exhaustive_component_necessary({i: set(range(8)) for i in range(3)})
    assert result['enumerated_component_placements'] == 1
    assert result['legal_class_partitions'] == result['necessary_support_survivors'] == 0
    assert result['status'] == 'EXHAUSTIVE_NECESSARY_SUPPORT_CONTRADICTION_FOR_DECLARED_FAMILY'
    assert not result['complete_visual_category_absence_proven']


def test_administrative_renaming_preserves_exhaustive_numeric_summary():
    original = isolated()
    renamed = {1000 - v: {100 - c for c in cs} for v, cs in original.items()}
    a, _ = exhaustive_component_necessary(original)
    b, _ = exhaustive_component_necessary(renamed)
    assert a == b


def test_bounds_refuse_not_silently_drop_components():
    with pytest.raises(ValueError, match='bound8components'):
        exhaustive_component_necessary({3 * c + i: {c} for c in range(9) for i in range(3)})
