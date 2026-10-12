from copy import deepcopy
import pytest
from src.trackocd_cvpr.joint_known_protocol import global_joint_feasibility, verify_global_assignment


def presence():
    return {3 * c + i: {c} for c in range(16) for i in range(3)}


def test_actual_global_solver_expansion_original_video_verifier_and_breadth():
    p = presence()
    public, private = global_joint_feasibility(p)
    assert public['status'] == 'VERIFIED_GLOBAL_KNOWN_ASSIGNMENT_NOT_FROZEN_PROTOCOL'
    assignment = private['assignment']
    assert verify_global_assignment(p, assignment)
    assert public['role_class_counts']['independent_final_pseudo'] >= 8
    assert sum(public['role_class_counts'].values()) == 16
    assert not public['new_protocol_frozen'] and public['training_GPU_hours'] == 0
    for kind in ('video_overlap', 'missing_category', 'float_id', 'low_support'):
        bad = deepcopy(assignment)
        if kind == 'video_overlap':
            bad['video_ids']['fit_known'].append(bad['video_ids']['independent_final_pseudo'][0])
        elif kind == 'missing_category':
            bad['classes']['independent_final_pseudo'].pop()
        elif kind == 'float_id':
            bad['classes']['fit_known'][0] = float(bad['classes']['fit_known'][0])
        else:
            bad['video_ids']['fit_known'] = []
        with pytest.raises(ValueError):
            verify_global_assignment(p, bad)


def test_global_scoped_infeasibility_and_no_fake_protocol_freeze():
    public, private = global_joint_feasibility({i: set(range(16)) for i in range(3)})
    assert public['solver_status'] == 2 and public['status'] == 'SOLVER_SCOPED_INFEASIBLE'
    assert not public['feasible_joint_assignment_verified'] and 'assignment' not in private


def test_fixed_budget_refuses_unregistered_parameter_search():
    with pytest.raises(ValueError):
        global_joint_feasibility(presence(), time_limit=31)
