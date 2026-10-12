import copy
from types import SimpleNamespace
import numpy as np
import pytest
from src.trackocd_cvpr.joint_known_protocol import forced_group_certificate, build_joint, joint_feasibility, verify_assignment


def separate():
    return {c * 4 + k: {c} for c in range(8) for k in range(4)}


def test_necessary_equality_chain_has_integer_capacity_certificate():
    presence = separate()
    for video in (0, 1, 2, 3):
        presence[video].update({1, 2})
    for video in (4, 5, 6, 7):
        presence[video].add(2)
    public, private = forced_group_certificate(presence)
    assert public['forced_component_sizes_sorted'][0] == 3
    assert public['balanced_group_max_classes'] == 2
    assert public['balanced_family_integer_capacity_contradiction'] is True
    assert any(e['category_X'] == 0 and e['category_Y'] == 1 and e['X_videos_without_Y'] == 0 for e in private['directed_support_witnesses'])
    def forbidden_solver(**kwargs):
        raise AssertionError('Exact contradiction requires no solver')
    result, _ = joint_feasibility(presence, solve=forbidden_solver)
    assert result['solver_executed'] is False


def test_two_noncooccurring_videos_do_not_falsely_force_equality():
    presence = separate()
    presence[0].add(1)
    presence[1].add(1)
    _, private = forced_group_certificate(presence)
    assert not any(e['category_X'] == 0 and e['category_Y'] == 1 for e in private['directed_support_witnesses'])


def test_real_bounded_synthetic_solver_and_independent_verifier():
    presence = separate()
    public, private = joint_feasibility(presence, time_limit=5, node_limit=100)
    assert public['solver_executed'] is True
    assert public['feasible_joint_assignment_verified'] is True
    assert verify_assignment(presence, private['assignment'])
    assert public['complete_visual_category_absence_proven'] is False
    assert public['new_protocol_frozen'] is False


def fixture_assignment():
    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    folds = []
    for f in range(4):
        final, dev = groups[f], groups[(f + 1) % 4]
        remainder = [c for c in range(8) if c not in final + dev]
        classes = dict(zip(('fit_known', 'policy_train_pseudo', 'development_pseudo', 'independent_final_pseudo'),
                           (remainder[:2], remainder[2:], dev, final)))
        folds.append({'classes': classes, 'video_ids': {role: [c * 4 + k for c in cs for k in range(3 if role == 'fit_known' else 2)]
                                                       for role, cs in classes.items()}})
    return {'groups': groups, 'folds': folds}


@pytest.mark.parametrize('mutation', ['duplicate_class', 'bad_cycle', 'video_overlap', 'quota', 'conflicting_positive'])
def test_actual_independent_certificate_mutations_refused(mutation):
    presence, assignment = separate(), fixture_assignment()
    assert verify_assignment(presence, assignment)
    changed = copy.deepcopy(assignment)
    if mutation == 'duplicate_class':
        changed['groups'][0][0] = 2
    elif mutation == 'bad_cycle':
        changed['folds'][0]['classes']['development_pseudo'] = [4, 5]
    elif mutation == 'video_overlap':
        changed['folds'][0]['video_ids']['policy_train_pseudo'].append(changed['folds'][0]['video_ids']['fit_known'][0])
    elif mutation == 'quota':
        changed['folds'][0]['video_ids']['fit_known'].pop()
    else:
        video = changed['folds'][0]['video_ids']['fit_known'][0]
        presence[video].add(changed['folds'][0]['classes']['independent_final_pseudo'][0])
    with pytest.raises(ValueError):
        verify_assignment(presence, changed)


def test_timeout_is_not_infeasibility_or_a_frozen_protocol():
    def timeout(**kwargs):
        assert kwargs['options']['threads'] == 1 and kwargs['options']['random_seed'] == 2093001
        assert not kwargs['c'].any()
        return SimpleNamespace(status=1, message='Synthetic time limit', x=None)
    public, _ = joint_feasibility(separate(), solve=timeout)
    assert public['status'] == 'INCONCLUSIVE_NO_VERIFIED_ASSIGNMENT'
    assert public['solver_reported_scoped_infeasibility_not_formal_global_data_proof'] is False


def test_invalid_solver_vector_not_rounded_into_success():
    def fractional(**kwargs):
        return SimpleNamespace(status=0, message='Synthetic', x=np.full(len(kwargs['c']), 0.5))
    with pytest.raises(ValueError, match='binary'):
        joint_feasibility(separate(), solve=fractional)


def test_inconsistent_backend_status_not_declared_verified_success():
    def contradictory(**kwargs):
        return SimpleNamespace(status=2, message='Synthetic contradictory backend', x=np.zeros(len(kwargs['c'])))
    with pytest.raises(ValueError, match='Unexpected solver status'):
        joint_feasibility(separate(), solve=contradictory)


def test_sparse_problem_is_bounded_and_input_order_invariant():
    first = build_joint(separate())
    second = build_joint(dict(reversed(list(separate().items()))))
    assert first[0].shape == second[0].shape
    assert (first[0] != second[0]).nnz == 0
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[2], second[2])
    with pytest.raises(ValueError):
        joint_feasibility(separate(), time_limit=31)


def test_ignoring_class_names_is_not_silently_permitting_bool_category_IDs():
    assignment = fixture_assignment()
    assignment['groups'][0][1] = True
    with pytest.raises(ValueError, match='integer'):
        verify_assignment(separate(), assignment)


def test_unbalanced_capacity_certificate_is_stronger_only_when_component_large():
    presence = {v: set(range(8)) for v in range(4)}
    public, _ = forced_group_certificate(presence)
    assert public['forced_component_sizes_sorted'] == [8]
    assert public['nonempty_four_role_unbalanced_capacity_contradiction'] is True
    assert public['unbalanced_final_capacity_ceiling_with_dev1_policy2_fit2'] == 3


def test_supervisor_resource_function_does_not_manage_foreign_processes(tmp_path, monkeypatch):
    from scripts.trackocd_cvpr.audit_joint_known_protocol import resources_ok
    from pathlib import Path
    import time
    monkeypatch.setattr(Path, 'read_text', lambda self: 'MemTotal: 1000 kB\nMemAvailable: 500 kB\n')
    cfg = {'rss_bytes_ceiling': 1024, 'wall_seconds_ceiling': 120}
    safety = {'system_ram_reserve_fraction': 0.25, 'disk_free_reserve_bytes': 0}
    assert resources_ok(cfg, safety, tmp_path, time.monotonic(), 100) is None
    assert resources_ok(cfg, safety, tmp_path, time.monotonic(), 2048) == 'owned_worker_RSS'
    assert resources_ok(cfg, safety, tmp_path, time.monotonic() - 121, 100) == 'owned_job_wall'
