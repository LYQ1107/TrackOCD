import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.trackocd_cvpr.joint_known_protocol import forced_group_certificate, verify_assignment, joint_feasibility
from src.trackocd_cvpr.known_census_protocol import ROLES


def union_fixture(extra_outside=False):
    support = {0: {0, 1, 2, 3}, 1: {0, 1, 2, 4}, 2: {3, 4, 5, 6} if extra_outside else {3, 4, 5}}
    support.update({c: {100 + c * 3 + k for k in range(3)} for c in range(3, 12)})
    presence = {}
    for c, videos in support.items():
        for video in videos:
            presence.setdefault(video, set()).add(c)
    return presence


def test_union_closure_forces_equality_that_pairwise_graph_misses():
    initial, _ = forced_group_certificate(union_fixture())
    closure, private = forced_group_certificate(union_fixture(), union_support=True)
    assert initial['forced_component_sizes_sorted'][0] == 2
    assert closure['forced_component_sizes_sorted'][0] == 3
    assert closure['union_support_forced_merges'] == 1
    assert private['union_forced_group_witnesses'] == [{'category_X': 2, 'already_forced_group': [0, 1],
                                                      'X_videos_without_group_union': 1}]


def test_exact_two_outside_union_not_falsely_merged():
    public, private = forced_group_certificate(union_fixture(True), union_support=True)
    assert public['forced_component_sizes_sorted'][0] == 2
    assert public['union_support_forced_merges'] == 0
    assert private['union_forced_group_witnesses'] == []


def assignment(groups):
    folds = []
    for f in range(4):
        final, dev = groups[f], groups[(f + 1) % 4]
        remaining = [c for c in range(8) if c not in final + dev]
        classes = dict(zip(ROLES, (remaining[:-2], remaining[-2:], dev, final)))
        folds.append({'classes': classes, 'video_ids': {r: [c * 4 + k for c in cs for k in range(3 if r == 'fit_known' else 2)]
                                                       for r, cs in classes.items()}})
    return {'groups': groups, 'folds': folds}


def test_explicit_unbalanced_family_retains_every_class_and_video_isolation():
    presence = {c * 4 + k: {c} for c in range(8) for k in range(4)}
    witness = assignment([[0, 1, 2], [3], [4, 5, 6], [7]])
    assert verify_assignment(presence, witness, balanced=False)
    with pytest.raises(ValueError):
        verify_assignment(presence, witness)
    corrupted = copy.deepcopy(witness)
    corrupted['folds'][0]['video_ids']['development_pseudo'].append(corrupted['folds'][0]['video_ids']['fit_known'][0])
    with pytest.raises(ValueError):
        verify_assignment(presence, corrupted, balanced=False)


def test_unbalanced_integer_contradiction_does_not_launch_redundant_solver():
    def forbidden(**kwargs):
        raise AssertionError('No solver after exact necessary contradiction')
    public, _ = joint_feasibility({v: set(range(8)) for v in range(4)}, balanced=False, union_support=True, solve=forbidden)
    assert public['solver_executed'] is False
    assert public['integer_necessary_conditions']['nonempty_four_role_unbalanced_capacity_contradiction'] is True


def test_unbalanced_noncontradiction_requires_solver_not_assumed_feasible():
    presence = {c * 4 + k: {c} for c in range(8) for k in range(4)}
    calls = []
    def timeout(**kwargs):
        calls.append(True)
        return SimpleNamespace(status=1, x=None, message='Synthetic timeout')
    public, _ = joint_feasibility(presence, balanced=False, union_support=True, solve=timeout)
    assert calls == [True] and public['status'] == 'INCONCLUSIVE_NO_VERIFIED_ASSIGNMENT'


def test_registration_changes_balance_only_not_safety_or_category_pool():
    root = Path(__file__).resolve().parents[2]
    a = json.loads((root / 'configs/trackocd_cvpr/joint_known_protocol_v1.json').read_text())
    b = json.loads((root / 'configs/trackocd_cvpr/joint_known_protocol_v2_unbalanced.json').read_text())
    assert b['four_balanced_final_groups'] is False and b['union_support_closure'] is True
    for field in ('private_Known_index_sha256', 'private_exclusion_index_sha256', 'roles_sha256', 'expected',
                  'expected_eligible_pool', 'eligible_threshold', 'every_eligible_class_final_once',
                  'development_is_next_cyclic_group', 'policy_pseudo_classes_per_fold', 'minimum_fit_classes_per_fold',
                  'fit_videos_per_class', 'eval_videos_per_class', 'positive_conflict_rules_unchanged', 'training_GPU_hours'):
        assert b[field] == a[field]
