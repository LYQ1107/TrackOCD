import copy
import pytest
from src.trackocd_cvpr.disjoint_video_witness import allocate_witness, verify_witness


def inputs():
    presence = {0: {10, 20}, 1: {10, 20}, 2: {10}, 3: {10}, 4: {20}, 5: {20}}
    roles = {'fit': {10}, 'heldout': {20}}
    candidates = {'fit': set(presence), 'heldout': {0, 1, 4, 5}}
    return presence, roles, candidates, {'fit': 3, 'heldout': 2}


def test_actual_disjoint_certificate_not_overlapping_candidate_counts():
    presence, roles, candidates, minima = inputs()
    public, witness = allocate_witness(presence, roles, candidates, minima)
    assert public['globally_video_disjoint_sufficient_witness_found'] is True
    assert public['matched_slots'] == public['required_category_video_slots'] == 5
    assert len(witness) == 5 and verify_witness(presence, roles, candidates, minima, witness)
    assert public['complete_visual_category_absence_proven'] is False
    assert public['new_protocol_frozen'] is False


def test_individually_sufficient_candidates_can_conflict_globally():
    presence = {0: {10, 20}, 1: {10, 20}, 2: {10, 20}}
    roles = {'fit': {10}, 'heldout': {20}}
    candidates = {'fit': set(presence), 'heldout': set(presence)}
    public, witness = allocate_witness(presence, roles, candidates, {'fit': 2, 'heldout': 2})
    assert public['individual_necessary_support_deficits'] == 0
    assert public['matched_slots'] == 3 and len(witness) == 3
    assert public['globally_video_disjoint_sufficient_witness_found'] is False
    assert public['failure_proves_general_assignment_impossible'] is False


def test_restrictive_slot_matching_fail_is_not_general_impossibility():
    # A general allocation may assign videos0,1 to fit, satisfying BOTH fit
    # classes, and2,3 to heldout. Restrictive one-class slots need6videos.
    presence = {0: {10, 11}, 1: {10, 11}, 2: {20}, 3: {20}}
    public, _ = allocate_witness(presence, {'fit': {10, 11}, 'heldout': {20}},
                                 {'fit': {0, 1}, 'heldout': {2, 3}}, {'fit': 2, 'heldout': 2})
    assert public['individual_necessary_support_deficits'] == 0
    assert public['globally_video_disjoint_sufficient_witness_found'] is False
    assert public['failure_proves_general_assignment_impossible'] is False


def test_augmenting_path_reassigns_instead_of_greedy_starvation():
    presence = {0: {10, 20}, 1: {10, 20}, 2: {10, 20}, 3: {10, 20}}
    roles = {'a': {10}, 'b': {20}}
    candidates = {'a': {0, 1, 2}, 'b': {0, 1, 3}}
    public, witness = allocate_witness(presence, roles, candidates, {'a': 2, 'b': 2})
    assert public['matched_slots'] == 4
    assert verify_witness(presence, roles, candidates, {'a': 2, 'b': 2}, witness)


@pytest.mark.parametrize('mutation', ['remove', 'wrong_role', 'wrong_category', 'bad_slot', 'duplicate_slot'])
def test_actual_certificate_mutations_rejected(mutation):
    presence, roles, candidates, minima = inputs()
    _, witness = allocate_witness(presence, roles, candidates, minima)
    changed = copy.deepcopy(witness)
    video = next(v for v, row in changed.items() if row['role'] == 'fit')
    if mutation == 'remove':
        del changed[video]
    elif mutation == 'wrong_role':
        changed[video]['role'] = 'heldout'
    elif mutation == 'wrong_category':
        changed[video]['category_id'] = 999
    elif mutation == 'bad_slot':
        changed[video]['required_slot'] = True
    else:
        other = next(v for v, row in changed.items() if row['role'] == 'fit' and v != video)
        changed[video]['required_slot'] = changed[other]['required_slot']
    with pytest.raises(ValueError):
        verify_witness(presence, roles, candidates, minima, changed)


def test_input_order_does_not_pick_a_different_certificate():
    presence, roles, candidates, minima = inputs()
    first = allocate_witness(presence, roles, candidates, minima)
    assert allocate_witness(dict(reversed(list(presence.items()))), dict(reversed(list(roles.items()))),
                            candidates, minima) == first


def test_missing_support_and_resource_bounds_fail_without_dropping_classes():
    presence, roles, candidates, minima = inputs()
    candidates['heldout'] = {4}
    public, _ = allocate_witness(presence, roles, candidates, minima)
    assert public['individual_necessary_support_deficits'] == 1
    assert public['roles']['heldout']['classes'] == 1
    with pytest.raises(ValueError, match='bound'):
        allocate_witness(presence, roles, candidates, minima, max_slots=4)
    roles['heldout'].add(10)
    with pytest.raises(ValueError, match='Disjoint'):
        allocate_witness(presence, roles, candidates, minima)


def test_independent_verifier_rejects_changed_overlapping_class_roles():
    presence, roles, candidates, minima = inputs()
    _, witness = allocate_witness(presence, roles, candidates, minima)
    roles['heldout'].add(10)
    with pytest.raises(ValueError, match='Disjoint'):
        verify_witness(presence, roles, candidates, minima, witness)
