from src.trackocd_cvpr.checkpoint_diagnostics import diagnostic_replay
from src.trackocd_cvpr.state_trajectories import ExternalStateAudit, replay_state_trajectories
from tests.trackocd_cvpr.test_checkpoint_diagnostics import fixture, fingerprint, CONSTANTS


def test_state_replay_exact_prediction_parity_and_label_role_mutation():
    model, rows, bank, labels, proto = fixture()
    order = [1, 2, 3, 4]
    old, _, _ = diagnostic_replay(rows, order, 1, proto, bank, model, labels, CONSTANTS, -2.)
    new, _, trace = replay_state_trajectories(rows, order, 1, proto, bank, model, labels, CONSTANTS, -2.)
    assert fingerprint(old) == fingerprint(new)
    altered = {k: {'category_id': 1000 + i, 'simulation_role': 'pseudo_novel'} for i, k in enumerate(labels)}
    changed, _, _ = replay_state_trajectories(rows, order, 1, proto, bank, model, altered, CONSTANTS, -2.)
    assert fingerprint(new) == fingerprint(changed)
    assert len(trace) == len(rows)


def test_purity_known_bad_binding_and_conflict_are_distinct():
    a = ExternalStateAudit()
    birth = a.observe(0, 1, {'kind': 'NEW', 'token': 'S:0'}, None, 1, 'known')
    assert birth['first_bad_write_sequence'] == 0
    assert birth['first_cross_category_conflict_sequence'] is None
    pure = a.observe(1, 2, {'kind': 'EXISTING', 'token': 'S:0'}, 'S:0', 1, 'known')
    assert pure['selected_before']['purity'] == pure['selected_after']['purity'] == 1.
    conflict = a.observe(2, 3, {'kind': 'EXISTING', 'token': 'S:0'}, 'S:0', 2, 'pseudo_novel')
    assert conflict['first_cross_category_conflict_sequence'] == 2
    assert conflict['selected_after']['purity'] == 2 / 3
    assert conflict['candidate_before']['mixed'] is False
    wait = a.observe(3, 4, {'kind': 'WAIT'}, 'S:0', 3, 'pseudo_novel')
    assert wait['candidate_before']['mixed'] is True
    assert a.finish()['states'][0]['reuse_after_first_bad_write'] == 2
    assert a.finish()['states'][0]['reuse_after_mixed'] == 0
    assert a.finish()['states'][0]['queries_exposing_mixed_candidate'] == 1


def test_duplicate_birth_is_not_cross_category_pollution():
    a = ExternalStateAudit()
    a.observe(0, 1, {'kind': 'NEW', 'token': 'S:0'}, None, 2, 'pseudo_novel')
    a.observe(1, 2, {'kind': 'NEW', 'token': 'S:1'}, 'S:0', 2, 'pseudo_novel')
    r = a.finish()
    assert r['counts']['duplicate_pseudo_novel_birth'] == 1
    assert r['mixed_state_count'] == 0 and r['states_with_bad_write'] == 0
