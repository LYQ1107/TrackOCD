import copy
import numpy as np
import torch
from src.trackocd_core.persistent_policy import CandidateMemory, DecisionMLP
from src.trackocd_cvpr.checkpoint_diagnostics import (
    visual_step, diagnostic_replay, geometry_diagnostics, checkpoint_gradient_replay,
)

CONSTANTS = {'wrong_known': 8., 'wrong_existing_pollution': 10., 'known_as_new': 4.,
             'wrong_new_fragmentation': 3., 'wait_base': .5, 'wait_per_observation': .05,
             'later_predicted_memory_effect': 1.}


def fixture():
    torch.manual_seed(1027)
    model = DecisionMLP().eval().requires_grad_(False)
    rows = [{'key': str(i), 'video_id': i + 1, 'physical_track_id': i,
             'observation_count': 1, 'frame_ids': [1]} for i in range(4)]
    bank = {str(i): {'embedding': np.asarray([1., 0.] if i % 2 == 0 else [0., 1.], np.float32),
                     'uncertainty': 0., 'maturity': 1., 'quality': np.ones(1, np.float32),
                     'elapsed': 0., 'actual_prefix': 1} for i in range(4)}
    labels = {str(i): {'category_id': i % 2 + 1, 'simulation_role': 'known' if i % 2 == 0 else 'pseudo_novel'}
              for i in range(4)}
    proto = {1: np.asarray([1., 0.], np.float32)}
    return model, rows, bank, labels, proto


def fingerprint(sealed):
    return [(e.kind, e.known_category_id, e.token, e.observed_prefix) for e in sealed.events]


def test_labels_roles_and_purity_do_not_change_actual_predictions():
    model, rows, bank, labels, proto = fixture()
    a, _, _ = diagnostic_replay(rows, [1, 2, 3, 4], 1, proto, bank, model, labels, CONSTANTS, 0.)
    shuffled = {k: {'category_id': 900 + i, 'simulation_role': 'pseudo_novel'} for i, k in enumerate(labels)}
    b, _, _ = diagnostic_replay(rows, [1, 2, 3, 4], 1, proto, bank, model, shuffled, CONSTANTS, 0.)
    assert fingerprint(a) == fingerprint(b)


def test_future_append_and_physical_id_renumber():
    model, rows, bank, labels, proto = fixture()
    a, _, _ = diagnostic_replay(rows[:2], [1, 2], 1, proto, bank, model, labels, CONSTANTS, 0.)
    b, _, _ = diagnostic_replay(rows, [1, 2, 3, 4], 1, proto, bank, model, labels, CONSTANTS, 0.)
    assert fingerprint(a) == fingerprint(b)[:2]
    renumbered = [{**r, 'physical_track_id': 7777 - r['physical_track_id']} for r in rows]
    c, _, _ = diagnostic_replay(renumbered, [1, 2, 3, 4], 1, proto, bank, model, labels, CONSTANTS, 0.)
    assert fingerprint(b) == fingerprint(c)


def test_step_accepts_only_observable_inputs():
    model, _, bank, _, proto = fixture()
    memory = CandidateMemory(proto)
    item = copy.deepcopy(bank['0'])
    a = visual_step(model, memory, item, 0.)
    item.update({'category_id': -99, 'role': 'novel', 'GT_purity': False})
    b = visual_step(model, memory, item, 0.)
    assert a[:3] == b[:3]
    np.testing.assert_array_equal(a[4], b[4])
    assert memory.anonymous == {}


def test_gradient_replay_is_not_a_fit():
    model, rows, bank, labels, proto = fixture()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = checkpoint_gradient_replay(model, rows, [1, 2, 3, 4], 1, proto, bank, labels,
                                        CONSTANTS, {1: 1., 2: 1.}, 0., 1., .25)
    assert result['optimizer_steps'] == 0
    assert result['gradient_norms']['nominal_CE'] > 0
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert not any(p.requires_grad for p in model.parameters())


def test_geometry_distinguishes_centered_and_uncentered():
    _, rows, bank, labels, _ = fixture()
    annotated = [{**r, **labels[r['key']]} for r in rows]
    vectors = np.stack([bank[r['key']]['embedding'] for r in rows])
    result = geometry_diagnostics(vectors, annotated, vectors)
    assert result['new_errors_vs_raw'] == 0
    assert result['errors_corrected_vs_raw'] == 0
    assert result['spectra']['uncentered']['effective_rank'] > result['spectra']['centered']['effective_rank']
