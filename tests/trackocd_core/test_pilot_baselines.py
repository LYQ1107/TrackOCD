import numpy as np

from scripts.trackocd_core.run_gt_pilot_baselines import registered_orders, replay
from src.trackocd_core.baselines import anonymous_memory_size, decide_prefix, make_baseline
from src.trackocd_core.features import prefix_view

THRESHOLDS = {"known_cosine_threshold": .65, "existing_cosine_threshold": .55,
              "dpmeans_known_distance_threshold": .35, "dpmeans_lambda_distance": .45}


def visual_view(vectors):
    n = len(vectors)
    return prefix_view(np.asarray(vectors), np.ones((n, 4)), np.ones(n), np.arange(n), n)


def test_frame_vote_provisional_news_do_not_create_hidden_tokens():
    x, y = np.eye(768, dtype=np.float32)[:2]
    model = make_baseline("B0_frame_snapshot_vote", {}, THRESHOLDS)
    first = decide_prefix(model, visual_view([x, y]))
    assert first["kind"] == "NEW" and first["token"] == "A:0"
    assert anonymous_memory_size(model) == 1
    second = decide_prefix(model, visual_view([x, y]))
    assert second["kind"] == "EXISTING" and second["token"] == "A:0"
    assert anonymous_memory_size(model) == 1 and model._anonymous['A:0'][1] == 2


def test_frame_vote_ties_use_first_observation_and_known_does_not_pollute_memory():
    x, y = np.eye(768, dtype=np.float32)[:2]
    model = make_baseline("B0_frame_snapshot_vote", {1: x}, THRESHOLDS)
    action = decide_prefix(model, visual_view([x, y]))
    assert action['kind'] == 'KNOWN' and action['category_id'] == 1
    assert anonymous_memory_size(model) == 0


def test_common_gates_match_and_recovered_arbitration_is_not_rewritten():
    known = np.eye(768, dtype=np.float32)[0]
    vector = np.zeros(768, dtype=np.float32)
    vector[:2] = [.5, np.sqrt(.75)]
    for name in ("B1_track_nearest", "B2_track_dpmeans"):
        model = make_baseline(name, {1: known}, THRESHOLDS)
        assert decide_prefix(model, visual_view([vector]))['kind'] == 'NEW'
        assert decide_prefix(model, visual_view([vector]))['kind'] == 'EXISTING'


def test_registered_orders_and_prefix_completion_routing_are_deterministic():
    orders = registered_orders(list(range(12)))
    assert list(orders) == ['main', 'seed1027', 'seed1028', 'seed1029']
    assert orders == registered_orders(list(reversed(range(12))))
    assert all(set(order) == set(range(12)) for order in orders.values())
    vector = np.eye(768, dtype=np.float32)[0]

    class Cache:
        def get_prefix(self, key, observed):
            return visual_view([vector] * observed)

    # The first-starting track finishes its prefix later; not routed by
    # initial frame, full duration or a GT semantic label.
    routes = [{'key': 'a', 'video_id': 1, 'physical_track_id': 7, 'frame_ids': [0, 10]},
              {'key': 'b', 'video_id': 1, 'physical_track_id': 8, 'frame_ids': [1, 2]}]
    sealed, runtime = replay(Cache(), routes, {}, THRESHOLDS, 'B1_track_nearest', [1], 2)
    assert [e.physical_key.local_track_id for e in sealed.events] == ['8', '7']
    assert [e.kind for e in sealed.events] == ['NEW', 'EXISTING']
    assert runtime['anonymous_memory_counts'] == [1, 1]


def test_external_representation_callback_receives_only_visible_input():
    vector = np.eye(768, dtype=np.float32)[0]

    class Cache:
        def get_prefix(self, key, observed):
            return visual_view([vector] * observed)

    seen = []

    def represent(view):
        assert len(view.visual) == 2
        assert not hasattr(view, 'category_id') and not hasattr(view, 'physical_track_id')
        seen.append(len(view.visual))
        return np.array([1., 0.], dtype=np.float32)

    routes = [{'key': 'private-route-only', 'video_id': 1, 'physical_track_id': 7, 'frame_ids': [0, 1, 999]}]
    sealed, _ = replay(Cache(), routes, {}, THRESHOLDS, 'B1_track_nearest', [1], 2, represent=represent)
    assert seen == [2]
    assert sealed.events[0].kind == 'NEW' and sealed.events[0].observed_prefix == 2
