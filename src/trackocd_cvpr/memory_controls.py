"""Train-only bounded memory/known-count and routing counterfactuals."""
from collections import Counter
import hashlib
import numpy as np
from src.trackocd_core.persistent_policy import CandidateMemory
from src.trackocd_core.train_first_replay import causal_rows
from src.trackocd_core.evaluation import DecisionEvent, TrackKey, seal_decisions
from src.trackocd_cvpr.checkpoint_diagnostics import visual_step


class CapacityMemory(CandidateMemory):
    def __init__(self, prototypes, capacity):
        super().__init__(prototypes)
        if capacity is not None and capacity < 1:
            raise ValueError('Invalid anonymous capacity')
        self.capacity = capacity
        self.evictions = 0

    def apply(self, action, known_id, token, evidence, quality):
        prediction = super().apply(action, known_id, token, evidence, quality)
        # FIFO uses insertion/birth order only, no category, purity or future.
        while self.capacity is not None and len(self.anonymous) > self.capacity:
            del self.anonymous[next(iter(self.anonymous))]
            self.evictions += 1
        return prediction


def prototype_subset(prototypes, count):
    """Fixed visual-hash order, not semantic ID or Val-selected class choice."""
    if count > len(prototypes) or count < 1:
        raise ValueError('Unsupported Known count')
    order = sorted(prototypes, key=lambda k: hashlib.sha256(np.asarray(prototypes[k], '<f4').tobytes()).digest())
    return {k: prototypes[k] for k in order[:count]}


def controlled_replay(routes, order, prefix, prototypes, bank, predictor, wait_bias, capacity):
    memory = CapacityMemory(prototypes, capacity)
    events, counts = [], Counter()
    maximum = 0
    for seq, route in enumerate(causal_rows(routes, order, prefix)):
        item = bank[route['key']]
        action, k, t, _, _ = visual_step(predictor, memory, item, wait_bias)
        pred = memory.apply(action, k, t, item['embedding'], item['quality'])
        counts[pred['kind']] += 1
        maximum = max(maximum, len(memory.anonymous))
        events.append(DecisionEvent(seq, TrackKey(route['video_id'], str(route['physical_track_id'])),
                                    item['actual_prefix'], pred['kind'],
                                    known_category_id=pred.get('category_id'), token=pred.get('token')))
    return seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=tuple(prototypes)), {
        'counts': dict(counts), 'capacity': capacity, 'evictions': memory.evictions,
        'maximum_resident_states': maximum, 'final_resident_states': len(memory.anonymous),
        'known_prototype_count': len(prototypes), 'eviction_rule': 'FIFO_BIRTH_ORDER_NO_GT',
        'commitment_coverage': 1 - counts['WAIT'] / max(len(routes), 1),
        'diagnostic_distribution_shift_not_new_method': True,
    }


def renumber_routes(routes):
    return [{**r, 'physical_track_id': 1000000000 - int(r['physical_track_id'])} for r in routes]


def compare_renumbered(left, right, original, renumbered, order, prefix):
    def decisions(sealed, routes):
        keys = {(r['video_id'], str(r['physical_track_id'])): r['key'] for r in routes}
        by = {}
        for e in sealed.events:
            by[keys[(e.physical_key.video_id, e.physical_key.local_track_id)]] = (e.kind, e.known_category_id, e.token)
        return by
    a, b = decisions(left, original), decisions(right, renumbered)
    if set(a) != set(b):
        raise ValueError('Renumbering changed physical universe')
    order_a = [r['key'] for r in causal_rows(original, order, prefix)]
    order_b = [r['key'] for r in causal_rows(renumbered, order, prefix)]
    return {'physical_tracks': len(a), 'action_or_token_changed': sum(a[k] != b[k] for k in a),
            'event_order_positions_changed': sum(x != y for x, y in zip(order_a, order_b)),
            'token_numbers_not_permutation_aligned': True,
            'raw_decision_difference_is_stricter_than_cluster_equivalence': True,
            'old_routing_uses_numeric_ID_for_equal_time_ties': True}
