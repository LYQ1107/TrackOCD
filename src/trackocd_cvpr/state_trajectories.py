"""Evaluator-only trajectories around unchanged frozen visual predictions.

This is a diagnostic observer, not reversible memory and not a live controller.
GT statistics are never arguments to visual_step or CandidateMemory.apply.
The inherited numeric-ID tie ordering is deliberately preserved for parity.
"""
from collections import Counter, defaultdict
from src.trackocd_core.persistent_policy import CandidateMemory, SupervisionHistory
from src.trackocd_core.train_first_replay import causal_rows
from src.trackocd_core.evaluation import DecisionEvent, TrackKey, seal_decisions
from src.trackocd_cvpr.checkpoint_diagnostics import visual_step


class ExternalStateAudit:
    def __init__(self):
        self.members = defaultdict(Counter)
        self.videos = defaultdict(set)
        self.states = {}
        self.counts = Counter()

    def snapshot(self, token):
        counts = self.members.get(token, {})
        n = sum(counts.values())
        return {'member_count': n, 'category_count': len(counts),
                'purity': max(counts.values()) / n if n else None,
                'mixed': len(counts) > 1,
                'video_count': len(self.videos.get(token, set()))}

    def observe(self, sequence, video, prediction, candidate_token, category, role):
        kind = prediction['kind']
        self.counts[kind] += 1
        candidate_before = self.snapshot(candidate_token)
        token = prediction.get('token')
        before = self.snapshot(token)
        bad = False
        if kind in {'NEW', 'EXISTING'}:
            if kind == 'NEW':
                if token in self.states:
                    raise ValueError('Token recycled')
                self.states[token] = {'birth_sequence': sequence, 'birth_video': video,
                    'first_bad_write_sequence': None, 'first_cross_category_conflict_sequence': None,
                    'reuse_after_first_bad_write': 0, 'reuse_after_mixed': 0,
                    'queries_exposing_mixed_candidate': 0}
            elif token not in self.states:
                raise ValueError('Absent future token')
            s = self.states[token]
            conflict = bool(set(self.members[token]) - {category})
            bad = role == 'known' or conflict
            self.counts['wrong_existing'] += kind == 'EXISTING' and (role == 'known' or conflict)
            self.counts['cross_category_conflicting_writes'] += conflict
            self.counts['known_anonymous_writes'] += role == 'known'
            self.counts['duplicate_pseudo_novel_birth'] += kind == 'NEW' and role == 'pseudo_novel' and any(
                category in c for other, c in self.members.items() if other != token)
            if kind == 'EXISTING':
                s['reuse_after_first_bad_write'] += s['first_bad_write_sequence'] is not None
                s['reuse_after_mixed'] += before['mixed']
            if bad and s['first_bad_write_sequence'] is None:
                s['first_bad_write_sequence'] = sequence
            if conflict and s['first_cross_category_conflict_sequence'] is None:
                s['first_cross_category_conflict_sequence'] = sequence
            self.members[token][category] += 1
            self.videos[token].add(video)
        elif kind == 'KNOWN':
            self.counts['wrong_known'] += role != 'known' or prediction['category_id'] != category
        if candidate_before['mixed']:
            self.states[candidate_token]['queries_exposing_mixed_candidate'] += 1
        return {'candidate_token': candidate_token, 'selected_token': token,
                'selected_known_id': prediction.get('category_id'),
                'candidate_before': candidate_before, 'selected_before': before,
                'selected_after': self.snapshot(token), 'bad_anonymous_write': bad,
                'first_bad_write_sequence': self.states[token]['first_bad_write_sequence'] if token else None,
                'first_cross_category_conflict_sequence': self.states[token]['first_cross_category_conflict_sequence'] if token else None}

    def finish(self):
        states = [{**s, 'token': token, **self.snapshot(token)} for token, s in self.states.items()]
        return {'counts': dict(self.counts), 'states': states, 'state_count': len(states),
                'mixed_state_count': sum(s['mixed'] for s in states),
                'states_with_bad_write': sum(s['first_bad_write_sequence'] is not None for s in states),
                'category_values_published': False, 'GT_controlled_track_reference': True,
                'premature_commit_causality': 'PENDING_FIXED_HISTORY_CAUSAL_COUNTERFACTUAL',
                'protocol': 'OLD_INDEPENDENT_PREFIX_NUMERIC_ID_ROUTING_NOT_DENSE_ONLINE'}


def replay_state_trajectories(routes, order, prefix, proto, bank, model, labels, constants, wait_bias):
    memory, supervision, observer = CandidateMemory(proto), SupervisionHistory(), ExternalStateAudit()
    events, trace = [], []
    for seq, route in enumerate(causal_rows(routes, order, prefix)):
        item = bank[route['key']]
        action, k, t, cues, logits = visual_step(model, memory, item, wait_bias)
        # Commit solely from visible evidence BEFORE touching any label or audit.
        prediction = memory.apply(action, k, t, item['embedding'], item['quality'])
        label = labels[route['key']]
        role, category = label['simulation_role'], label['category_id']
        if role not in {'known', 'pseudo_novel'}:
            raise ValueError('Only registered Train roles')
        costs, target, credit = supervision.costs(category, role == 'known', k, t, item['actual_prefix'], constants)
        observed = observer.observe(seq, route['video_id'], prediction, t, category, role)
        supervision.update(prediction, category, seq)
        event = DecisionEvent(seq, TrackKey(route['video_id'], str(route['physical_track_id'])),
                              item['actual_prefix'], prediction['kind'],
                              known_category_id=prediction.get('category_id'), token=prediction.get('token'))
        events.append(event)
        trace.append({'sequence': seq, 'video_id': route['video_id'], 'physical_key': route['key'],
            'actual_prefix': item['actual_prefix'], 'action': prediction['kind'],
            'category_id_evaluator_only': int(category), 'role_evaluator_only': role,
            'nominal_target_evaluator_only': int(target), 'delayed_credit_sequence_evaluator_only': credit,
            'logits': logits.tolist(), 'cues': cues.tolist(), **observed})
    return seal_decisions(events, video_order=order, prefix_cap=prefix, known_ids=tuple(proto)), observer.finish(), trace
