"""Posthoc diagnostics of retained Train logits; never a policy component."""
from bisect import bisect_right
from collections import Counter
import math
import numpy as np

ACTIONS = ('KNOWN', 'EXISTING', 'NEW', 'WAIT')
EDGES = (-10., -5., -2., -1., 0., 1., 2., 5., 10.)


class LogitForensics:
    def __init__(self):
        self.counts = Counter()
        self.channels = {}
        self.first_mixed_candidate_sequence = None

    def observe(self, logits, features, action, role, known_rank, anonymous_rank,
                costs, target, actual_prefix, wait_bias, sequence, delayed_credit):
        logits = np.asarray(logits, np.float32)
        adjusted = logits.copy()
        adjusted[3] += np.float32(wait_bias)
        if not np.isfinite(logits).all() or logits.shape != (4,) or features is None or len(features) != 17:
            raise ValueError('Invalid retained causal logits/features')
        if int(adjusted.argmax()) != action:
            raise ValueError('Retained action differs from exact float32 frozen operating point')
        if role not in ('known', 'pseudo_novel'):
            raise ValueError('Only permitted Train simulation roles')
        self.counts['queries'] += 1
        self.counts[ACTIONS[action]] += 1
        wrong_known = action == 0 and (role != 'known' or known_rank != 1)
        wrong_existing = action == 1 and costs[1] > 0
        wrong_birth = action == 2 and role == 'known'
        self.counts['wrong_known'] += wrong_known
        self.counts['wrong_existing'] += wrong_existing
        self.counts['known_as_new'] += wrong_birth
        self.counts['duplicate_pseudo_novel_birth'] += action == 2 and role == 'pseudo_novel' and costs[2] > 0
        self.counts['unnecessary_WAIT_under_nominal_current_top1_target'] += action == 3 and target != 3
        self.counts['known_correct_exposed_top1'] += role == 'known' and known_rank == 1
        self.counts['known_correct_exposed_top1_but_wrong_action'] += role == 'known' and known_rank == 1 and action != 0
        self.counts['anonymous_correct_cross_video_exposed_top1'] += role == 'pseudo_novel' and anonymous_rank == 1
        self.counts['anonymous_correct_cross_video_exposed_top1_but_wrong_action'] += role == 'pseudo_novel' and anonymous_rank == 1 and action != 1
        self.counts['known_correct_in_top8_but_not_exposed_top1'] += role == 'known' and known_rank is not None and 1 < known_rank <= 8
        self.counts['pseudo_no_correct_pure_cross_video_candidate_in_actual_memory'] += role == 'pseudo_novel' and anonymous_rank is None
        self.counts['wrong_commit_with_prefix_le2_not_proven_premature'] += actual_prefix <= 2 and (wrong_known or wrong_existing or wrong_birth)
        if delayed_credit is not None:
            self.counts['mixed_top_candidate_queries'] += 1
            self.counts['EXISTING_to_mixed_top_candidate'] += action == 1
            if self.first_mixed_candidate_sequence is None:
                self.first_mixed_candidate_sequence = int(sequence)
        for i, action_name in enumerate(ACTIONS):
            available = bool(features[15 + i]) if i < 2 else True
            for view, values in (('raw', logits), ('effective_WAIT_bias', adjusted)):
                name = f'{action_name}:{view}'
                stats = self.channels.setdefault(name, {'count': 0, 'masked': 0, 'sum': 0., 'sumsq': 0.,
                    'min': None, 'max': None, 'histogram': [0] * (len(EDGES) + 1)})
                if not available:
                    if values[i] > -9e8:
                        raise ValueError('Unavailable action was not masked')
                    stats['masked'] += 1
                    continue
                value = float(values[i])
                stats['count'] += 1
                stats['sum'] += value
                stats['sumsq'] += value * value
                stats['min'] = value if stats['min'] is None else min(value, stats['min'])
                stats['max'] = value if stats['max'] is None else max(value, stats['max'])
                stats['histogram'][bisect_right(EDGES, value)] += 1

    def finish(self):
        channels = {}
        for name, s in self.channels.items():
            n = s['count']
            mean = s['sum'] / n if n else None
            std = math.sqrt(max(s['sumsq'] / n - mean * mean, 0.)) if n else None
            channels[name] = {k: v for k, v in s.items() if k not in ('sum', 'sumsq')}
            channels[name].update(mean=mean, std_population=std)
        return {'counts': dict(self.counts), 'logit_channels': channels, 'histogram_edges': list(EDGES),
                'first_mixed_top_candidate_sequence': self.first_mixed_candidate_sequence,
                'premature_commit_causality': 'NOT_ESTABLISHED_FROM_DIFFERENT_HISTORY_PREFIX_REPLAYS',
                'per_token_trajectory': 'TOKEN_NOT_RETAINED_IN_THIS_TRACE_REQUIRES_SEPARATE_DIAGNOSTIC',
                'data_role': 'EXISTING_TRAIN_FROZEN_CHECKPOINT_POSTHOC_ONLY'}
