"""Posthoc exact reconstruction from already retained private state rows."""
from collections import Counter


class CrossVideoTrajectories:
    def __init__(self):
        self.next_sequence = 0
        self.states = {}

    def snapshot(self, token):
        s = self.states.get(token)
        c = s['categories'] if s else {}
        n = sum(c.values())
        return {'member_count': n, 'category_count': len(c),
                'purity': max(c.values()) / n if n else None,
                'mixed': len(c) > 1, 'video_count': len(s['videos']) if s else 0}

    def consume(self, row):
        if row['sequence'] != self.next_sequence:
            raise ValueError('Missing/duplicate/out-of-order retained sequence')
        self.next_sequence += 1
        token, candidate = row['selected_token'], row['candidate_token']
        if row['candidate_before'] != self.snapshot(candidate) or row['selected_before'] != self.snapshot(token):
            raise ValueError('Retained prior membership/purity differs from actual chronology')
        if row['action'] in {'NEW', 'EXISTING'}:
            if row['action'] == 'NEW':
                if token in self.states:
                    raise ValueError('Recycled anonymous identity')
                self.states[token] = {'categories': Counter(), 'videos': set(), 'first_bad_write_sequence': None,
                    'cross_video_existing_reuses': 0, 'cross_video_existing_reuses_after_bad_write': 0,
                    'cross_video_existing_reuses_after_mixed': 0, 'existing_reuses': 0}
            elif token not in self.states:
                raise ValueError('Absent/future state')
            s = self.states[token]
            if row['action'] == 'EXISTING':
                cross = bool(s['videos'] - {row['video_id']})
                s['existing_reuses'] += 1
                s['cross_video_existing_reuses'] += cross
                s['cross_video_existing_reuses_after_bad_write'] += cross and s['first_bad_write_sequence'] is not None
                s['cross_video_existing_reuses_after_mixed'] += cross and len(s['categories']) > 1
            s['categories'][row['category_id_evaluator_only']] += 1
            s['videos'].add(row['video_id'])
            s['first_bad_write_sequence'] = row['first_bad_write_sequence']
        if row['selected_after'] != self.snapshot(token):
            raise ValueError('Retained post-write membership/purity differs')

    def finish(self):
        return [{'token': token, **self.snapshot(token),
                 **{k: v for k, v in s.items() if k not in {'categories', 'videos', 'first_bad_write_sequence'}}}
                for token, s in self.states.items()]
