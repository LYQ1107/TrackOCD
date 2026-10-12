"""EXTERNAL causal outcome audit; no GT argument to policy/model/memory.

Only separately registered workers may provide actual authorized labels.
This static per-route truth interface is for GT-track controls only. Predicted
tracks need separately qualified frame-level correspondence/ID-switch/unmatched
handling; do not assign one inferred GT category to an entire predicted route.
Truth-unavailable tracks remain physical denominators, not BG/Novel negatives.
Primary reuse requires certified semantic history and another video, not self
continuation. Route moves create conservative provenance parent edges: a fresh
token cannot wash a polluted predecessor's history. This is NOT the final
learned-M split/merge CT definition or a full-semantic-GT H-score/purity metric.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenTruth:
    members: frozenset
    categories: frozenset
    has_Known: bool
    has_unknown: bool

    def certified_for(self, category):
        return not self.has_Known and not self.has_unknown and self.categories == frozenset([category])


@dataclass(frozen=True, slots=True)
class AuditedDecision:
    action: str
    known_correct: bool
    certified_other_physical_reuse: bool
    certified_cross_video_reuse: bool
    uncertifiable_reuse: bool


class CausalCategoryAudit:
    """Audit AFTER atomic predictions, all peers against one pre-frame history.

    No policy/controller reference, scores returned or bank mutation API exists.
    Prefix snapshots are slices of ONE uncapped causal stream, not independent
    globally capped replays. Short tracks use their last sealed available event
    at report time only; full future length is never used by the scorer.
    """
    def __init__(self, external_labels, known_output_bindings, *, prefixes=(1, 2, 4, 8, 16), max_tokens=8753):
        self._labels = dict(external_labels); self._known = dict(known_output_bindings)
        self._prefixes = tuple(prefixes); self._bound = max_tokens
        if (not self._labels or type(max_tokens) is not int or max_tokens < 1
                or any(type(n) is not int or n < 1 for n in self._prefixes)
                or len(set(self._prefixes)) != len(self._prefixes)):
            raise ValueError('Fixed physical scope and observed prefix/token bounds')
        known_values = set(self._known.values())
        if (not self._known or len(known_values) != len(self._known)
                or any(type(k) is not int or type(v) is not int or min(k, v) < 0 for k, v in self._known.items())):
            raise ValueError('Bijective legitimate offline Known output bindings')
        for (video, route), (kind, category) in self._labels.items():
            if type(video) is not int or type(route) is not str or not route:
                raise ValueError('External physical bookkeeping, not numeric model features')
            if kind in ('Known', 'declared_pseudoNovel'):
                if type(category) is not int or category < 0:
                    raise ValueError('Existing authorized semantic truth only')
                if (category in known_values) != (kind == 'Known'):
                    raise ValueError('Frozen disjoint namedKnown / declared pseudo roles')
            elif category is not None:
                raise ValueError('Unknown/outside truth masked, not invented Novel/BG')
        self._members = {}; self._parents = {}; self._latest_anon = {}
        self._seen = Counter(); self._last = {}; self._snapshots = {n: {} for n in self._prefixes}
        self._observations = self._groups = 0; self._first_Known_anon = self._first_pseudo_Known = None

    def token_truth(self, token, *, include_route_ancestors=True):
        members = set(); stack = [token]; visited = set()
        while stack:
            current = stack.pop()
            if current in visited: continue  # Real reassignment cycles remain, never pruned.
            visited.add(current); members.update(self._members.get(current, ()))
            if include_route_ancestors: stack.extend(self._parents.get(current, ()))
        labels = [self._labels[key] for key in members]
        return TokenTruth(frozenset(members), frozenset(c for _, c in labels if c is not None),
                          any(kind == 'Known' for kind, _ in labels), any(c is None for _, c in labels))

    def audit_frame(self, video, result):
        if len(result.events) != len(result.routing_groups) or any(not group for group in result.routing_groups):
            raise ValueError('Aligned complete canonical groups, no silent zip truncation')
        # Compute every chosen Existing summary before appending ANY current
        # peer label. Current-frame GT cannot turn another peer into a positive.
        prior = {}
        for event in result.events:
            if event.action == 'EXISTING':
                if event.token not in self._members: raise ValueError('Existing state must predate frame')
                prior[event.token] = self.token_truth(event.token)
        writes = []; births = set(); current_keys = set()
        for aliases, event in zip(result.routing_groups, result.events):
            group_seq = self._groups; self._groups += 1
            if event.action not in ('KNOWN', 'NEW', 'EXISTING', 'WAIT'):
                raise ValueError('Actual policy actions only')
            if event.action == 'KNOWN' and event.category_id not in self._known:
                raise ValueError('Legal offline Known output bindings')
            if event.action in ('NEW', 'EXISTING') and event.token is None:
                raise ValueError('Anonymous outputs require their actual allocated token')
            if event.action == 'NEW':
                if event.token in self._members or event.token in births:
                    raise ValueError('No recycled birth/GT-reset token')
                births.add(event.token)
            for route in aliases:
                key = (video, route)
                if key not in self._labels or key in current_keys:
                    raise ValueError('Every physical input in fixed scope, unique within frame')
                current_keys.add(key); kind, category = self._labels[key]
                truth = prior.get(event.token)
                certified = kind == 'declared_pseudoNovel' and truth is not None and truth.certified_for(category)
                other = certified and any(member != key for member in truth.members)
                cross = certified and any(member[0] != video for member in truth.members)
                row = AuditedDecision(event.action,
                    kind == 'Known' and event.action == 'KNOWN' and self._known[event.category_id] == category,
                    bool(other), bool(cross), bool(event.action == 'EXISTING' and truth.has_unknown))
                self._seen[key] += 1; self._observations += 1; self._last[key] = row
                if self._seen[key] in self._snapshots: self._snapshots[self._seen[key]][key] = row
                if kind == 'Known' and event.action in ('NEW', 'EXISTING') and self._first_Known_anon is None:
                    self._first_Known_anon = group_seq
                if kind == 'declared_pseudoNovel' and event.action == 'KNOWN' and self._first_pseudo_Known is None:
                    self._first_pseudo_Known = group_seq
                if event.action in ('NEW', 'EXISTING'):
                    writes.append((key, event.token, self._latest_anon.get(key)))
        if len(self._members) + len(births) > self._bound: raise ValueError('Declared token bound, no eviction/drop')
        for key, token, previous in writes:
            self._members.setdefault(token, set()).add(key)
            self._parents.setdefault(token, set())
            if previous is not None and previous != token: self._parents[token].add(previous)
            self._latest_anon[key] = token

    def _report_rows(self, rows):
        if set(rows) != set(self._labels): raise ValueError('No absent/short/unknown physical subset')
        known = defaultdict(list); pseudo = defaultdict(list); pseudo_anon = 0; absorbed = 0
        for key, row in rows.items():
            kind, category = self._labels[key]
            if kind == 'Known': known[category].append(float(row.known_correct))
            if kind == 'declared_pseudoNovel':
                pseudo[category].append(float(row.certified_cross_video_reuse))
                pseudo_anon += int(row.action in ('NEW', 'EXISTING')); absorbed += int(row.action == 'KNOWN')
        if not known or not pseudo: raise ValueError('Full fixed Known/pseudo query strata required')
        mean = lambda v: sum(v) / len(v)
        flatten = lambda d: [v for group in d.values() for v in group]
        return {'physical_tracks': len(rows), 'Known_queries': len(flatten(known)), 'Known_classes_observed': len(known),
            'Known_micro_accuracy': mean(flatten(known)), 'Known_macro_accuracy': mean([mean(v) for v in known.values()]),
            'declared_pseudo_queries': len(flatten(pseudo)), 'declared_pseudo_classes': len(pseudo),
            'certified_cross_video_reuse_micro_lower_bound': mean(flatten(pseudo)),
            'certified_cross_video_reuse_macro_lower_bound': mean([mean(v) for v in pseudo.values()]),
            'declared_pseudo_anonymous_commit_queries': pseudo_anon, 'declared_pseudo_absorbed_as_Known_queries': absorbed,
            'all_physical_commit_track_coverage': sum(r.action != 'WAIT' for r in rows.values()) / len(rows),
            'unscorable_reuse_queries_retained': sum(r.uncertifiable_reuse for r in rows.values())}

    def finish(self):
        if set(self._seen) != set(self._labels): raise ValueError('Every fixed physical track must appear')
        snapshots = {str(n): self._report_rows({key: self._snapshots[n].get(key, self._last[key]) for key in self._labels})
                     for n in self._prefixes}
        snapshots['all_available'] = self._report_rows(self._last)
        categories = Counter()
        for token in self._members:
            truth = self.token_truth(token)
            if truth.has_Known or len(truth.categories) > 1: status = 'verified_non_pseudo_or_mixed_history'
            elif truth.has_unknown: status = 'semantic_unscorable_history'
            else: status = 'certified_pseudo_history'
            categories[status] += 1
        return {'observations': self._observations, 'canonical_groups': self._groups, 'prefix_snapshots': snapshots,
            'strict_route_reassignment_history': dict(categories), 'allocated_tokens': len(self._members),
            'first_Known_anonymous_write_canonical_group': self._first_Known_anon,
            'first_pseudo_absorbed_as_Known_canonical_group': self._first_pseudo_Known,
            'unknown_declared_background_or_Novel': False,
            'prefix_slices_one_uncapped_causal_stream_not_separately_capped_replays': True,
            'self_support_exclusion_or_full_semantic_GT_Hscore_CT_or_Goal_acceptance': False}
