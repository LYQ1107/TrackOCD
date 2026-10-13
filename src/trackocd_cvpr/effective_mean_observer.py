"""External after-sealing effective Mean observer; prediction path unchanged.

Patches only constructor bindings in one registered own process. No existing
source edited, no labels/scores returned to controller or input callbacks.
All full physical/prefix denominators and original dirty-history audit remain.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import struct
import numpy as np
from .causal_category_audit import CausalCategoryAudit, AuditedDecision
from .prefix_member_bank import PrefixMemberBank
from .self_excluded_mean_control import preframe_excluded_mean
from .effective_excluded_mean_audit import audit_chosen_excluded_mean
from .versioned_reliability_memory import MemoryVersion


def audit_unexcluded_mean(bank, selected_version, labels, epochs, *, frame_index):
    """Explicit full Mean baseline, NOT an EMA latent reconstructed as Mean.

An unused private lookup handle requests an unfiltered Raw view, never an
actual model query/feature. Its derived receipt is NOT claimed as a recorded
query receipt: actual chosen full version Raw/prototype bytes must match first.
"""
    if type(bank) is not PrefixMemberBank or type(selected_version) is not MemoryVersion:
        raise ValueError('Actual full Mean snapshot and chosen original version required')
    unbound = '__external_audit_unused__'
    while (bank._epoch, unbound) in bank._members: unbound += '_'
    states, derived = preframe_excluded_mean(bank, (unbound,), mode='exclude_query_alias_scope', frame_index=frame_index)
    token = selected_version.token
    if token not in states or states[token].prototype() is None:
        raise ValueError('Actual valid full Mean prototype required')
    row = next(row for row in bank.raw_support_snapshot() if row.token == token)
    raw_hash = hashlib.sha256()
    for payload in sorted(row.payloads): raw_hash.update(struct.pack('<Q', len(payload))); raw_hash.update(payload)
    if (selected_version.support_sha256 != raw_hash.hexdigest()
            or selected_version.retained_groups != len(row.payloads) or selected_version.active_groups != len(row.payloads)
            or selected_version.commit_watermark > len(bank.commit_history())
            or selected_version.prototype_float64_bytes != np.asarray(states[token].prototype(), dtype='<f8').tobytes()):
        raise ValueError('Chosen actual full Mean Raw/prototype binding, no stale EMA/M substitution')
    return audit_chosen_excluded_mean(bank, (unbound,), derived, token, labels, epochs, frame_index=frame_index)


class EffectiveMeanObserver:
    def __init__(self, labels, bindings, prefixes):
        self._base = CausalCategoryAudit(labels, bindings, prefixes=tuple(prefixes))
        self._labels, self._known = dict(labels), dict(bindings)
        self._seen, self._last, self._snapshots = Counter(), {}, {n: {} for n in prefixes}
        self._last_frame = {}; self._statuses = Counter(); self._private = []; self._failed = False

    def observe(self, video, result, prior, views, chosen, versions, epochs, mode):
        if self._failed: raise ValueError('Failed whole effective audit cannot accept partial results')
        try: self._observe(video, result, prior, views, chosen, versions, epochs, mode)
        except Exception:
            self._failed = True; raise

    def _observe(self, video, result, prior, views, chosen, versions, epochs, mode):
        if (video in self._last_frame and result.frame_index <= self._last_frame[video]
                or epochs.get(prior._epoch) != video or len(chosen) != len(result.events)
                or len(result.events) != len(result.routing_groups)
                or (mode != 'Mean_latest' and len(views) != len(result.events))):
            raise ValueError('Complete actual frame/query/selected-view alignment required')
        version_map = {(v.token, v.revision): v for v in versions}; planned = []; frame_trace = []
        statuses = Counter(); current_keys = set()
        for i, (aliases, event, ref) in enumerate(zip(result.routing_groups, result.events, chosen)):
            truth = None
            if event.action == 'EXISTING':
                if ref not in version_map or ref[0] != event.token:
                    raise ValueError('Actual selected version correspondence required')
                if mode == 'Mean_latest':
                    truth = audit_unexcluded_mean(prior, version_map[ref], self._labels, epochs, frame_index=result.frame_index)
                else:
                    truth = audit_chosen_excluded_mean(prior, aliases, views[i], event.token,
                        self._labels, epochs, frame_index=result.frame_index)
                statuses[truth.status] += 1
            elif ref is not None:
                raise ValueError('Only actual Existing has a pre-frame version')
            event_rows = []
            for route in aliases:
                key = (video, route)
                if key not in self._labels or key in current_keys: raise ValueError('Every fixed physical input exactly once per frame')
                current_keys.add(key); kind, category = self._labels[key]
                certified = truth is not None and kind == 'declared_pseudoNovel' and truth.certified_for_declared_pseudo(category)
                other = certified and any(member != key for member in truth.selected_members)
                cross = certified and any(member[0] != video for member in truth.selected_members)
                # Any unknown contributing label stays unscorable even if a
                # separate Known voter already proves bad anonymous support.
                unknown = truth is not None and any(self._labels[k][1] is None for k in truth.selected_members)
                row = AuditedDecision(event.action,
                    kind == 'Known' and event.action == 'KNOWN' and self._known.get(event.category_id) == category,
                    bool(other), bool(cross), bool(event.action == 'EXISTING' and unknown))
                planned.append((key, row)); event_rows.append(row)
            frame_trace.append((tuple(event_rows), None if truth is None else
                (truth.status, truth.selected_groups, truth.numeric_prototype_sha256)))
        self._last_frame[video] = result.frame_index; self._statuses.update(statuses)
        for key, row in planned:
            self._seen[key] += 1; self._last[key] = row
            if self._seen[key] in self._snapshots: self._snapshots[self._seen[key]][key] = row
        self._private.append((video, result.frame_index, tuple(frame_trace)))

    def finish(self):
        if self._failed or set(self._last) != set(self._labels): raise ValueError('Complete physical effective audit, no failed/subset acceptance')
        snapshots = {str(n): self._base._report_rows({k: self._snapshots[n].get(k, self._last[k]) for k in self._labels})
                     for n in self._snapshots}
        snapshots['all_available'] = self._base._report_rows(self._last)
        return {'effective_selected_support_prefix_snapshots': snapshots,
            'chosen_effective_support_status_counts': dict(self._statuses),
            'full_physical_scope': len(self._labels), 'metadata_frames_observed': len(self._private),
            'physical_observations_observed': sum(self._seen.values()),
            'original_dirty_history_replaced_or_full_semantic_truth_statistical_independence_official_CT_Pred_or_Goal_accepted': False}, tuple(self._private)


@contextmanager
def observed_mean_constructors(labels, bindings, prefixes, epoch_order, *, rename=False, videos=()):
    """Own-process external observer hook; real full trace equality required."""
    from . import self_excluded_policy_rollout as rollout
    base, excluded = rollout.StateEMAMemberControl, rollout.SelfExcludedMeanMemberControl
    if base.__name__ != 'StateEMAMemberControl' or excluded.__name__ != 'SelfExcludedMeanMemberControl':
        raise ValueError('Original constructors, no nested/unknown observer hook')
    if rename:
        routes = {v: {int(t): 'excluded-renamed-' + str(len(a['track_id']) - i - 1)
                      for i, t in enumerate(a['track_id'])} for v, a in videos}
        labels = {(v, routes[v][int(t)]): label for (v, t), label in labels.items()}
    observer = EffectiveMeanObserver(labels, bindings, prefixes)
    def observed_ingest(control, parent, frame, rows, mode):
        prior, _ = control.state_snapshot()
        result = parent(frame, rows)  # original atomic prediction path seals FIRST
        if control._ema.momentum != 0.: raise ValueError('Effective Mean audit cannot reinterpret an EMA latent')
        epochs = {i: epoch_order[i] for i in range(prior._epoch + 1)}
        views = () if mode == 'Mean_latest' else control.query_view_receipts()
        observer.observe(epochs[prior._epoch], result, prior, views, control.chosen_version_bindings(),
            control.memory_versions(), epochs, mode)
        return result
    class ObservedMean(base):
        def ingest(self, frame, rows): return observed_ingest(self, super().ingest, frame, rows, 'Mean_latest')
    class ObservedExcluded(excluded):
        def ingest(self, frame, rows): return observed_ingest(self, super().ingest, frame, rows, self._exclusion_mode)
    rollout.StateEMAMemberControl, rollout.SelfExcludedMeanMemberControl = ObservedMean, ObservedExcluded
    try: yield observer
    finally: rollout.StateEMAMemberControl, rollout.SelfExcludedMeanMemberControl = base, excluded
