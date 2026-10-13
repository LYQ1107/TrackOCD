"""EXTERNAL M chosen-version causal outcomes, GT-control only, no feedback.

Bitsets compact the FULL fixed physical denominator/ancestor history, never
prune dirty members. Strict past-history certification is NOT current-active
purity, self-excluded decision support, full semantic truth, official CT or
frame-matched Pred evidence. Caller must gate source/authority/cache/all9fits.
"""
from collections import Counter
from .causal_category_audit import CausalCategoryAudit, AuditedDecision
from .prefix_member_bank import MemberCommit
from .versioned_reliability_memory import MemoryVersion


class VersionedCausalOutcomeAudit:
    def __init__(self, external_labels, known_bindings, *, prefixes=(1, 2, 4, 8, 16),
                 max_versions=65536, max_commits=65536):
        self._base = CausalCategoryAudit(external_labels, known_bindings, prefixes=prefixes)
        self._labels = dict(external_labels)
        if len(self._labels) > 4096 or any(type(n) is not int or n < 1 for n in (max_versions, max_commits)):
            raise ValueError('Fixed bounded external physical/whole-history scope required')
        # These bookkeeping bit positions NEVER enter model/scorer/ranking.
        self._bits = {key: 1 << i for i, key in enumerate(sorted(self._labels))}
        self._Known = self._unknown = 0; self._categories = {}; self._videos = {}
        for key, (kind, category) in self._labels.items():
            bit = self._bits[key]
            self._videos[key[0]] = self._videos.get(key[0], 0) | bit
            if kind == 'Known': self._Known |= bit
            if category is None: self._unknown |= bit
            if kind == 'declared_pseudoNovel': self._categories[category] = self._categories.get(category, 0) | bit
        self._version_bound, self._commit_bound = max_versions, max_commits
        self._versions, self._commits, self._epoch_videos = (), (), {}
        self._member_token, self._members, self._latest = {}, {}, {}
        self._strict, self._watermarks, self._pending = {}, {}, {}
        self._last = {}; self._snapshots = {n: {} for n in prefixes}; self._seen = Counter()
        self._failed = False

    def start_video(self, epoch, video):
        if (type(epoch) is not int or epoch != len(self._epoch_videos) or video not in self._videos
                or video in self._epoch_videos.values()):
            raise ValueError('Exact actual sequential video epochs, no semantic/GT reset')
        self._epoch_videos[epoch] = video

    def _consume(self, versions, commits):
        versions, commits = tuple(versions), tuple(commits)
        if (len(versions) > self._version_bound or len(commits) > self._commit_bound
                or versions[:len(self._versions)] != self._versions or commits[:len(self._commits)] != self._commits
                or any(type(v) is not MemoryVersion for v in versions[len(self._versions):])
                or any(type(c) is not MemberCommit for c in commits[len(self._commits):])):
            raise ValueError('Actual append-only complete typed versions/commits, no prefix rewriting/eviction')
        cursor = len(self._commits)
        for version in versions[len(self._versions):]:
            if not cursor <= version.commit_watermark <= len(commits):
                raise ValueError('Monotone actual commit watermark')
            if cursor < version.commit_watermark and self._pending:
                raise ValueError('Every actual transfer must create its preserved destination version')
            while cursor < version.commit_watermark:
                commit = commits[cursor]; cursor += 1
                if commit.video_epoch not in self._epoch_videos:
                    raise ValueError('Actual played video provenance before external truth join')
                key = (self._epoch_videos[commit.video_epoch], commit.routing_id)
                if key not in self._bits or self._member_token.get(key) != commit.previous_token:
                    raise ValueError('Full fixed physical scope and actual prior anonymous binding')
                bit = self._bits[key]
                if commit.previous_token is not None:
                    self._members[commit.previous_token] &= ~bit
                    if commit.previous_token != commit.token:
                        if commit.previous_token not in self._latest: raise ValueError('Preserved source version required')
                        self._pending.setdefault(commit.token, set()).add((commit.previous_token, self._latest[commit.previous_token]))
                self._members[commit.token] = self._members.get(commit.token, 0) | bit
                self._member_token[key] = commit.token
            ref = (version.token, version.revision)
            if version.revision != self._latest.get(version.token, -1) + 1 or ref in self._strict:
                raise ValueError('Nonrecycled version identity')
            required = self._pending.pop(version.token, set())
            if version.token in self._latest: required.add((version.token, self._latest[version.token]))
            if not required <= set(version.parents): raise ValueError('No omitted own/actual transfer ancestor')
            mask = self._members.get(version.token, 0)
            for parent in version.parents:
                if parent not in self._strict: raise ValueError('Preserved strictly past parents, no future/cycle')
                mask |= self._strict[parent]
            self._strict[ref], self._watermarks[ref] = mask, version.commit_watermark
            self._latest[version.token] = version.revision
        if cursor != len(commits) or self._pending:
            raise ValueError('Every actual commit/transfer covered; no successful subset')
        self._versions, self._commits = versions, commits

    def audit_frame(self, video, result, chosen_versions, versions, commits):
        if self._failed: raise ValueError('Failed whole audit cannot resume into a favorable subset')
        try:
            self._audit_frame(video, result, chosen_versions, versions, commits)
        except Exception:
            self._failed = True
            raise

    def _audit_frame(self, video, result, chosen_versions, versions, commits):
        """Call AFTER own predictions; Existing certification uses pre-frame version.

        Post-frame peer truth cannot retroactively certify an earlier decision.
        New outputs have no prior reuse certificate even when birth peers match.
        """
        chosen = tuple(chosen_versions); prior_watermark = len(self._commits)
        if len(chosen) != len(result.events) or len(result.events) != len(result.routing_groups):
            raise ValueError('Exact event/physical group/chosen-version alignment')
        self._consume(versions, commits)
        planned = []
        for aliases, event, ref in zip(result.routing_groups, result.events, chosen):
            if event.action == 'EXISTING':
                if (ref is None or ref[0] != event.token or ref not in self._strict
                        or self._watermarks[ref] > prior_watermark):
                    raise ValueError('Actual chosen pre-frame version, no future peer/GT support')
                mask = self._strict[ref]
            else:
                if ref is not None: raise ValueError('Only Existing has chosen prior version')
                mask = 0
            for route in aliases:
                key = (video, route)
                if key not in self._bits: raise ValueError('Every physical input in fixed scope')
                kind, category = self._labels[key]
                allowed = self._categories.get(category, 0)
                certified = bool(kind == 'declared_pseudoNovel' and mask and not mask & ~allowed)
                planned.append((key, AuditedDecision(event.action,
                    kind == 'Known' and event.action == 'KNOWN' and self._base._known.get(event.category_id) == category,
                    certified and bool(mask & ~self._bits[key]), certified and bool(mask & ~self._videos[video]),
                    event.action == 'EXISTING' and bool(mask & self._unknown))))
        self._base.audit_frame(video, result)  # independent actual-route audit, no policy feedback
        for key, row in planned:
            self._seen[key] += 1; self._last[key] = row
            if self._seen[key] in self._snapshots: self._snapshots[self._seen[key]][key] = row

    def _status(self, mask):
        if mask & self._Known: return 'VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'
        present = sum(bool(mask & category) for category in self._categories.values())
        if present > 1: return 'VERIFIED_BAD_MIXED_PSEUDO_HISTORY'
        if not mask or mask & self._unknown: return 'UNSCORABLE_EMPTY_OR_UNKNOWN_HISTORY'
        return 'CERTIFIED_ONE_DECLARED_PSEUDO_CLASS_LOWER_BOUND'

    def finish(self):
        if self._failed: raise ValueError('Failed whole audit, no partial result acceptance')
        if set(self._last) != set(self._labels): raise ValueError('Full physical scope, short/unknown not dropped')
        base = self._base.finish()
        snapshots = {str(n): self._base._report_rows({key: self._snapshots[n].get(key, self._last[key]) for key in self._labels})
                     for n in self._snapshots}
        snapshots['all_available'] = self._base._report_rows(self._last)
        return {'actual_route_history_control': base, 'chosen_preframe_version_prefix_snapshots': snapshots,
            'version_records': len(self._versions), 'actual_commits': len(self._commits),
            'strict_all_version_status_counts': dict(Counter(self._status(mask) for mask in self._strict.values())),
            'strict_latest_version_status_counts': dict(Counter(self._status(self._strict[(t, r)]) for t, r in self._latest.items())),
            'full_physical_bitset_scope': len(self._bits),
            'all_actual_parents_commits_tombstones_and_past_sealed_versions_retained': True,
            'certification_is_past_lineage_not_current_active_independence': True,
            'current_active_purity_official_CT_full_semantic_truth_Pred_or_self_exclusion_claim': False}
