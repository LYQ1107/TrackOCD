"""External version-history semantic lower bounds, no controller feedback.

Only already authorized GT-control targets may be supplied. Unknown remains
unknown; this is NOT full semantic truth, official CT, Pred matching, calibrated
purity, or a self-excluded category-reuse certificate. Current active-member
purity cannot be inferred from numeric weight hashes/active-group counts.
"""
from collections import Counter
from .candidate_training_loss import lawful_label
from .versioned_reliability_memory import MemoryVersion


def _status(labels):
    if any(label is not None and label[0] == 'Known' for label in labels):
        return 'VERIFIED_BAD_ANONYMOUS_KNOWN_HISTORY'
    pseudo = {label for label in labels if label is not None}
    if len(pseudo) > 1: return 'VERIFIED_BAD_MIXED_PSEUDO_HISTORY'
    if not labels or None in labels: return 'UNSCORABLE_EMPTY_OR_UNKNOWN_HISTORY'
    return 'CERTIFIED_ONE_DECLARED_PSEUDO_CLASS_LOWER_BOUND'


def audit_version_lineage(versions, commits, external_targets, epoch_to_video, *, max_union_references=500000):
    """Actual append-only commit watermark + entire parent DAG, no laundering.

    Every new version inherits all ancestor semantic evidence even after moves,
    quarantine, tombstones, recovery or later weight changes. No past correctness
    is rewritten by later pollution: sealed earlier version status stays fixed.
    Bounds fail the WHOLE audit, never evict dirty ancestors or drop denominators.
    """
    versions, commits = tuple(versions), tuple(commits)
    if (len(versions) > 65536 or len(commits) > 65536 or type(max_union_references) is not int
            or max_union_references < 1 or any(type(v) is not MemoryVersion for v in versions)):
        raise ValueError('Bounded actual version/commit history only')
    known_versions, latest, members = {}, {}, {}
    all_counts, latest_counts = Counter(), Counter()
    cursor = references = 0
    pending_transfers = {}
    for version in versions:
        if not cursor <= version.commit_watermark <= len(commits):
            raise ValueError('Monotone actual commit watermark, no future history')
        if cursor < version.commit_watermark and pending_transfers:
            raise ValueError('Every actual transfer needs its destination version ancestry')
        while cursor < version.commit_watermark:
            commit = commits[cursor]; cursor += 1
            key = (commit.video_epoch, commit.routing_id)
            if commit.video_epoch not in epoch_to_video:
                raise ValueError('Complete actual played-video provenance required')
            target = lawful_label(external_targets.get((epoch_to_video[commit.video_epoch], commit.routing_id),
                                                       ('no_retained_semantic_truth', None)))
            if commit.previous_token is not None:
                if key not in members.get(commit.previous_token, {}):
                    raise ValueError('Actual prior member binding required, no oracle transfer')
                del members[commit.previous_token][key]
                if commit.previous_token != commit.token:
                    if commit.previous_token not in latest:
                        raise ValueError('Transfer source needs its preserved past version')
                    pending_transfers.setdefault(commit.token, set()).add(
                        (commit.previous_token, latest[commit.previous_token]))
            members.setdefault(commit.token, {})[key] = target
        binding = (version.token, version.revision)
        if binding in known_versions or version.revision != latest.get(version.token, -1) + 1:
            raise ValueError('Append-only nonrecycled version identity required')
        if version.lifecycle not in ('Tentative', 'Confirmed', 'Quarantined'):
            raise ValueError('Explicit prospective lifecycle')
        labels = set(members.get(version.token, {}).values())
        if version.token in latest and (version.token, latest[version.token]) not in version.parents:
            raise ValueError('New revision must retain previous own-version ancestry')
        required = pending_transfers.pop(version.token, set())
        if not required <= set(version.parents):
            raise ValueError('Actual transfer source ancestry cannot be omitted')
        for parent in version.parents:
            if parent not in known_versions:
                raise ValueError('Parents must be real preserved past versions, no cycles/future edges')
            labels.update(known_versions[parent])
        references += len(labels)
        if references > max_union_references:
            raise ValueError('Whole lineage-union bound, no dirty-ancestor pruning')
        labels = frozenset(labels)
        known_versions[binding] = labels; latest[version.token] = version.revision
        all_counts[_status(labels)] += 1
    if pending_transfers or cursor != len(commits):
        raise ValueError('Every actual commit/transfer must be covered by version records')
    for token, revision in latest.items(): latest_counts[_status(known_versions[(token, revision)])] += 1
    return {'version_records': len(versions), 'actual_commits': len(commits),
        'final_tokens_including_tombstones': len(latest), 'strict_all_version_status_counts': dict(all_counts),
        'strict_latest_version_status_counts': dict(latest_counts),
        'all_parent_history_and_empty_tombstones_retained': True,
        'GT_feedback_or_current_active_version_purity_or_official_CT_or_Pred_claim': False,
        'full_semantic_GT_truth_or_self_excluded_reuse_certificate': False}
