"""Past committed Raw support exclusion prerequisite; no GT or memory edits.

Scope/opaque routing handles are used ONLY to select support, never emitted as
features. This adapter deliberately reads the pinned bank's frozen internal
records: the public Raw snapshot omits provenance. A registered caller must
pin that dependency and take this view before current-frame commits.
Numeric-source tests are not a scientific independence/purity certificate.
"""
from dataclasses import dataclass
import hashlib
from .prefix_member_bank import PrefixMemberBank, CommittedRawSupport, _Support

MODES = ('exclude_query_alias_scope', 'exclude_current_video')


@dataclass(frozen=True)
class ExcludedSupportView:
    supports: tuple[CommittedRawSupport, ...]
    original_groups: int
    retained_groups: int
    original_routed_members: int
    retained_routed_members: int
    mode: str
    independent_semantic_purity_or_full_self_excluded_evaluation_accepted: bool = False


def past_support_excluding(bank, query_routes, *, mode):
    """Exclude whole old alias groups, or every current-video support group.

    Excluding a single query handle is insufficient when its previous payload
    is shared by another exact alias. Whole (epoch,frame,payload) groups are
    removed, not subtracting one cumulative prefix vote. Previous videos with
    identical payloads remain separate scopes, never deduplicated across them.
    Nothing is pruned from the real bank, commits or dirty version history.
    """
    if type(bank) is not PrefixMemberBank or mode not in MODES:
        raise ValueError('Exact pinned bank and explicit exclusion control required')
    queries = tuple(query_routes)
    if (not queries or any(type(q) is not str or not q for q in queries)
            or len(set(queries)) != len(queries)):
        raise ValueError('Nonempty distinct opaque current query alias handles')
    members = tuple(bank._members.items()); epoch = bank._epoch
    if len(members) > 4096 or len(bank._tokens) > 4096:
        raise ValueError('Whole support scope bound, no partial pruning')
    latest = {}
    for commit in bank.commit_history(): latest[(commit.video_epoch, commit.routing_id)] = commit
    if set(latest) != {key for key, _ in members}:
        raise ValueError('Actual full retained members and append-only commit provenance required')
    for key, item in members:
        if (type(item) is not _Support or type(key) is not tuple or len(key) != 2
                or type(key[0]) is not int or type(key[1]) is not str
                or key[0] != item.epoch or not 0 <= item.epoch <= epoch
                or item.token not in bank._tokens or type(item.payload) is not bytes):
            raise ValueError('Pinned frozen past support provenance; no future or injected numeric state')
        record = latest[key]
        if (record.token != item.token or record.frame_index != item.frame
                or hashlib.sha256(item.payload).hexdigest() != record.visible_input_sha256):
            raise ValueError('Actual last committed support, not fabricated or stale route selection')
    def group(item): return item.epoch, item.frame, item.payload
    query_set = set(queries)
    excluded = {group(item) for key, item in members if key[0] == epoch and key[1] in query_set}
    original = bank.raw_support_snapshot()
    by_token = {token: {} for token in bank._tokens}; retained_members = 0
    for _, item in members:
        drop = item.epoch == epoch if mode == 'exclude_current_video' else group(item) in excluded
        if not drop:
            by_token[item.token][group(item)] = item.payload
            retained_members += 1
    supports = tuple(CommittedRawSupport(token, tuple(groups[key] for key in sorted(groups)))
        for token, groups in sorted(by_token.items()))
    return ExcludedSupportView(supports, sum(len(row.payloads) for row in original),
        sum(len(row.payloads) for row in supports), len(members), retained_members, mode)
