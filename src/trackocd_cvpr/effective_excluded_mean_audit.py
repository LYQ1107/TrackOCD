"""EXTERNAL selected Mean Raw support truth, separate from full ancestry.

Caller must capture the real pre-frame bank and actual query receipt, then call
AFTER sealed predictions under source/authority/whole-input gates. No policy,
memory mutation, asset/label loading or supervision feedback exists here.
Only query-specific excluded Mean is supported, NOT EMA latent or learned M.
Unknown labels prevent certification; labels/keys in returned objects PRIVATE.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from .prefix_member_bank import PrefixMemberBank, CommittedRawSupport
from .independent_support_view import MODES, past_support_excluding
from .self_excluded_mean_control import QueryMeanViewReceipt, preframe_excluded_mean


@dataclass(frozen=True)
class EffectiveMeanTruth:
    # PRIVATE evaluator correspondence, never a scorer input or public row.
    selected_members: frozenset
    categories: frozenset
    status: str
    selected_groups: int
    numeric_prototype_sha256: str
    original_history_repaired_or_full_semantic_truth_or_statistical_independence_accepted: bool = False

    def certified_for_declared_pseudo(self, category):
        if type(category) is not int or category < 0: raise ValueError('External eligible declared pseudo category only')
        return (self.status == 'CERTIFIED_ONE_DECLARED_PSEUDO_EFFECTIVE_SUPPORT_LOWER_BOUND'
                and self.categories == frozenset([category]))

    def independent_reuse_bounds(self, category, query_key):
        if (type(query_key) is not tuple or len(query_key) != 2 or type(query_key[0]) is not int
                or type(query_key[1]) is not str):
            raise ValueError('External actual physical query correspondence required')
        certified = self.certified_for_declared_pseudo(category)
        return {'selected_other_physical_member_reuse_lower_bound': bool(certified and any(k != query_key for k in self.selected_members)),
            'selected_cross_video_member_reuse_lower_bound': bool(certified and any(k[0] != query_key[0] for k in self.selected_members)),
            'semantic_or_statistical_independence_or_full_truth_accepted': False}


def audit_chosen_excluded_mean(preframe_bank, query_routes, actual_receipt, chosen_token,
                             external_labels, epoch_to_video, *, frame_index):
    """Audit the actual selected positive-weight Mean contributors, not parents.

Complete numeric view receipt must match BEFORE any semantic lookup. Missing
provenance/membership/labels is a whole-audit error, not an ignored member.
Any known voter disproves pure anonymous-pseudo support; mixed declared classes
disprove single-class support. Otherwise unknown prevents certification.
Mean groups remain distinct across video/frame scopes; aliases are semantic
correspondence, not extra numeric votes or independent-sample evidence.
"""
    if (type(preframe_bank) is not PrefixMemberBank or type(actual_receipt) is not QueryMeanViewReceipt
            or actual_receipt.mode not in MODES or type(chosen_token) is not str):
        raise ValueError('Exact actual excluded Mean bank/receipt/chosen binding required; no EMA/M shortcut')
    if (set(epoch_to_video) != set(range(preframe_bank._epoch + 1))
            or any(type(k) is not int or type(v) is not int or min(k, v) < 0 for k, v in epoch_to_video.items())
            or len(set(epoch_to_video.values())) != len(epoch_to_video)):
        raise ValueError('All actually played past epochs only, no future or recycled video mapping')
    queries = tuple(query_routes)
    states, expected = preframe_excluded_mean(preframe_bank, queries,
        mode=actual_receipt.mode, frame_index=frame_index)
    if expected != actual_receipt:
        raise ValueError('Actual whole filtered Raw/numeric view receipt differs; never use original dirty prototype as substitute')
    if chosen_token not in states or states[chosen_token].prototype() is None:
        raise ValueError('Actual selected nonempty valid numeric prototype required, no tombstone revival')
    view = past_support_excluding(preframe_bank, queries, mode=actual_receipt.mode)
    def scope(item): return item.epoch, item.frame, item.payload
    current = preframe_bank._epoch; query_set = set(queries)
    removed = {scope(item) for key, item in preframe_bank._members.items() if key[0] == current and key[1] in query_set}
    groups = {row.token: {} for row in view.supports}; members = set()
    for key, item in preframe_bank._members.items():
        drop = item.epoch == current if actual_receipt.mode == 'exclude_current_video' else scope(item) in removed
        if drop: continue
        groups[item.token][scope(item)] = item.payload
        if item.token == chosen_token:
            members.add((epoch_to_video[item.epoch], key[1]))
    reconstructed = tuple(CommittedRawSupport(token, tuple(rows[key] for key in sorted(rows)))
                          for token, rows in sorted(groups.items()))
    if reconstructed != view.supports or not members:
        raise ValueError('Exact complete contributor-to-numeric-group correspondence required')
    # Semantic truth is first accessed here, AFTER every numerical view check.
    labels = []
    for key in sorted(members):
        if key not in external_labels: raise ValueError('Every actual selected physical member remains in fixed external scope')
        kind, category = external_labels[key]
        if kind in ('Known', 'declared_pseudoNovel'):
            if type(category) is not int or category < 0: raise ValueError('Existing authorized semantic truth required')
        elif category is not None: raise ValueError('Unknown/outside labels masked, never BG or invented Novel')
        labels.append((kind, category))
    categories = frozenset(category for _, category in labels if category is not None)
    if any(kind == 'Known' for kind, _ in labels):
        status = 'VERIFIED_BAD_EFFECTIVE_ANONYMOUS_KNOWN_SUPPORT'
    elif len(categories) > 1:
        status = 'VERIFIED_BAD_EFFECTIVE_MIXED_DECLARED_PSEUDO_SUPPORT'
    elif any(category is None for _, category in labels):
        status = 'UNSCORABLE_EFFECTIVE_UNKNOWN_SUPPORT'
    else:
        status = 'CERTIFIED_ONE_DECLARED_PSEUDO_EFFECTIVE_SUPPORT_LOWER_BOUND'
    return EffectiveMeanTruth(frozenset(members), categories, status, len(groups[chosen_token]),
        hashlib.sha256(np.asarray(states[chosen_token].prototype(), dtype='<f8').tobytes()).hexdigest())
