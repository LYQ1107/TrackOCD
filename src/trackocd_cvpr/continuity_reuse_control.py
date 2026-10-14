"""Source-only factorial separation of routing continuity from category reuse.

Raw geometry/scoring/commit semantics stay unchanged. Whole own alias groups
may be excluded from category proposals; a NEW resolution may instead continue
ONE prior route token. This is explicitly NOT a scored category match. Known
and WAIT never trigger continuation, and no earlier history is erased. No GT,
weights, data, optimizer, purity or accepted empirical result in this module.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from .atomic_member_control import AtomicMemberControl, MemberControlEvent, MemberFrameResult
from .candidate_score_contract import build_candidates, resolve_scores
from .independent_support_view import past_support_excluding
from .prefix_member_bank import PrefixMemberBank

MODES = ('full_raw', 'full_raw_with_continuity', 'exclude_query_raw',
         'exclude_query_raw_with_continuity')


@dataclass(frozen=True, slots=True)
class ContinuityReceipt:
    scored_action: str
    scored_binding: int | str | None
    decision_origin: str
    previous_route_tokens: tuple[str, ...]
    original_support_groups: int
    retained_support_groups: int
    category_match_or_independent_support_certified: bool = False


def excluded_raw_state(bank, aliases, frame):
    """Validated past whole-group exclusion, EXACT original Raw sum geometry.

    Reuse the bank-produced float32 prefix vectors and original sum/quality/
    dispersion, without re-decoding into a separate Mean kernel. History survives.
    The temporary view is never committed or passed off as a clean history.
    """
    if type(bank) is not PrefixMemberBank:
        raise ValueError('Exact pinned Raw bank required')
    aliases = tuple(aliases)
    if type(frame) is not int or frame < 0 or bank._last_frame is not None and frame <= bank._last_frame:
        raise ValueError('Strict pre-frame Raw view required')
    checked = past_support_excluding(bank, aliases, mode='exclude_query_alias_scope')
    alias_set = set(aliases)
    scope = lambda s:(s.epoch,s.frame,s.payload)
    excluded = {scope(s) for key,s in bank._members.items()
                if key[0] == bank._epoch and key[1] in alias_set}
    view = deepcopy(bank)
    view._members = {key:s for key,s in bank._members.items() if scope(s) not in excluded}
    if view.raw_support_snapshot() != checked.supports:
        raise ValueError('Complete validated alias-group Raw view, no partial subtraction')
    return view.state_evidence(), checked.original_groups, checked.retained_groups


class ContinuityReuseControl(AtomicMemberControl):
    """Four finite arms; scores are numeric-only, continuation routing-only.

    Every query sees the same pre-frame bank, commits only after all succeed.
    An output EXISTING with ROUTE_CONTINUATION origin did NOT select a scored
    Existing candidate. External loss/audit must retain that distinction, never
    award semantic reuse or purity from this routing predicate alone.
    """
    def __init__(self, known_prototypes, score_candidates, *, mode, **kwargs):
        if mode not in MODES:raise ValueError('Exact four factorial continuity/reuse arms')
        super().__init__(known_prototypes, score_candidates, **kwargs)
        self._mode, self._continuity_receipts = mode, ()

    def continuity_receipts(self):return self._continuity_receipts

    def start_next_video(self):
        super().start_next_video()
        self._continuity_receipts = ()

    def ingest(self, frame_index, observations):
        router, bank = deepcopy(self._router), deepcopy(self._bank)
        batch = router.ingest(frame_index, observations)
        full = bank.state_evidence()
        original_groups = sum(s.support_groups for s in full.values())
        known = {key:np.frombuffer(value,dtype=np.float32) for key,value in self._known.items()}
        events, assignments, receipts, next_id = [], [], [], self._next_id
        for aliases, signature, query in zip(batch.routing_groups,batch.visible_input_sha256,batch.model_views):
            if self._mode.startswith('exclude_query_raw'):
                prior, original, retained = excluded_raw_state(bank,aliases,batch.frame_index)
            else:prior, original, retained = full, original_groups, original_groups
            snapshot = build_candidates(query,known,prior,K=self._K,max_groups=self._candidate_bound)
            features, context = snapshot.numeric_inputs()
            scores = np.asarray(self._score(features,context),dtype='<f4').copy()
            action, binding, reason = resolve_scores(snapshot,scores)
            scored_action, scored_binding = action, binding
            previous = tuple(sorted({bank._members[(bank._epoch,r)].token for r in aliases
                                    if (bank._epoch,r) in bank._members}))
            category, token, origin = None, None, 'SCORED_CANDIDATE'
            if action == 'KNOWN':category = binding
            elif action == 'EXISTING':token = binding
            elif action == 'NEW':
                if self._mode.endswith('_with_continuity') and previous:
                    if len(previous) == 1:
                        action, token, origin = 'EXISTING', previous[0], 'ROUTE_CONTINUATION_NOT_CATEGORY_MATCH'
                    else:
                        action, reason, origin = 'WAIT', 'CONFLICTING_ROUTE_CONTINUITY', 'CONFLICTING_ROUTE_CONTINUITY'
                else:token = f'S:{next_id}'; next_id += 1
            assignments.append(token)
            receipts.append(ContinuityReceipt(scored_action,scored_binding,origin,previous,original,retained))
            events.append(MemberControlEvent(signature,snapshot.feature_bytes,snapshot.context_bytes,
                snapshot.bindings,scores.tobytes(),action,category,token,reason))
        bank.apply_frame(batch,assignments)
        result = MemberFrameResult(batch.frame_index,tuple(events),batch.routing_groups)
        self._router, self._bank, self._next_id = router, bank, next_id
        self._continuity_receipts = tuple(receipts)
        return result
