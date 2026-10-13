"""Source-only, query-specific Mean support exclusion, not an accepted result.

Recompute every candidate from selected pre-frame Raw. Do NOT reuse an EMA
latent containing excluded members. The real prediction-owned bank and its
full dirty Mean version history stay intact; selected versions are therefore
conservative history references, NOT descriptions of filtered prototypes.
The extra view receipts bind those prototypes separately. No labels enter.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import struct
import numpy as np

from .atomic_member_control import MemberControlEvent, MemberFrameResult
from .candidate_score_contract import build_candidates, resolve_scores
from .frame_atomic_state_ema import StateEMAMemberControl
from .independent_support_view import MODES, past_support_excluding
from .prefix_member_bank import PrefixMemberBank, StateEvidence
from .reliability_support_bridge import committed_reliability_inputs


@dataclass(frozen=True)
class QueryMeanViewReceipt:
    mode: str
    original_groups: int
    retained_groups: int
    original_routed_members: int
    retained_routed_members: int
    selected_raw_sha256: str
    numeric_states_sha256: str
    # Original versions remain dirty; they do not bind filtered prototypes.
    original_version_reference_is_conservative_history_only: bool = True
    independent_semantic_purity_or_actual_development_accepted: bool = False


def preframe_excluded_mean(bank, query_routes, *, mode, frame_index):
    """Current frame must be later than ALL committed current-video support.

Opaque aliases/epochs select support only. No current observation, future
input, labels or persistent latent contributes to the numeric recomputation.
Whole bounds are enforced by the original adapter/Raw bridge; no pruning.
"""
    if type(bank) is not PrefixMemberBank:
        raise ValueError('Exact pinned bank required')
    if (type(frame_index) is not int or frame_index < 0
            or bank._last_frame is not None and frame_index <= bank._last_frame):
        raise ValueError('Strict pre-frame support timing required')
    queries = tuple(query_routes)
    view = past_support_excluding(bank, queries, mode=mode)
    # The adapter has validated exact commit/payload provenance. Count retained
    # routing members separately from deduplicated numeric support groups.
    def scope(item): return item.epoch, item.frame, item.payload
    query_set = set(queries)
    excluded = {scope(item) for key, item in bank._members.items()
                if key[0] == bank._epoch and key[1] in query_set}
    routed = {row.token: 0 for row in view.supports}
    retained = {row.token: {} for row in view.supports}
    for item in bank._members.values():
        drop = (item.epoch == bank._epoch if mode == 'exclude_current_video'
                else scope(item) in excluded)
        if not drop:
            routed[item.token] += 1
            retained[item.token][scope(item)] = item
    states, raw_digest, numeric_digest = {}, hashlib.sha256(), hashlib.sha256()
    for row in view.supports:
        token = row.token.encode('utf-8')  # correspondence, not scorer features
        raw_digest.update(struct.pack('<Q', len(token)) + token)
        raw_digest.update(struct.pack('<Q', len(row.payloads)))
        for payload in row.payloads:
            raw_digest.update(struct.pack('<Q', len(payload)) + payload)
        count = len(row.payloads)
        if count:
            raw, _, _ = committed_reliability_inputs(view.supports, row.token)
            vectors = raw.double().numpy()
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise ValueError('Nonzero whole Raw group aggregates required')
            vectors /= norms
            mean = vectors.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            valid = norm > 1e-12
            dispersion = float(np.clip(1 - (vectors @ (mean / norm)).mean(), 0, 1)) if valid else 1.
            state = StateEvidence(row.token, count, routed[row.token],
                np.asarray(mean, dtype='<f8').tobytes(),
                sum(retained[row.token][key].mean_quality for key in sorted(retained[row.token])) / count,
                dispersion, 'DEFINED' if valid else 'ZERO_RESULTANT')
        else:
            state = StateEvidence(row.token, 0, 0, np.zeros(768, dtype='<f8').tobytes(), 0., 0., 'EMPTY')
        states[row.token] = state
        encoded = repr(state).encode('utf-8')
        numeric_digest.update(struct.pack('<Q', len(encoded)) + encoded)
    if sum(routed.values()) != view.retained_routed_members:
        raise ValueError('Complete retained routing scope required')
    receipt = QueryMeanViewReceipt(mode, view.original_groups, view.retained_groups,
        view.original_routed_members, view.retained_routed_members,
        raw_digest.hexdigest(), numeric_digest.hexdigest())
    return states, receipt


class SelfExcludedMeanMemberControl(StateEMAMemberControl):
    """NEW source-only Mean control; neither state-EMA nor fitted M.

All queries use one frozen pre-frame bank, independently filtered per alias
group. Commit once AFTER every score succeeds. Original dirty versions and
transfers persist. Chosen version bindings audit original history only; a
registered evaluator MUST also retain/query-bind these numeric view receipts.
No automatic experiment, loss, asset loading or scientific acceptance here.
"""
    def __init__(self, known_prototypes, score_candidates, *, exclusion_mode, **kwargs):
        if exclusion_mode not in MODES or 'momentum' in kwargs:
            raise ValueError('Explicit supported exclusion mode; Mean only, no EMA latent')
        super().__init__(known_prototypes, score_candidates, momentum=0., **kwargs)
        self._exclusion_mode, self._query_views = exclusion_mode, ()

    def query_view_receipts(self): return self._query_views

    def ingest(self, frame_index, observations):
        router, bank, history = deepcopy(self._router), deepcopy(self._bank), deepcopy(self._ema)
        batch = router.ingest(frame_index, observations)
        history.refresh(bank)  # FULL original dirty history, not filtered state
        bindings = history.bindings()
        known = {k: np.frombuffer(v, dtype=np.float32) for k, v in self._known.items()}
        events, assignments, chosen, views, next_id = [], [], [], [], self._next_id
        for aliases, signature, query in zip(batch.routing_groups, batch.visible_input_sha256, batch.model_views):
            prior, receipt = preframe_excluded_mean(bank, aliases,
                mode=self._exclusion_mode, frame_index=batch.frame_index)
            candidates = build_candidates(query, known, prior, K=self._K, max_groups=self._candidate_bound)
            features, context = candidates.numeric_inputs()
            scores = np.asarray(self._score(features, context), dtype='<f4').copy()
            action, binding, reason = resolve_scores(candidates, scores)
            category, token = None, None
            if action == 'KNOWN': category = binding
            elif action == 'EXISTING': token = binding
            elif action == 'NEW': token = f'S:{next_id}'; next_id += 1
            assignments.append(token)
            chosen.append(bindings.get(token) if action == 'EXISTING' else None)
            views.append(receipt)
            events.append(MemberControlEvent(signature, candidates.feature_bytes, candidates.context_bytes,
                candidates.bindings, scores.tobytes(), action, category, token, reason))
        bank.apply_frame(batch, assignments)
        history.refresh(bank)
        result = MemberFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._bank, self._ema, self._next_id = router, bank, history, next_id
        self._chosen, self._query_views = tuple(chosen), tuple(views)
        return result
