"""Atomic Raw controller with candidate-only numeric callback and prefix bank.

No trained result, oracle purity, learned quarantine, dense proof or old-checkpoint
parity. Same-frame groups see one pre-frame bank, never partially committed
peer decisions. KNOWN/WAIT do not add/remove anonymous support: retain prior
committed anonymous support; Known-driven detachment is a future explicit rule.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter
from src.trackocd_cvpr.prefix_member_bank import PrefixMemberBank
from src.trackocd_cvpr.candidate_score_contract import build_candidates, resolve_scores, unit


@dataclass(frozen=True, slots=True)
class MemberControlEvent:
    visible_input_sha256: str
    candidate_feature_bytes: bytes
    query_context_bytes: bytes
    candidate_bindings_for_external_loss: tuple
    score_bytes: bytes
    action: str
    category_id: int | None
    token: str | None
    ambiguity_reason: str | None


@dataclass(frozen=True, slots=True)
class MemberFrameResult:
    frame_index: int
    events: tuple[MemberControlEvent, ...]
    routing_groups: tuple[tuple[str, ...], ...]

    def routed_predictions(self):
        return tuple((identity, event) for aliases, event in zip(self.routing_groups, self.events) for identity in aliases)


class AtomicMemberControl:
    def __init__(self, known_prototypes, score_candidates, *, K=8, max_candidate_groups=4096,
                 max_tracks=4096, max_observations=65536, bank_limits=None):
        if not callable(score_candidates) or type(K) is not int or K not in (1, 3, 5, 8):
            raise ValueError('Numeric scorer and finite K required')
        if type(max_candidate_groups) is not int or max_candidate_groups < 1:
            raise ValueError('Positive candidate bound required')
        if any(type(key) is not int or key < 0 for key in known_prototypes):
            raise ValueError('Offline Known output bindings required')
        if len(known_prototypes) > max_candidate_groups:
            raise ValueError('Known-binding bound; no hidden filtering')
        self._known = {key: unit(value).tobytes() for key, value in known_prototypes.items()}
        self._router = CurrentObservationRouter(max_tracks=max_tracks, max_observations=max_observations)
        self._bank = PrefixMemberBank(**(bank_limits or {}))
        self._score = score_candidates
        self._K, self._candidate_bound, self._next_id = K, max_candidate_groups, 0

    def start_next_video(self):
        bank = deepcopy(self._bank)
        bank.start_next_video()
        router = CurrentObservationRouter(max_tracks=self._router.max_tracks, max_observations=self._router.max_observations)
        self._bank, self._router = bank, router

    def state_snapshot(self):
        return deepcopy(self._bank), self._next_id

    def ingest(self, frame_index, observations):
        router, bank = deepcopy(self._router), deepcopy(self._bank)
        batch = router.ingest(frame_index, observations)
        prior = bank.state_evidence()
        known = {key: np.frombuffer(value, dtype=np.float32) for key, value in self._known.items()}
        events, assignments, next_id = [], [], self._next_id
        for signature, view in zip(batch.visible_input_sha256, batch.model_views):
            candidates = build_candidates(view, known, prior, K=self._K, max_groups=self._candidate_bound)
            features, context = candidates.numeric_inputs()
            # Callback receives no routing IDs, token/semantic bindings or GT.
            scores = np.asarray(self._score(features, context), dtype='<f4').copy()
            action, binding, reason = resolve_scores(candidates, scores)
            category, token = None, None
            if action == 'KNOWN':
                category = binding
            elif action == 'EXISTING':
                token = binding
            elif action == 'NEW':
                token = f'S:{next_id}'
                next_id += 1
            assignments.append(token)
            events.append(MemberControlEvent(signature, candidates.feature_bytes, candidates.context_bytes,
                candidates.bindings, scores.tobytes(), action, category, token, reason))
        bank.apply_frame(batch, assignments)  # One bounded commit after every score succeeds.
        result = MemberFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._bank, self._next_id = router, bank, next_id
        return result
