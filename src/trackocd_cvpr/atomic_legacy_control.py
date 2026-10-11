"""Transactional current-frame scheduling for a legacy Top1/17-cue control.

Not a candidate-aware scorer, reversible semantic memory or dense benchmark.
Routing IDs reach only the scheduler/output broadcaster, never the policy.
Indistinguishable complete visible prefixes cause one decision and one write.
Legacy prefix-aggregate writes remain unchanged, including their repeated-prefix
weighting; this adapter does not claim to fix that scientific limitation. Raw
quality-weighted evidence/cues are a diagnostic control, not learned A1/A2 or
historical checkpoint parity. No policy is trained or selected by this module.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from src.trackocd_core.persistent_policy import ACTIONS, CandidateMemory
from src.trackocd_cvpr.causal_observation_router import CurrentObservationRouter


@dataclass(frozen=True, slots=True)
class AtomicControlEvent:
    visible_input_sha256: str
    cues_float32_bytes: bytes
    action: str
    category_id: int | None
    token: str | None


@dataclass(frozen=True, slots=True)
class AtomicFrameResult:
    frame_index: int
    events: tuple[AtomicControlEvent, ...]
    routing_groups: tuple[tuple[str, ...], ...]

    def routed_predictions(self):
        """External routing only; no GT lookup or retroactive ledger edits."""
        return tuple((identity, event) for group, event in zip(self.routing_groups, self.events)
                     for identity in group)


class AtomicLegacyControl:
    """Owns current-video prefixes; category memory persists across videos.

    choose_action receives ONLY an immutable17-cue array, and returns an action
    index in KNOWN/EXISTING/NEW/WAIT. Callback purity/trained-checkpoint lineage
    must be audited separately. Copy-on-frame isolates memory and router writes
    until all groups succeed. Callback side effects are not rolled back.
    """
    def __init__(self, known_prototypes, choose_action, *, max_tracks=4096,
                 max_observations=65536, max_anonymous=4096):
        if not callable(choose_action):
            raise ValueError('A policy callback is required')
        if type(max_anonymous) is not int or max_anonymous < 1:
            raise ValueError('Positive anonymous-state bound required')
        if any(np.asarray(value).shape != (768,) for value in known_prototypes.values()):
            raise ValueError('Common768D Known prototypes required')
        if any(type(key) is not int or key < 0 for key in known_prototypes):
            raise ValueError('Canonical offline Known integer keys required')
        self._router = CurrentObservationRouter(max_tracks=max_tracks,
                                               max_observations=max_observations)
        self._memory = CandidateMemory(known_prototypes)
        self._choose_action = choose_action
        self._max_anonymous = max_anonymous

    def start_next_video(self):
        # No physical ID carries across videos; anonymous tokens do persist.
        self._router = CurrentObservationRouter(max_tracks=self._router.max_tracks,
                                               max_observations=self._router.max_observations)

    def state_snapshot(self):
        """Detached external diagnostic snapshot, not model supervision."""
        return deepcopy(self._memory)

    def ingest(self, frame_index, observations):
        router, memory = deepcopy(self._router), deepcopy(self._memory)
        batch = router.ingest(frame_index, observations)
        events = []
        for signature, view in zip(batch.visible_input_sha256, batch.model_views):
            evidence = view.weighted_mean()
            norms = np.linalg.norm(view.visual, axis=1)
            if np.any(norms <= 1e-12):
                raise ValueError('Zero Raw observation cannot define angular uncertainty')
            weights = view.quality.astype(np.float64)
            if weights.sum() <= 1e-12:
                weights = np.ones(len(weights))
            weights /= weights.sum()
            uncertainty = float(np.clip(1 - weights @ ((view.visual / norms[:, None]) @ evidence), 0, 1))
            maturity = float(1 / np.square(weights).sum())
            cues, known_id, token = memory.candidates(
                evidence, uncertainty, maturity, view.quality,
                int(view.elapsed_frames[-1]))
            payload = np.asarray(cues, dtype='<f4').tobytes()
            immutable_cues = np.frombuffer(payload, dtype='<f4')
            action = self._choose_action(immutable_cues)
            if type(action) is not int or not 0 <= action < len(ACTIONS):
                raise ValueError('Policy must return a valid integer action')
            if ACTIONS[action] == 'NEW' and len(memory.anonymous) >= self._max_anonymous:
                raise ValueError('Anonymous-state bound; no silent eviction')
            prediction = memory.apply(action, known_id, token, evidence, view.quality)
            events.append(AtomicControlEvent(signature, payload, prediction['kind'],
                                             prediction.get('category_id'), prediction.get('token')))
        result = AtomicFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._memory = router, memory
        return result
