"""Actual current-frame atomic controller with prospective M support versions.

Source-only until registered genuine training. All queries see the SAME frozen
pre-frame memory; post-frame revisions cannot change their sealed decisions.
No GT/semantic targets are accepted here. External audit must preserve strict
history/parents even when quarantined members cease contributing prototypes.
"""
from copy import deepcopy
import numpy as np
from .atomic_member_control import AtomicMemberControl, MemberControlEvent, MemberFrameResult
from .candidate_score_contract import build_candidates, resolve_scores
from .versioned_reliability_memory import VersionedReliabilityMemory


class ReliabilityMemberControl(AtomicMemberControl):
    def __init__(self, known_prototypes, score_candidates, member_model, *, quarantine=False,
                 max_versions=65536, **kwargs):
        super().__init__(known_prototypes, score_candidates, **kwargs)
        self._member_model = member_model
        self._memory = VersionedReliabilityMemory(quarantine=quarantine, max_versions=max_versions)
        self._preframe_numeric = {}
        self._chosen_versions = ()

    def memory_versions(self):
        return self._memory.versions()

    def preframe_training_evidence(self):
        # External correspondence only. Loss targets NEVER return to ingest.
        return dict(self._preframe_numeric)

    def chosen_version_bindings(self):
        return self._chosen_versions

    def ingest(self, frame_index, observations):
        router, bank, memory = deepcopy(self._router), deepcopy(self._bank), deepcopy(self._memory)
        batch = router.ingest(frame_index, observations)
        prior, numeric = memory.refresh(bank, self._member_model)
        versions = memory.current_bindings()
        known = {key: np.frombuffer(value, dtype=np.float32) for key, value in self._known.items()}
        events, assignments, chosen, next_id = [], [], [], self._next_id
        for signature, view in zip(batch.visible_input_sha256, batch.model_views):
            candidates = build_candidates(view, known, prior, K=self._K, max_groups=self._candidate_bound)
            features, context = candidates.numeric_inputs()
            scores = np.asarray(self._score(features, context), dtype='<f4').copy()
            action, binding, reason = resolve_scores(candidates, scores)
            category, token = None, None
            if action == 'KNOWN': category = binding
            elif action == 'EXISTING': token = binding
            elif action == 'NEW': token = f'S:{next_id}'; next_id += 1
            assignments.append(token)
            chosen.append(versions.get(token) if action == 'EXISTING' else None)
            events.append(MemberControlEvent(signature, candidates.feature_bytes, candidates.context_bytes,
                candidates.bindings, scores.tobytes(), action, category, token, reason))
        bank.apply_frame(batch, assignments)
        # Births/member moves create append-only post-frame versions, but no
        # partially committed peer state was available to any current decision.
        memory.refresh(bank, self._member_model)
        result = MemberFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._bank, self._memory, self._next_id = router, bank, memory, next_id
        self._preframe_numeric, self._chosen_versions = numeric, tuple(chosen)
        return result
