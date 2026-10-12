"""Causal state-EMA of the complete latest-member Raw mean, ZERO fit parameters.

This is explicitly a STATE-estimate EMA, not an incoming-observation EMA or a
new independent vote per cumulative prefix. One update per changed token/frame
Raw multiset; same-frame queries use its prior state. Raw members/commit/parent
history remain intact. It does not certify semantic purity or self-exclusion.
"""
from copy import deepcopy
import hashlib
import struct
import numpy as np
from .prefix_member_bank import StateEvidence
from .versioned_reliability_memory import MemoryVersion
from .reliability_support_bridge import committed_reliability_inputs
from .atomic_member_control import AtomicMemberControl, MemberControlEvent, MemberFrameResult
from .candidate_score_contract import build_candidates, resolve_scores


class FrameAtomicStateEMA:
    def __init__(self, *, momentum=.9, max_versions=65536):
        if (type(momentum) is not float or not 0 <= momentum < 1
                or type(max_versions) is not int or max_versions < 1):
            raise ValueError('Explicit fixed momentum/whole-version bound required')
        self.momentum = momentum
        self._bound = max_versions
        self._versions, self._current, self._latent, self._commit_count = (), {}, {}, 0

    def __deepcopy__(self, memo):
        other = type(self).__new__(type(self)); memo[id(self)] = other
        other.momentum, other._bound = self.momentum, self._bound
        other._versions, other._current = self._versions, dict(self._current)
        other._latent, other._commit_count = dict(self._latent), self._commit_count
        return other

    def versions(self): return self._versions

    def bindings(self): return {t: (t, v.revision) for t, v in self._current.items()}

    def refresh(self, bank):
        snapshot, commits = bank.raw_support_snapshot(), bank.commit_history()
        if len(commits) < self._commit_count: raise ValueError('Actual history cannot shrink')
        prior, staged, latent = dict(self._current), dict(self._current), dict(self._latent)
        touched, transfers = set(), {}
        for commit in commits[self._commit_count:]:
            touched.add(commit.token)
            if commit.previous_token is not None and commit.previous_token != commit.token:
                touched.add(commit.previous_token)
                if commit.previous_token not in prior: raise ValueError('Preserved transfer source version required')
                transfers.setdefault(commit.token, set()).add((commit.previous_token, prior[commit.previous_token].revision))
        original = bank.state_evidence(); states, records = {}, []
        for row in snapshot:
            fingerprint = hashlib.sha256()
            for payload in sorted(row.payloads):
                fingerprint.update(struct.pack('<Q', len(payload))); fingerprint.update(payload)
            support_hash = fingerprint.hexdigest(); old = prior.get(row.token)
            # No second EMA update merely because pre/post-frame refresh repeats
            # an unchanged snapshot. Exact alias multiplicity is not a new vote.
            changed_raw = old is None or old.support_sha256 != support_hash
            parents = set(transfers.get(row.token, ()))
            if old is not None: parents.add((row.token, old.revision))
            count = len(row.payloads)
            if count:
                raw, _, _ = committed_reliability_inputs(snapshot, row.token)
                vectors = raw.double().numpy()
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                mean = vectors.mean(axis=0)  # unnormalized: zero mean is valid
                previous = latent.get(row.token)
                if changed_raw:
                    value = mean if previous is None else (self.momentum * np.frombuffer(previous, dtype='<f8')
                                                           + (1 - self.momentum) * mean)
                    latent[row.token] = np.asarray(value, dtype='<f8').tobytes()
                value = np.frombuffer(latent[row.token], dtype='<f8')
                norm = float(np.linalg.norm(value)); valid = norm > 1e-12
                prototype = value / norm if valid else None
                dispersion = float(np.clip(1 - (vectors @ prototype).mean(), 0, 1)) if valid else 1.
                states[row.token] = StateEvidence(row.token, count, original[row.token].routed_members,
                    value.tobytes(), original[row.token].mean_quality, dispersion, 'DEFINED' if valid else 'ZERO_RESULTANT')
                prototype_bytes = None if prototype is None else np.asarray(prototype, dtype='<f8').tobytes()
                lifecycle = 'Tentative' if valid else 'Quarantined'
            else:
                # Empty tombstones cannot become candidates from old latent EMA.
                # If later revived, old latent/history persists: no clean reset.
                states[row.token] = original[row.token]; prototype_bytes = None; lifecycle = 'Quarantined'
            numeric_hash = hashlib.sha256(struct.pack('<d', self.momentum)
                + (latent.get(row.token, b'')) + struct.pack('<Q', count)).hexdigest()
            if changed_raw or row.token in touched or old is None:
                record = MemoryVersion(row.token, 0 if old is None else old.revision + 1, tuple(sorted(parents)),
                    support_hash, numeric_hash, lifecycle, count, count, len(commits), prototype_bytes)
                staged[row.token] = record; records.append(record)
        if len(self._versions) + len(records) > self._bound: raise ValueError('Whole EMA version bound, no eviction')
        self._versions += tuple(records); self._current, self._latent = staged, latent
        self._commit_count = len(commits)
        return states


class StateEMAMemberControl(AtomicMemberControl):
    """Atomic actual controller, Mean(momentum0) or state-EMA(momentum.9).

    No trained weights. Known/WAIT keep old anonymous Raw support just as the
    registered controls do; no GT-driven detachment or deleted wrong history.
    """
    def __init__(self, known_prototypes, score_candidates, *, momentum=.9, max_versions=65536, **kwargs):
        super().__init__(known_prototypes, score_candidates, **kwargs)
        self._ema = FrameAtomicStateEMA(momentum=momentum, max_versions=max_versions)
        self._chosen = ()

    def memory_versions(self): return self._ema.versions()

    def chosen_version_bindings(self): return self._chosen

    def ingest(self, frame_index, observations):
        router, bank, ema = deepcopy(self._router), deepcopy(self._bank), deepcopy(self._ema)
        batch = router.ingest(frame_index, observations)
        prior = ema.refresh(bank); bindings = ema.bindings()
        known = {k: np.frombuffer(v, dtype=np.float32) for k, v in self._known.items()}
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
            assignments.append(token); chosen.append(bindings.get(token) if action == 'EXISTING' else None)
            events.append(MemberControlEvent(signature, candidates.feature_bytes, candidates.context_bytes,
                candidates.bindings, scores.tobytes(), action, category, token, reason))
        bank.apply_frame(batch, assignments); ema.refresh(bank)
        result = MemberFrameResult(batch.frame_index, tuple(events), batch.routing_groups)
        self._router, self._bank, self._ema, self._next_id = router, bank, ema, next_id
        self._chosen = tuple(chosen)
        return result
