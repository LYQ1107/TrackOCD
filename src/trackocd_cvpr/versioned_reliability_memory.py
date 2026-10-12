"""Prospective numeric memory versions, no GT-driven repair or past rollback.

Untrained M integration prerequisite. Lifecycle gates are explicit experimental
constants, NOT calibrated posterior/purity thresholds or independent support
certificates. All Raw members, commits, tombstones and old versions survive.
Strict semantic lineage/CT must be measured externally, including every parent.
"""
from dataclasses import dataclass
import hashlib
import struct
import numpy as np
import torch

from .prefix_member_bank import StateEvidence
from .member_reliability import weighted_support
from .reliability_support_bridge import committed_reliability_inputs


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    token: str
    revision: int
    parents: tuple[tuple[str, int], ...]
    support_sha256: str
    numeric_sha256: str
    lifecycle: str
    retained_groups: int
    active_groups: int
    commit_watermark: int
    prototype_float64_bytes: bytes | None


class VersionedReliabilityMemory:
    def __init__(self, *, quarantine=False, max_versions=65536,
                 quarantine_gate=.25, confirmation_gate=.75, confirmation_dispersion=.25):
        if (type(quarantine) is not bool or type(max_versions) is not int or max_versions < 1
                or not 0 < quarantine_gate < confirmation_gate < 1
                or not 0 <= confirmation_dispersion <= 1):
            raise ValueError('Explicit finite numeric lifecycle gates/bounds required')
        self.quarantine = quarantine
        self._gates = quarantine_gate, confirmation_gate, confirmation_dispersion
        self._max = max_versions
        self._versions, self._current, self._commit_count = (), {}, 0

    def __deepcopy__(self, memo):
        """Frozen version/parent/prototype bytes share; mutable bindings copy."""
        other = type(self).__new__(type(self)); memo[id(self)] = other
        other.quarantine, other._gates, other._max = self.quarantine, self._gates, self._max
        other._versions, other._current = self._versions, dict(self._current)
        other._commit_count = self._commit_count
        return other

    def versions(self):
        return self._versions

    def current_bindings(self):
        return {token: (token, v.revision) for token, v in self._current.items()}

    def refresh(self, bank, model):
        """Use only already committed prefixes and actual member moves.

        Returns effective candidate states and graph-retaining numeric evidence
        for an EXTERNAL loss. Validation is transactional; failure cannot erase
        history. No labels/role/physical-ID arithmetic enters model forward.
        """
        snapshot, commits = bank.raw_support_snapshot(), bank.commit_history()
        if len(commits) < self._commit_count:
            raise ValueError('Append-only actual commit history cannot shrink')
        prior = dict(self._current); transfers = {}
        for commit in commits[self._commit_count:]:
            if commit.previous_token is not None and commit.previous_token != commit.token:
                if commit.previous_token not in prior:
                    raise ValueError('Transfer source requires its preserved prior version')
                transfers.setdefault(commit.token, set()).add(
                    (commit.previous_token, prior[commit.previous_token].revision))
        states, numeric, staged, records = {}, {}, dict(prior), []
        original = bank.state_evidence()
        for row in snapshot:
            parents = set(transfers.get(row.token, ()))
            if row.token in prior: parents.add((row.token, prior[row.token].revision))
            payload_hash = hashlib.sha256()
            for payload in sorted(row.payloads):
                payload_hash.update(struct.pack('<Q', len(payload))); payload_hash.update(payload)
            support_digest = payload_hash.hexdigest()
            if row.payloads:
                raw, quality, count = committed_reliability_inputs(snapshot, row.token)
                value = weighted_support(model, raw, quality, count)
                if not torch.isfinite(value.logits).all():
                    raise ValueError('Nonfinite learned memory logits, no corrupt state commit')
                numeric[row.token] = value
                vectors = raw.detach().double().numpy()
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                probabilities = value.logits.detach().sigmoid().numpy()
                weights = value.weights.detach().double().numpy()
                active = probabilities >= self._gates[0] if self.quarantine else np.ones(len(raw), dtype=bool)
                total = (vectors[active] * weights[active, None]).sum(axis=0)
                norm = float(np.linalg.norm(total))
                valid = bool(active.any() and norm > 1e-12)
                proto = total / norm if valid else None
                dispersion = float(np.clip(1 - norm / weights[active].sum(), 0, 1)) if active.any() else 1.
                lifecycle = ('Quarantined' if not valid else 'Confirmed'
                    if int(active.sum()) >= 2 and float(probabilities[active].mean()) >= self._gates[1]
                    and dispersion <= self._gates[2] else 'Tentative')
                # Tokens/bindings do not enter numeric digest/model inputs.
                numeric_bytes = (value.features.detach().numpy().astype('<f4').tobytes()
                    + value.logits.detach().numpy().astype('<f4').tobytes() + active.tobytes())
                states[row.token] = StateEvidence(row.token, int(active.sum()), original[row.token].routed_members,
                    total.astype('<f8').tobytes(), float(quality[torch.from_numpy(active)].mean()) if active.any() else 0.,
                    dispersion, 'DEFINED' if valid else 'EMPTY' if not active.any() else 'ZERO_RESULTANT')
                prototype_bytes = None if proto is None else proto.astype('<f8').tobytes()
                active_count = int(active.sum())
            else:
                numeric_bytes = b''; prototype_bytes = None; active_count = 0
                lifecycle = 'Quarantined'; states[row.token] = original[row.token]
            numeric_digest = hashlib.sha256(numeric_bytes).hexdigest()
            old = prior.get(row.token)
            changed = (old is None or old.support_sha256 != support_digest or old.numeric_sha256 != numeric_digest
                       or bool(transfers.get(row.token)))
            if changed:
                record = MemoryVersion(row.token, 0 if old is None else old.revision + 1,
                    tuple(sorted(parents)), support_digest, numeric_digest, lifecycle, len(row.payloads),
                    active_count, len(commits), prototype_bytes)
                staged[row.token] = record; records.append(record)
        if len(self._versions) + len(records) > self._max:
            raise ValueError('Version bound, no eviction or historical purity laundering')
        self._versions += tuple(records); self._current = staged; self._commit_count = len(commits)
        return states, numeric
