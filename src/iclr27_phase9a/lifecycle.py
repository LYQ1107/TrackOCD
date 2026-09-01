"""Causal semantic-state lifecycle used by TrackOCD Phase 9A.

The implementation deliberately keeps physical tracking and semantic memory
separate.  A physical track gets a public novel state immediately, but that
state is not eligible for cross-track reuse until a learned maturity head
marks it reusable.  No decision consults rows after the current timestamp.

The heads are small linear models trained on legal episodic supervision (see
``training/train_lifecycle.py``).  Keeping inference in numpy makes the
strict replay auditable and avoids accidentally sharing autograd state with
the online memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-8)


@dataclass
class TrackEvidence:
    """Causal trajectory evidence for one physical track."""

    mean: Optional[np.ndarray] = None
    count: int = 0
    uncertainty: float = 0.0
    consistency: float = 0.0

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
        z = unit(z)
        if self.mean is None:
            self.mean = z.copy()
            self.count = 1
            self.uncertainty = 0.0
            self.consistency = 1.0
            return self.mean.copy(), self.count, self.uncertainty, self.consistency

        old = self.mean
        old_count = self.count
        self.consistency = float(np.dot(old, z))
        delta = z - old
        # Welford-style scalar dispersion; the previous mean is the only
        # information needed and therefore remains strictly causal.
        self.uncertainty = (
            self.uncertainty * old_count + float(np.mean(delta * delta))
        ) / float(old_count + 1)
        self.mean = unit((old * old_count + z) / float(old_count + 1))
        self.count += 1
        return self.mean.copy(), self.count, self.uncertainty, self.consistency


@dataclass
class SemanticState:
    """Public semantic state and its causal lifecycle fields."""

    prototype: np.ndarray
    uncertainty: float
    consistency: float
    birth_known_score: float
    age: int
    evidence_count: float
    maturity_score: float
    provenance: str
    birth_track: Tuple[int, int]
    public_id: int
    reusable_flag: bool = False
    support_tracks: set = field(default_factory=set)
    quarantined: bool = False

    def update(self, h: np.ndarray, uncertainty: float, consistency: float,
               weight: float,
               track_key: Tuple[int, int]) -> None:
        weight = max(float(weight), 1.0)
        old = self.prototype
        self.prototype = unit((old * self.evidence_count + h * weight) /
                              (self.evidence_count + weight))
        self.evidence_count += weight
        self.age += 1
        self.uncertainty = float(uncertainty)
        self.consistency = float(consistency)
        self.support_tracks.add(track_key)

    def as_dict(self) -> dict:
        return {
            "prototype": self.prototype.astype(float).tolist(),
            "uncertainty": float(self.uncertainty),
            "consistency": float(self.consistency),
            "birth_known_score": float(self.birth_known_score),
            "age": int(self.age),
            "evidence_count": float(self.evidence_count),
            "maturity_score": float(self.maturity_score),
            "provenance": self.provenance,
            "birth_track": [int(self.birth_track[0]), int(self.birth_track[1])],
            "public_id": int(self.public_id),
            "reusable_flag": bool(self.reusable_flag),
            "support_tracks": len(self.support_tracks),
            "quarantined": bool(self.quarantined),
        }


class LifecycleHeads:
    """Binary semantic evidence, maturity and reuse heads.

    ``action`` row 0 is learned KNOWN evidence and row 1 is open-world
    evidence.  Both novel births and false births therefore receive an
    immediate public NEW_NOVEL identity; the state-only maturity head,
    trained with explicit false-birth negatives, controls later reuse.
    Pairwise reuse logits are compared to zero, never to a hand-set frame
    count.
    """

    def __init__(self, action_w: np.ndarray, action_b: np.ndarray,
                 maturity_w: np.ndarray, maturity_b: float,
                 reuse_w: np.ndarray, reuse_b: float):
        self.action_w = np.asarray(action_w, dtype=np.float32)
        self.action_b = np.asarray(action_b, dtype=np.float32)
        self.maturity_w = np.asarray(maturity_w, dtype=np.float32)
        self.maturity_b = float(maturity_b)
        self.reuse_w = np.asarray(reuse_w, dtype=np.float32)
        self.reuse_b = float(reuse_b)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-x))

    def action_logits(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) @ self.action_w.T + self.action_b

    def maturity_logit(self, x: np.ndarray) -> float:
        return float(np.asarray(x, dtype=np.float32) @ self.maturity_w + self.maturity_b)

    def reuse_logit(self, x: np.ndarray) -> float:
        return float(np.asarray(x, dtype=np.float32) @ self.reuse_w + self.reuse_b)

    def maturity_score(self, x: np.ndarray) -> float:
        return float(self._sigmoid(self.maturity_logit(x)))

    @classmethod
    def load(cls, path) -> "LifecycleHeads":
        d = np.load(path)
        return cls(d["action_w"], d["action_b"], d["maturity_w"],
                   float(d["maturity_b"]), d["reuse_w"], float(d["reuse_b"]))

    def save(self, path) -> None:
        np.savez(path, action_w=self.action_w, action_b=self.action_b,
                 maturity_w=self.maturity_w, maturity_b=self.maturity_b,
                 reuse_w=self.reuse_w, reuse_b=self.reuse_b)


def action_features(h: np.ndarray, known_proto: np.ndarray, score: float,
                    prior_hits: float, age: int, uncertainty: float,
                    consistency: float) -> Tuple[np.ndarray, np.ndarray]:
    sims = np.asarray(h, dtype=np.float32) @ np.asarray(known_proto, dtype=np.float32).T
    order = np.argsort(-sims)
    best, second = float(sims[order[0]]), float(sims[order[1]])
    # Physical score/prior are intentionally not identity logits.  They are
    # retained in the call signature for auditability, while semantic
    # knownness is driven by foundation similarity and causal trajectory
    # evidence only.
    x = np.asarray([best, second, best - second,
                    min(float(age), 50.0) / 50.0,
                    float(uncertainty), float(consistency)], dtype=np.float32)
    return x, sims


def maturity_features(state: SemanticState) -> np.ndarray:
    return np.asarray([
        np.log1p(max(float(state.age), 0.0)),
        np.log1p(max(float(state.evidence_count), 0.0)),
        float(state.uncertainty),
        float(state.consistency),
        1.0 - float(state.birth_known_score),
        min(float(len(state.support_tracks)), 20.0) / 20.0,
    ], dtype=np.float32)


def reuse_features(h: np.ndarray, state: SemanticState, query_age: int,
                   query_uncertainty: float, known_best: float,
                   known_margin: float) -> np.ndarray:
    return np.asarray([
        float(np.dot(h, state.prototype)),
        float(state.maturity_score),
        np.log1p(max(float(state.evidence_count), 0.0)),
        float(state.uncertainty),
        min(float(query_age), 50.0) / 50.0,
        float(query_uncertainty),
        float(known_best),
        float(known_margin),
    ], dtype=np.float32)


class CausalLifecycle:
    """Strict online semantic lifecycle over a frozen feature stream."""

    def __init__(self, known_prototypes: np.ndarray, known_ids: Iterable[int],
                 heads: LifecycleHeads, max_states: int = 512,
                 decision_prototypes: np.ndarray | None = None,
                 decision_ids: Iterable[int] | None = None,
                 no_lifecycle: bool = False, fixed_maturity: int | None = None,
                 no_false_birth: bool = False, trajectory: bool = True):
        self.known_prototypes = np.asarray(known_prototypes, dtype=np.float32)
        self.known_ids = [int(x) for x in known_ids]
        self.decision_prototypes = np.asarray(
            self.known_prototypes if decision_prototypes is None else decision_prototypes,
            dtype=np.float32)
        self.decision_ids = [int(x) for x in (
            self.known_ids if decision_ids is None else decision_ids)]
        self.heads = heads
        self.max_states = int(max_states)
        self.no_lifecycle = bool(no_lifecycle)
        self.fixed_maturity = fixed_maturity
        self.no_false_birth = bool(no_false_birth)
        self.trajectory = bool(trajectory)
        self.tracks: Dict[Tuple[int, int], TrackEvidence] = {}
        self.track_slots: Dict[Tuple[int, int], int] = {}
        self.states: list[SemanticState] = []
        self.quarantine: Dict[Tuple[int, int], SemanticState] = {}
        self.next_quarantine_id = 200000
        self.events = []

    def _state_maturity(self, state: SemanticState) -> None:
        if self.fixed_maturity is not None:
            state.maturity_score = min(
                float(state.evidence_count) / max(float(self.fixed_maturity), 1.0),
                1.0)
            state.reusable_flag = bool(
                (not state.quarantined) and
                state.evidence_count >= float(self.fixed_maturity))
            return
        logit = self.heads.maturity_logit(maturity_features(state))
        state.maturity_score = self.heads._sigmoid(logit)
        state.reusable_flag = bool(
            (not state.quarantined) and (self.no_lifecycle or logit > 0.0))

    def _spawn(self, h: np.ndarray, uncertainty: float, key,
               quarantined: bool, known_score: float) -> SemanticState:
        if quarantined or len(self.states) >= self.max_states:
            sid = self.next_quarantine_id
            self.next_quarantine_id += 1
            state = SemanticState(
                prototype=h.copy(), uncertainty=float(uncertainty), consistency=1.0,
                birth_known_score=float(known_score), age=1,
                evidence_count=1.0, maturity_score=0.0,
                provenance="causal_birth_quarantine", birth_track=key,
                public_id=sid, quarantined=True)
            self.quarantine[key] = state
            return state
        slot = len(self.states)
        state = SemanticState(
            prototype=h.copy(), uncertainty=float(uncertainty), consistency=1.0,
            birth_known_score=float(known_score), age=1,
            evidence_count=1.0, maturity_score=0.0,
            provenance="causal_birth_candidate", birth_track=key,
            public_id=100000 + slot)
        self.states.append(state)
        self._state_maturity(state)
        self.track_slots[key] = slot
        return state

    def _update_state(self, state: SemanticState, h: np.ndarray,
                      uncertainty: float, consistency: float, key) -> None:
        state.update(h, uncertainty, consistency, weight=1.0, track_key=key)
        self._state_maturity(state)

    def _maybe_promote_quarantine(self, key, state: SemanticState) -> bool:
        """Promote a noisy birth only after learned causal evidence matures."""
        if not state.quarantined or len(self.states) >= self.max_states:
            return False
        logit = self.heads.maturity_logit(maturity_features(state))
        if logit <= 0.0:
            return False
        state.quarantined = False
        state.provenance = "causal_birth_promoted"
        state.reusable_flag = True
        state.maturity_score = self.heads._sigmoid(logit)
        self.track_slots[key] = len(self.states)
        self.states.append(state)
        self.quarantine.pop(key, None)
        return True

    def step(self, z: np.ndarray, key: Tuple[int, int], score: float = 0.0,
             prior_hits: float = 0.0) -> dict:
        """Consume exactly one row and return its immediate semantic action."""
        key = (int(key[0]), int(key[1]))
        evidence = self.tracks.setdefault(key, TrackEvidence())
        if self.trajectory:
            h, age, uncertainty, consistency = evidence.update(z)
        else:
            h, age, uncertainty, consistency = unit(z), 1, 0.0, 1.0
        af, decision_sims = action_features(h, self.decision_prototypes, score,
                                   prior_hits, age, uncertainty, consistency)
        al = self.heads.action_logits(af)
        output_sims = h @ self.known_prototypes.T
        known_best = float(np.max(decision_sims))
        order = np.argsort(-decision_sims)
        known_margin = float(decision_sims[order[0]] - decision_sims[order[1]])
        cls = int(np.argmax(al))
        # A physical track that already owns a semantic state may keep
        # updating that state, even while it is still a candidate.  This is
        # the birth-immediate / reuse-delayed contract.
        if key in self.track_slots:
            slot = self.track_slots[key]
            state = self.states[slot]
            # A candidate born from a noisy proposal must not permanently
            # swallow a known object.  Let the learned known-vs-novel action
            # win when its semantic prototype is stronger than the candidate
            # prototype; a genuine novel track normally has the opposite
            # causal evidence ordering.
            if cls == 0 and known_best > float(np.dot(h, state.prototype)):
                sid = self.known_ids[int(np.argmax(output_sims))]
                out = {"action": "known", "semantic_id": int(sid),
                       "slot": None, "age": age, "known_score": known_best,
                       "maturity_score": 1.0, "reusable": False,
                       "physical_key": key}
                self.events.append(out)
                return out
            self._update_state(state, h, uncertainty, consistency, key)
            out = self._out("existing", state, key, age, known_best)
            self.events.append(out)
            return out
        if key in self.quarantine:
            state = self.quarantine[key]
            # A low-quality false birth may later be explained as a known
            # object.  Allow the learned known action to correct that local
            # contamination; candidate novel states remain track-owned.
            if cls == 0:
                sid = self.known_ids[int(np.argmax(output_sims))]
                out = {"action": "known", "semantic_id": int(sid),
                       "slot": None, "age": age, "known_score": known_best,
                       "maturity_score": 1.0, "reusable": False,
                       "physical_key": key}
                self.events.append(out)
                return out
            state.update(h, uncertainty, consistency, 1.0, key)
            self._maybe_promote_quarantine(key, state)
            out = self._out("existing", state, key, age, known_best)
            self.events.append(out)
            return out

        if cls == 0:
            sid = self.known_ids[int(np.argmax(output_sims))]
            out = {"action": "known", "semantic_id": int(sid),
                   "slot": None, "age": age, "known_score": known_best,
                   "maturity_score": 1.0, "reusable": False,
                   "physical_key": key}
            self.events.append(out)
            return out

        # For a novel-looking row, a trusted candidate competes with a new
        # birth using a learned pair score.  Untrusted candidates are masked
        # and therefore cannot contaminate another physical track.
        best_slot, best_logit = None, -np.inf
        for slot, state in enumerate(self.states):
            if not state.reusable_flag:
                continue
            rf = reuse_features(h, state, age, uncertainty, known_best,
                                known_margin)
            logit = self.heads.reuse_logit(rf)
            if logit > best_logit:
                best_slot, best_logit = slot, logit
        if best_slot is not None and best_logit > 0.0 and cls != 0:
            state = self.states[best_slot]
            self.track_slots[key] = best_slot
            self._update_state(state, h, uncertainty, consistency, key)
            out = self._out("existing", state, key, age, known_best)
            self.events.append(out)
            return out

        # Both true novel births and false births are public immediately;
        # false-looking births are quarantined and never enter reusable
        # semantic memory.
        state = self._spawn(h, uncertainty, key, quarantined=(cls != 1),
                            known_score=known_best)
        out = self._out("new", state, key, age, known_best)
        self.events.append(out)
        return out

    @staticmethod
    def _out(action: str, state: SemanticState, key, age: int,
             known_score: float) -> dict:
        return {
            "action": action,
            "semantic_id": int(state.public_id),
            "slot": (None if state.quarantined else int(state.public_id - 100000)),
            "age": int(age),
            "known_score": float(known_score),
            "maturity_score": float(state.maturity_score),
            "reusable": bool(state.reusable_flag),
            "physical_key": key,
        }

    def state_snapshot(self) -> list[dict]:
        return [s.as_dict() for s in self.states] + [s.as_dict() for s in self.quarantine.values()]
