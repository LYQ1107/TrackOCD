"""Stage-1 reliability-gated replay for the frozen Phase-8A B model.

This file intentionally keeps Architecture B's representation, adapter,
create head, known centroids, score temperature, and state update unchanged.
The only intervention is a causal mask over *online-born novel states*: a
state is public immediately when B chooses ``new``, but a different physical
track cannot attach to it until its reliability posterior supports reuse.

The gate is deliberately small and pre-registered.  Each online-born state
keeps a Beta posterior over a latent ``reliable`` probability.  On each
accepted owner/attached-track observation, the posterior receives one soft
observation made from (i) trajectory consistency, (ii) feature stability,
and (iii) normalized score certainty.  The posterior probability
``P(reliable > 0.5)`` is the reliability score; reuse is allowed only when
that probability is greater than 0.5.  The neutral Beta(2, 2) prior means a
fresh state is a candidate, without imposing a fixed frame-count rule.

All decisions are chronological and use only the current row and state
statistics accumulated before/at that row.  GT is used only by the separate
post-replay contamination analysis, never by this replay.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import betainc

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.iclr27_phase8a.model.adapter import (  # noqa: E402
    CausalTrajectoryAdapter,
    TorchSemanticStateSet,
)
from src.iclr27_phase8a.model.create_head import CreateHead  # noqa: E402
from src.iclr27_phase8a.training.train_amortized import phys_vec  # noqa: E402
from src.iclr27_phase8a.training.train_bsp import (  # noqa: E402
    compute_centroids,
    load_assets,
)
from src.iclr27_phase7a.training.train_reliability_head import (  # noqa: E402
    load_tse,
    project,
)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


@dataclass
class TrackEvidence:
    """Prefix-causal trajectory statistics for one physical track."""

    prev: np.ndarray | None = None
    mean: np.ndarray | None = None
    m2: float = 0.0
    count: int = 0
    last_consistency: float = 1.0
    feature_variance: float = 0.0

    def update(self, h: np.ndarray) -> tuple[float, float, int]:
        h = _unit(h)
        if self.prev is None:
            self.prev = h.copy()
            self.mean = h.copy()
            self.count = 1
            self.last_consistency = 1.0
            self.feature_variance = 0.0
            return self.last_consistency, self.feature_variance, self.count

        # Cosine consistency is converted to [0, 1] before entering the
        # reliability posterior.  No future row is consulted.
        self.last_consistency = float(np.clip(
            (float(np.dot(self.prev, h)) + 1.0) * 0.5, 0.0, 1.0))
        assert self.mean is not None
        old_mean = self.mean.copy()
        old_count = self.count
        new_count = old_count + 1
        delta = h - old_mean
        self.mean = old_mean + delta / float(new_count)
        # Scalar Welford dispersion, averaged over embedding dimensions.
        self.m2 += float(np.mean(delta * (h - self.mean)))
        self.count = new_count
        self.feature_variance = max(self.m2 / max(new_count - 1, 1), 0.0)
        self.prev = h.copy()
        return self.last_consistency, self.feature_variance, self.count


@dataclass
class ReliabilityState:
    """Causal lifecycle fields attached to one online-born semantic state."""

    birth_track: tuple[int, int]
    prototype: np.ndarray
    age: int = 1
    evidence_count: float = 1.0
    trajectory_consistency: float = 1.0
    feature_variance: float = 0.0
    uncertainty: float = 0.5
    alpha: float = 2.0
    beta: float = 2.0
    support_tracks: set[tuple[int, int]] = field(default_factory=set)
    reliability_score: float = 0.5
    reuse_allowed: bool = False

    def _refresh(self) -> None:
        # P(theta > .5 | Beta(alpha, beta)); the 0.5 decision is the neutral
        # posterior-majority rule, not a frame-count or Q1-tuned threshold.
        self.reliability_score = float(np.clip(
            1.0 - betainc(self.alpha, self.beta, 0.5), 0.0, 1.0))
        self.reuse_allowed = bool(self.reliability_score > 0.5)

    def update(self, h: np.ndarray, consistency: float, variance: float,
               certainty: float, state_count: float,
               track_key: tuple[int, int]) -> float:
        """Consume one accepted state observation and return soft evidence."""
        h = _unit(h)
        old = self.prototype
        self.age += 1
        self.evidence_count = float(state_count)
        self.trajectory_consistency = float(consistency)
        self.feature_variance = float(max(variance, 0.0))
        self.uncertainty = float(np.clip(1.0 - certainty, 0.0, 1.0))
        # Dimension-free stability: RMS dispersion of a unit embedding is
        # converted to a [0, 1] temporal-stability signal.
        stability = float(np.exp(-math.sqrt(
            max(self.feature_variance, 0.0) * max(h.size, 1))))
        q = float(np.clip(
            max(consistency, 0.0) * max(stability, 0.0)
            * max(certainty, 0.0), 0.0, 1.0) ** (1.0 / 3.0))
        self.alpha += q
        self.beta += 1.0 - q
        # Keep the same online prototype that B uses for matching, while the
        # lifecycle copy makes the evidence auditable without gradients.
        w = max(float(state_count), 1.0)
        self.prototype = _unit((old * max(w - 1.0, 0.0) + h) / w)
        self.support_tracks.add(track_key)
        self._refresh()
        return q

    def as_dict(self, slot: int) -> dict:
        return {
            "slot": int(slot),
            "birth_track": [int(self.birth_track[0]), int(self.birth_track[1])],
            "prototype_norm": float(np.linalg.norm(self.prototype)),
            "uncertainty": float(self.uncertainty),
            "age": int(self.age),
            "evidence_count": float(self.evidence_count),
            "trajectory_consistency": float(self.trajectory_consistency),
            "feature_variance": float(self.feature_variance),
            "reliability_score": float(self.reliability_score),
            "reuse_allowed": bool(self.reuse_allowed),
            "posterior_alpha": float(self.alpha),
            "posterior_beta": float(self.beta),
            "support_track_count": int(len(self.support_tracks)),
        }


def load_rows(path: str) -> tuple[list[dict], list[str]]:
    with open(ROOT / path, newline="") as f:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames or [])
        return [dict(r) for r in reader], names


def _chrono(rows: list[dict]) -> list[int]:
    return sorted(
        range(len(rows)),
        key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
                       int(rows[i].get("proposal_local_id") or 0),
                       int(rows[i]["track_id"])),
    )


def _entropy_certainty(logits: torch.Tensor) -> float:
    if logits.numel() <= 1:
        return 1.0
    p = torch.softmax(logits.detach().float(), dim=0)
    entropy = float(-(p * torch.log(torch.clamp(p, min=1e-12))).sum())
    denom = math.log(float(logits.numel()))
    if denom <= 0.0:
        return 1.0
    return float(np.clip(1.0 - entropy / denom, 0.0, 1.0))


def _append_unique(names: list[str], extra: list[str]) -> list[str]:
    out = list(names)
    for name in extra:
        if name not in out:
            out.append(name)
    return out


def replay(args) -> dict:
    dev = torch.device(args.device)
    ck = torch.load(ROOT / args.adapter_ckpt, map_location=dev,
                    weights_only=False)
    ck_args = ck.get("args", {})
    dim = int(ck_args.get("dim", 128))
    temp = float(ck.get("temp", ck_args.get("temp", 20.0)))
    frame_level = bool(ck_args.get("frame_level", False))

    adapter = CausalTrajectoryAdapter(
        dim=dim, rho_init=0.0, sigma2=1.0,
        frame_level=frame_level).to(dev)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    create_head = CreateHead(dim=dim).to(dev)
    create_head.load_state_dict(ck["create_head"])
    create_head.eval()

    tse, _, _ = load_tse(dev)
    _, ep, _ = load_assets("hard")
    z_anchor = project(dev, tse, ep["feats"].astype(np.float32))
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_anchor, dev)

    rows, fieldnames = load_rows(args.proposals)
    feats = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    if len(rows) != len(feats):
        raise ValueError(f"proposal/feature length mismatch: {len(rows)} vs {len(feats)}")
    z_all = project(dev, tse, feats)

    states = TorchSemanticStateSet(
        dim=dim, max_slots=4096, sigma2=1.0,
        score_mode="cosine", cosine_temp=temp).to(dev)
    states.init_known(mu.detach(), cnt.detach())
    known_n = int(states.n)

    track_state = {}
    track_count: dict[tuple[int, int], int] = {}
    track_evidence: dict[tuple[int, int], TrackEvidence] = {}
    birth_track: dict[int, tuple[int, int]] = {}
    reliability: dict[int, ReliabilityState] = {}
    # ``None`` entries correspond to the frozen known centroid slots.
    rel_by_slot: list[ReliabilityState | None] = [None] * known_n

    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    sem_kscore = [""] * len(rows)
    sem_slot = [""] * len(rows)
    sem_rel = [""] * len(rows)
    sem_allowed = [""] * len(rows)
    sem_blocked = [""] * len(rows)
    sem_original_slot = [""] * len(rows)
    events: list[dict] = []
    counts = Counter()

    chrono = _chrono(rows)
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            z = torch.from_numpy(z_all[i]).to(dev).unsqueeze(0)
            h, state = adapter(z, prev)
            h = h[0]
            h_np = h.detach().cpu().numpy().astype(np.float32)
            age = int(track_count.get(key, 0) + 1)
            track_count[key] = age
            track_state[key] = state.detach()
            ev = track_evidence.setdefault(key, TrackEvidence())
            consistency, track_var, _ = ev.update(h_np)
            w = 1.0 if frame_level else float(age)

            scores = states.log_scores(h, w)
            best_sim = scores.max() if states.n else torch.zeros((), device=dev)
            phys = phys_vec(rows[i].get("score", 0.0),
                            rows[i].get("prior_hits", 0.0), age, dev)
            create_logit = temp * create_head(h, phys, best_sim)
            original_logits = states.logits(h, w, create_logit.reshape(1))
            original_pred = int(torch.argmax(original_logits))
            original_slot = (None if original_pred == states.n
                             else original_pred)
            p_assign = (1.0 / (1.0 + torch.exp(
                create_logit - torch.logsumexp(scores, dim=0)))
                        if states.n else torch.zeros((), device=dev))
            certainty = _entropy_certainty(original_logits)

            # Mask only online-born, not-yet-reusable states for a different
            # physical track.  The birth owner may always continue updating
            # its candidate; this is the causal candidate->trusted lifecycle.
            masked_scores = scores.clone()
            blocked: list[int] = []
            for slot in range(known_n, states.n):
                rel = rel_by_slot[slot]
                owner = birth_track.get(slot)
                if (rel is not None and owner is not None
                        and key != owner and not rel.reuse_allowed):
                    masked_scores[slot] = -torch.inf
                    blocked.append(slot)
            masked_logits = torch.cat([masked_scores,
                                       create_logit.reshape(1)])
            pred = int(torch.argmax(masked_logits))
            gate_blocked = (original_pred < states.n
                            and original_pred in blocked)
            if gate_blocked:
                counts["blocked_untrusted_assignment"] += 1

            # Emit the selected B-compatible public action, updating B's
            # state process exactly once.  A birth is immediate even though
            # the newly created state is not yet reusable by another track.
            birth = False
            selected_slot: int | None
            if pred == states.n:
                selected_slot = states.spawn(h, w)
                if selected_slot is None:
                    # Match B's full-memory fallback, although this cannot be
                    # reached with the Q1-sized stream and 4096 slots.
                    selected_slot = int(torch.argmax(masked_scores))
                    pred = selected_slot
                else:
                    birth = True
                    birth_track[selected_slot] = key
                    rel = ReliabilityState(
                        birth_track=key, prototype=h_np.copy())
                    rel._refresh()
                    reliability[selected_slot] = rel
                    rel_by_slot.append(rel)
            else:
                selected_slot = pred

            if selected_slot is None:
                raise RuntimeError("semantic state selection returned no slot")
            if not birth:
                prov = int(states.provenance[selected_slot])
                states.assign(selected_slot, h, w)
                if prov == 1:
                    rel = rel_by_slot[selected_slot]
                    if rel is None:
                        raise RuntimeError(f"missing reliability state for slot {selected_slot}")
                    # The evidence posterior is updated only after the gate
                    # has accepted this assignment.  Thus a blocked track
                    # cannot make the state look reliable.
                    rel.update(h_np, consistency, track_var, certainty,
                               float(states.count[selected_slot]), key)
                    reliability[selected_slot] = rel

            prov = int(states.provenance[selected_slot])
            if prov == 0:
                action = "known"
                sid = str(known_ids[selected_slot])
            elif birth:
                action = "new"
                sid = str(100000 + selected_slot)
            else:
                action = "existing"
                sid = str(100000 + selected_slot)

            rel = rel_by_slot[selected_slot] if prov == 1 else None
            sem_action[i] = action
            sem_sid[i] = sid
            sem_kscore[i] = f"{float(p_assign):.6f}"
            sem_slot[i] = str(selected_slot)
            sem_rel[i] = (f"{rel.reliability_score:.6f}" if rel else "")
            sem_allowed[i] = (str(int(rel.reuse_allowed)) if rel else "")
            sem_blocked[i] = ";".join(str(x) for x in blocked)
            sem_original_slot[i] = ("new" if original_slot is None
                                    else str(original_slot))
            counts[action] += 1
            if birth:
                counts["candidate_birth"] += 1
            if gate_blocked:
                counts["blocked_to_new"] += int(birth)

            events.append({
                "row_index": int(i),
                "video_id": int(key[0]),
                "track_id": int(key[1]),
                "frame_id": int(rows[i]["frame_id"]),
                "key": [int(key[0]), int(key[1])],
                "original_action": ("new" if original_slot is None else
                                     ("known" if original_slot < known_n
                                      else "existing")),
                "original_slot": (None if original_slot is None
                                  else int(original_slot)),
                "selected_action": action,
                "selected_slot": int(selected_slot),
                "selected_sid": int(sid),
                "gate_blocked": bool(gate_blocked),
                "blocked_slots": [int(x) for x in blocked],
                "birth": bool(birth),
                "trajectory_consistency": float(consistency),
                "track_feature_variance": float(track_var),
                "score_certainty": float(certainty),
                "reliability": (rel.as_dict(selected_slot) if rel else None),
            })

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["sem_kscore", "sem_slot", "sem_reliability",
             "sem_reuse_allowed", "sem_gate_blocked", "sem_original_slot"]
    names = _append_unique(fieldnames, extra)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out["sem_action"] = sem_action[i]
            out["sem_sid"] = sem_sid[i]
            out["sem_kscore"] = sem_kscore[i]
            out["sem_slot"] = sem_slot[i]
            out["sem_reliability"] = sem_rel[i]
            out["sem_reuse_allowed"] = sem_allowed[i]
            out["sem_gate_blocked"] = sem_blocked[i]
            out["sem_original_slot"] = sem_original_slot[i]
            writer.writerow(out)

    events_path = ROOT / args.out_events
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with open(events_path, "w") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    states_out = {
        "known_state_count": known_n,
        "final_state_count": int(states.n),
        "candidate_birth_count": int(counts["candidate_birth"]),
        "blocked_untrusted_assignment": int(
            counts["blocked_untrusted_assignment"]),
        "blocked_to_new": int(counts["blocked_to_new"]),
        "actions": dict(Counter(sem_action)),
        "final_online_states": [
            rel.as_dict(slot) for slot, rel in sorted(reliability.items())
        ],
        "posterior_prior": {"alpha": 2.0, "beta": 2.0},
        "reuse_rule": "P(reliable > 0.5 | Beta posterior) > 0.5",
        "reliability_observation": (
            "geometric mean of trajectory consistency, dimension-free "
            "feature stability, and normalized B-logit certainty"),
        "device": str(dev),
        "checkpoint": args.adapter_ckpt,
        "proposals": args.proposals,
        "features": args.feats,
    }
    summary_path = ROOT / args.out_summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(states_out, indent=2, default=float))
    return states_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--adapter-ckpt", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-events", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    result = replay(args)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("final_online_states",)}, indent=2))
    print("wrote", ROOT / args.out_csv)
    print("wrote", ROOT / args.out_events)
    print("wrote", ROOT / args.out_summary)


if __name__ == "__main__":
    main()
