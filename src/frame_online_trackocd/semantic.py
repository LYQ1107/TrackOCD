"""Frame-online semantic observation and causal semantic memory.

Every detection is observed with:
  - frozen DINOv2 single-frame feature (768-d),
  - frozen Phase 4F M2 aggregator + gate,
  - frozen train-known prototypes,
  - causal novel semantic memory M_{t-1} (built only from frames < t).

Semantic observations precede physical association (B2) or follow it
(B1); in both cases memory is updated only after association, and all
detections of one frame share the same M_{t-1} (frame-synchronous).

Physical identity and semantic identity are kept separate: track semantic
state is keyed by physical track id but a novel semantic ID can be shared
by different physical tracks.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit.evaluate import embed_track
from src.orbit_msr.protocol import known_stats


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


class NovelSemanticMemory:
    """Causal novel semantic memory: virtual semantic categories."""

    def __init__(self, novel_update_rate=0.2, min_birth_sim=0.6,
                 matching_mode="absolute", margin_threshold=0.05,
                 entropy_threshold=1.6):
        self.novel_update_rate = novel_update_rate
        self.min_birth_sim = min_birth_sim
        self.matching_mode = matching_mode
        self.margin_threshold = float(margin_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.protos = {}          # sem_id -> normalized z
        self.support = defaultdict(int)
        self.reliability = {}
        self.prototype_weight = {}
        self.next_id = 1000000    # virtual semantic IDs (never physical IDs)
        # causal commit log (frame/track -> global memory write)
        self.commit_events = []
        self.rel_calls = 0
        self.rel_rejects = 0

    def best(self, z):
        if not self.protos:
            return -1.0, None
        P = np.stack(list(self.protos.values())).astype(np.float32)
        ns = P @ z
        order = np.argsort(ns)[::-1]
        return float(ns[order[0]]), int(list(self.protos)[order[0]])

    def relative_ok(self, z, best_cos, gid):
        """Relative EXISTING_NOVEL criterion (Phase 4L, causal-only)."""
        self.rel_calls += 1
        if gid is None or best_cos < self.min_birth_sim:
            self.rel_rejects += 1
            return False
        if self.matching_mode == "absolute":
            return True
        if not self.protos:
            self.rel_rejects += 1
            return False
        P = np.stack(list(self.protos.values())).astype(np.float32)
        cos = P @ z
        order = np.argsort(cos)[::-1]
        if self.matching_mode == "margin":
            second = float(cos[order[1]]) if len(order) >= 2 else -1.0
            ok = (best_cos - second) >= self.margin_threshold
            if not ok:
                self.rel_rejects += 1
            return ok
        if self.matching_mode == "entropy":
            top5 = [float(cos[order[k]]) if k < len(order) else -1.0
                    for k in range(5)]
            soft = np.exp(np.clip(top5, -30, 30))
            soft = soft / soft.sum()
            ent = float(-(soft * np.log(soft + 1e-12)).sum())
            ok = ent <= self.entropy_threshold
            if not ok:
                self.rel_rejects += 1
            return ok
        raise ValueError("unknown matching_mode %s" % self.matching_mode)

    def propose(self, z, rel, weight=1.0):
        """Return (sem_id, is_new): reuse existing novel category if close
        enough, otherwise create one.  Causal: only current memory used."""
        best, sem_id = self.best(z)
        if sem_id is not None and self.relative_ok(z, best, sem_id):
            return sem_id, False
        sem_id = self.next_id
        self.next_id += 1
        self.protos[sem_id] = _norm(z.astype(np.float32))
        self.support[sem_id] = 1
        self.reliability[sem_id] = float(rel)
        self.prototype_weight[sem_id] = float(weight)
        return sem_id, True

    def update(self, sem_id, z, rel, weight=1.0):
        if sem_id not in self.protos:
            self.protos[sem_id] = _norm(z.astype(np.float32))
            self.support[sem_id] = 1
            self.reliability[sem_id] = float(rel)
            self.prototype_weight[sem_id] = float(weight)
            return
        p = self.protos[sem_id]
        w = max(0.0, min(1.0, float(weight)))
        p = (1.0 - self.novel_update_rate * w) * p + \
            self.novel_update_rate * w * z
        self.protos[sem_id] = _norm(p.astype(np.float32))
        self.support[sem_id] += 1
        r = self.reliability.get(sem_id, 0.0)
        self.reliability[sem_id] = 0.8 * r + 0.2 * float(rel)
        old_w = self.prototype_weight.get(sem_id, 1.0)
        self.prototype_weight[sem_id] = 0.8 * old_w + 0.2 * w

    def size(self):
        return len(self.protos)


class TrackSemState:
    """Causal per-physical-track semantic state."""

    def __init__(self, track_id, obs, prefix_mode="P1"):
        self.track_id = track_id
        self.prefix_mode = prefix_mode
        self.age = 1
        self.last_frame = -1
        self.known_belief = float(obs["p_known"])
        self.class_dist = obs["class_dist"].copy()
        self.novel_id = obs.get("novel_id")
        self.novel_conf = obs.get("novel_conf", 0.0)
        self.prefix_z = obs["z"].copy()
        self.wsum = 1.0
        self.wsum_z = obs["z"].copy()
        self.reliability = obs.get("reliability", 1.0)
        self.z_history = [obs["z"].copy()]
        # Phase 4L: causal admissibility evidence (online detector/tracker
        # outputs only; never GT).
        self.score_ema = 0.0
        self.mask_frac_ema = 0.0
        self.app_norm_ema = 0.0
        self.assoc_ap_ema = 0.0
        # Phase 4J: observation / commitment split.
        self.candidate_id = None       # track-local provisional novel id
        self.committed_sem_id = None   # global novel id once committed
        self.commit_frame = -1
        self.commit_age = None
        self.commit_action = None      # NEW_NOVEL / EXISTING_NOVEL
        self.novel_support = 0         # novel-like observations on this track
        self.last_action = "PROVISIONAL_NOVEL"
        self.observed_novel_id = None
        self.observed_novel_conf = 0.0
        # Phase 4M: semantic identity resolution state.  UNRESOLVED means
        # the track keeps its local soft semantics but has not written an
        # irreversible EXISTING_NOVEL / NEW_NOVEL identity to global memory.
        self.resolution_state = "resolved"
        self.unresolved_frames = 0

    def mark_committed(self, sem_id, action, frame_id):
        self.committed_sem_id = sem_id
        self.candidate_id = None
        self.commit_frame = int(frame_id)
        self.commit_age = self.age
        self.commit_action = action
        self.last_action = action

    def update(self, obs, alpha=0.5, aux=None):
        self.age += 1
        self.known_belief = (1 - alpha) * self.known_belief + alpha * obs["p_known"]
        self.class_dist = (1 - alpha) * self.class_dist + alpha * obs["class_dist"]
        self.class_dist = self.class_dist / (self.class_dist.sum() + 1e-12)
        # Phase 4L fix: the observed (argmax) novel id is current-frame
        # soft evidence only; it must NOT silently become the track's
        # committed identity, otherwise relative matching is bypassed.
        self.observed_novel_id = obs.get("novel_id")
        self.observed_novel_conf = obs.get("novel_conf", 0.0)
        self.z_history.append(obs["z"].copy())
        if len(self.z_history) > 8:
            self.z_history.pop(0)
        if aux:
            self.score_ema = 0.8 * self.score_ema + 0.2 * float(
                aux.get("det_score", 0.0))
            self.mask_frac_ema = 0.8 * self.mask_frac_ema + 0.2 * float(
                aux.get("mask_area_frac", 0.0))
            self.app_norm_ema = 0.8 * self.app_norm_ema + 0.2 * float(
                aux.get("appearance_norm", 0.0))
            self.assoc_ap_ema = 0.8 * self.assoc_ap_ema + 0.2 * float(
                aux.get("assoc_ap_score", 0.0))

    def prefix(self):
        """Causal prefix embedding Agg(f^1..f^t)."""
        if self.prefix_mode == "P0":
            return self.z_history[-1]
        if self.prefix_mode == "P2":
            return _norm((self.wsum_z / max(self.wsum, 1e-9)).astype(np.float32))
        # P1: causal running mean
        return _norm(np.mean(self.z_history, axis=0).astype(np.float32))

    def add_weighted(self, z, rel, score):
        if self.prefix_mode == "P2":
            w = float(score) * min(1.0, self.age / 5.0)
            self.wsum += w
            self.wsum_z = self.wsum_z + w * z


class SemanticStateManager:
    """Frame-synchronous semantic observation + causal memory manager."""

    def __init__(self, model, known_protos, radii, device,
                 prefix_mode="P1", theta_novel=0.6, class_temp=0.05,
                 belief_alpha=0.5, novel_update_rate=0.2,
                 memo_tracklet_frames=10, rel_default=1.0,
                 decision_threshold=0.5, decision_split_age=None,
                 commit_mode="M0", commit_min_age=2,
                 commit_min_support=2, provenance=None,
                 admissibility_mode="none", admissibility_config=None,
                 matching_mode="absolute", margin_threshold=0.05,
                 entropy_threshold=1.6, deferral_mode="none",
                 defer_margin=0.10, defer_entropy=1.6, defer_nk=0.25,
                 defer_ambiguity_coef=None,
                 defer_ambiguity_intercept=0.0,
                 defer_ambiguity_threshold=0.5,
                 validity_mode="none", validity_config=None,
                 validity_threshold=0.03):
        self.model = model
        self.known_protos = known_protos
        self.radii = radii
        self.device = device
        self.P_known = np.stack([known_protos[c] for c in sorted(known_protos)]
                                ).astype(np.float32)
        self.known_ids = sorted(known_protos)
        self.prefix_mode = prefix_mode
        self.theta_novel = theta_novel
        self.class_temp = class_temp
        self.belief_alpha = belief_alpha
        self.novel_update_rate = novel_update_rate
        self.memo_tracklet_frames = memo_tracklet_frames
        self.rel_default = rel_default
        # Phase 4J calibration / commitment configuration.
        if isinstance(decision_threshold, (tuple, list)):
            assert len(decision_threshold) == 2, \
                "two-band threshold needs (early, stable)"
            self.thr_early, self.thr_stable = map(float, decision_threshold)
        else:
            self.thr_early = self.thr_stable = float(decision_threshold)
        self.decision_split_age = decision_split_age
        self.commit_mode = commit_mode
        self.commit_min_age = int(commit_min_age)
        self.commit_min_support = int(commit_min_support)
        self.provenance = provenance
        self.admissibility_mode = admissibility_mode
        self.admissibility_config = admissibility_config or {}
        self.deferral_mode = deferral_mode
        self.defer_margin = float(defer_margin)
        self.defer_entropy = float(defer_entropy)
        self.defer_nk = float(defer_nk)
        self.defer_ambiguity_coef = (
            np.asarray(defer_ambiguity_coef, dtype=np.float32)
            if defer_ambiguity_coef is not None else None)
        self.defer_ambiguity_intercept = float(defer_ambiguity_intercept)
        self.defer_ambiguity_threshold = float(defer_ambiguity_threshold)
        self.validity_mode = validity_mode
        self.validity_config = validity_config or {}
        self.validity_threshold = float(validity_threshold)
        self.video_id = None
        self.current_frame = -1
        self.memory = NovelSemanticMemory(
            novel_update_rate, min_birth_sim=theta_novel,
            matching_mode=matching_mode, margin_threshold=margin_threshold,
            entropy_threshold=entropy_threshold)
        self.tracks = {}          # physical track id -> TrackSemState
        self._z_cache = {}
        self.branch_sticky = 0
        self.branch_soft = 0
        self.branch_new = 0
        self.branch_deferred = 0
        self._debug_branch = False
        self._last_video_id = None

    def admissibility(self, t):
        """Soft semantic-admissibility weight in [0,1] (Phase 4L)."""
        mode = self.admissibility_mode
        if mode == "none":
            return 1.0
        cfg = self.admissibility_config
        feats = cfg.get("features", [])
        vals = []
        for f in feats:
            if f == "det_score":
                vals.append(t.score_ema)
            elif f == "mask_area_frac":
                vals.append(t.mask_frac_ema)
            elif f == "appearance_norm":
                vals.append(t.app_norm_ema)
            elif f == "assoc_ap_score":
                vals.append(t.assoc_ap_ema)
            elif f == "track_age":
                vals.append(float(t.age))
            else:
                vals.append(0.0)
        if not vals:
            return 1.0
        mean = np.asarray(cfg.get("mean", [0.0] * len(vals)),
                          dtype=np.float32)
        scale = np.asarray(cfg.get("scale", [1.0] * len(vals)),
                           dtype=np.float32)
        coef = np.asarray(cfg.get("coef", [0.0] * len(vals)),
                          dtype=np.float32)
        x = (np.asarray(vals, dtype=np.float32) - mean) / \
            np.maximum(scale, 1e-6)
        logit = float(coef @ x + float(cfg.get("intercept", 0.0)))
        return float(1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0))))

    def validity(self, t, score=None, aux=None):
        """Object-validity score v_t in [0,1] (Phase 4N).

        Causal frontend evidence only (detector score EMA, track age,
        mask fraction EMA); gates semantic memory eligibility, never
        physical association.
        """
        if self.validity_mode == "none":
            return 1.0
        cfg = self.validity_config
        feats = cfg.get("features", [])
        vals = []
        for f in feats:
            if f == "score":
                vals.append(score if score is not None else t.score_ema)
            elif f == "track_age":
                vals.append(float(t.age))
            elif f == "mask_frac":
                m = (aux or {}).get("mask_area_frac")
                vals.append(m if m is not None else t.mask_frac_ema)
            else:
                vals.append(0.0)
        if not vals:
            return 1.0
        mean = np.asarray(cfg.get("mean", [0.0] * len(vals)),
                          dtype=np.float32)
        scale = np.asarray(cfg.get("scale", [1.0] * len(vals)),
                           dtype=np.float32)
        coef = np.asarray(cfg.get("coef", [0.0] * len(vals)),
                          dtype=np.float32)
        x = (np.asarray(vals, dtype=np.float32) - mean) / \
            np.maximum(scale, 1e-6)
        logit = float(coef @ x + float(cfg.get("intercept", 0.0)))
        return float(1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0))))

    def decision_threshold(self, age):
        """Calibrated routing threshold; two-band when split_age is set."""
        if self.decision_split_age is None:
            return self.thr_early
        return self.thr_stable if age > self.decision_split_age \
            else self.thr_early

    def commit_qualifies(self, t):
        """Minimum evidence before a track may write global novel memory."""
        if self.commit_mode == "M0":
            return True
        return (t.age >= self.commit_min_age
                and t.novel_support >= self.commit_min_support)

    def embed(self, feat):
        """Single-frame adapted embedding via the frozen M2 aggregator."""
        x = torch.as_tensor(feat, dtype=torch.float32,
                            device=self.device).view(1, 1, -1)
        mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            out = self.model.aggregate(x, mask)
        z = out["z"][0].cpu().numpy().astype(np.float32)
        rel = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
        return z, rel

    def observe(self, feats):
        """Observe all current-frame detections against M_{t-1}.

        feats: (N,768) float32 single-frame DINOv2 embeddings.
        Returns a list of observation dicts aligned to feats rows.
        """
        obs = []
        P_novel = (np.stack(list(self.memory.protos.values())).astype(np.float32)
                   if self.memory.protos else np.empty((0, 768), dtype=np.float32))
        novel_ids = list(self.memory.protos)
        n_novel = len(novel_ids)
        for f in feats:
            if not np.any(f):
                # failed crop (rare): neutral observation, no association bias
                obs.append({
                    "z": np.zeros(768, dtype=np.float32), "rel": 0.0,
                    "p_known": 0.5,
                    "class_dist": np.ones(len(self.known_ids),
                                          dtype=np.float32) /
                    max(len(self.known_ids), 1),
                    "best_known": 0.0, "best_novel": -1.0,
                    "known_margin": 0.0,
                    "novel_id": None, "novel_conf": 0.0,
                    "reliability": 0.0,
                })
                continue
            key = f.tobytes()
            if key in self._z_cache:
                z = self._z_cache[key]
                rel = 1.0
            else:
                z, rel = self.embed(f)
                self._z_cache[key] = z
            ks = self.P_known @ z
            best_known = float(ks.max()) if ks.shape[0] else -1.0
            order = np.argsort(ks)[::-1]
            known_margin = (float(ks[order[0]] - ks[order[1]])
                            if ks.shape[0] >= 2 else 0.0)
            class_dist = np.zeros(len(self.known_ids), dtype=np.float32)
            if ks.shape[0]:
                class_dist = np.exp(ks / self.class_temp)
                class_dist = class_dist / class_dist.sum()
            best_n = second_n = -1.0
            margin_n = 0.0
            dist_n = 1.0
            novel_id = None
            if P_novel.shape[0]:
                ns = P_novel @ z
                best_n = float(ns.max())
                order_n = np.argsort(ns)[::-1]
                second_n = float(ns[order_n[1]]) if ns.shape[0] >= 2 else best_n
                margin_n = best_n - second_n
                novel_id = int(novel_ids[order_n[0]])
                r_n = 0.3
                dist_n = (1.0 - best_n) / max(r_n, 1e-6)
            gs = known_stats(z, self.P_known, self.radii,
                             known_ids=self.known_ids,
                             best_n=best_n, second_n=second_n,
                             margin_n=margin_n, dist_n=dist_n,
                             rel=rel, track_len=1, n_novel=n_novel,
                             include_anchor=False)
            with torch.no_grad():
                logit = float(self.model.gate_forward(
                    torch.as_tensor([gs], dtype=torch.float32,
                                    device=self.device))[0])
            p_known = float(torch.sigmoid(torch.as_tensor(logit)))
            obs.append({
                "z": z, "rel": rel, "p_known": p_known,
                "class_dist": class_dist, "best_known": best_known,
                "known_margin": known_margin,
                "best_novel": best_n, "novel_id": novel_id,
                "novel_conf": best_n,
                "reliability": 1.0,
            })
        return obs

    def consistency(self, det_obs, track_obs, novel_weight=1.0):
        """Continuous semantic consistency in [-1,1]; soft, no hard gate."""
        p_j = det_obs["p_known"]
        p_i = track_obs.get("p_known", track_obs.get("known_belief", 0.5))
        same_class = float(np.dot(det_obs["class_dist"],
                                  track_obs["class_dist"]))
        same_novel = 0.0
        if det_obs.get("novel_id") is not None and \
                track_obs.get("novel_id") is not None:
            if det_obs["novel_id"] == track_obs["novel_id"]:
                same_novel = min(det_obs.get("novel_conf", 0.0),
                                 track_obs.get("novel_conf", 0.0))
        kk = p_j * p_i * (2.0 * same_class - 1.0)
        nn = (1.0 - p_j) * (1.0 - p_i) * same_novel * novel_weight
        conflict = -0.5 * (p_j * (1.0 - p_i) + (1.0 - p_j) * p_i)
        return float(np.clip(kk + nn + conflict, -1.0, 1.0))

    def defer_evidence(self, z):
        """Causal ambiguity evidence for the track prefix z (Phase 4M).

        Computed from M_{t-1} novel prototypes and the frozen known
        prototypes only; never GT.  Returns None when there is no novel
        memory (nothing to be ambiguous against).
        """
        if not self.memory.protos:
            return None
        P = np.stack(list(self.memory.protos.values())).astype(np.float32)
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        z = _norm(z.astype(np.float32))
        ids = list(self.memory.protos)
        cos = (P @ z).astype(np.float64)
        order = np.argsort(cos)[::-1]
        best_cos = float(cos[order[0]])
        second_cos = float(cos[order[1]]) if len(order) >= 2 else -1.0
        margin = best_cos - second_cos
        top5 = [float(cos[order[k]]) if k < len(order) else -1.0
                for k in range(5)]
        soft = np.exp(np.clip(top5, -30, 30))
        soft = soft / soft.sum()
        entropy = float(-(soft * np.log(soft + 1e-12)).sum())
        best_known = float((self.P_known @ z).max()) \
            if self.P_known.shape[0] else -1.0
        return {
            "best_cos": best_cos, "margin": margin, "entropy": entropy,
            "novel_minus_known": best_cos - best_known,
            "best_proto_id": int(ids[order[0]]),
        }

    def should_defer(self, z, obs=None):
        """Phase 4M deferral rule (causal, track-local)."""
        if self.deferral_mode == "none":
            return False
        ev = self.defer_evidence(z)
        if ev is None:
            return False
        if self.deferral_mode == "margin":
            return ev["margin"] < self.defer_margin
        if self.deferral_mode == "entropy":
            return ev["entropy"] > self.defer_entropy
        if self.deferral_mode == "hybrid":
            return ev["margin"] < self.defer_margin or \
                ev["novel_minus_known"] < self.defer_nk
        if self.deferral_mode == "ambiguity":
            nmk = ev["best_cos"] - float(obs.get("best_known", 0.0)) \
                if obs is not None else ev["novel_minus_known"]
            x = np.asarray([nmk, ev["margin"], ev["best_cos"]],
                           dtype=np.float32)
            logit = float(self.defer_ambiguity_coef @ x) + \
                self.defer_ambiguity_intercept
            s = 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))
            return s > self.defer_ambiguity_threshold
        raise ValueError("unknown deferral_mode %s" % self.deferral_mode)

    def _mark_unresolved(self, t):
        """Track-local UNRESOLVED_NOVEL: no global prototype write."""
        if t.candidate_id is None:
            t.candidate_id = "L%d" % t.track_id
        t.novel_id = t.candidate_id
        t.resolution_state = "unresolved"
        t.unresolved_frames += 1
        t.last_action = "UNRESOLVED_NOVEL"
        self.branch_deferred += 1

    def semantic_cost_matrix(self, det_obs, active_ids):
        """(N_det, M_track) consistency matrix; columns follow memo order."""
        if not active_ids:
            return None
        M = np.zeros((len(det_obs), len(active_ids)), dtype=np.float32)
        for j, d in enumerate(det_obs):
            for k, tid in enumerate(active_ids):
                t = self.tracks.get(tid)
                if t is None:
                    continue
                # Uncommitted tracks keep a track-local provisional novel
                # identity.  A novel-like detection of the same physical
                # track inherits that identity for this pair, so soft
                # semantics stay active before global commitment without
                # leaking to other physical tracks.
                det = d
                if t.candidate_id is not None and \
                        d["p_known"] < self.decision_threshold(t.age):
                    det = dict(d)
                    det["novel_id"] = t.candidate_id
                    det["novel_conf"] = max(d.get("novel_conf", 0.0),
                                            t.novel_conf, 0.5)
                td = {
                    "p_known": t.known_belief,
                    "class_dist": t.class_dist,
                    "novel_id": t.novel_id,
                    "novel_conf": t.novel_conf,
                }
                nid = det.get("novel_id")
                nw = self.memory.prototype_weight.get(nid, 1.0) \
                    if isinstance(nid, int) else 1.0
                c = self.consistency(det, td, novel_weight=nw)
                M[j, k] = c
        return M

    def prune(self, frame_id):
        for tid in list(self.tracks):
            if frame_id - self.tracks[tid].last_frame >= self.memo_tracklet_frames:
                del self.tracks[tid]

    def _apply_association(self, frame_id, pairs):
        """Shared post-association state update.

        pairs: iterable of (tid, obs, score).  `obs` is the frame-synchronous
        observation dict; score is the raw detection score.
        """
        # Physical track IDs are video-local; never merge semantic track
        # states across video boundaries (global novel memory persists).
        if self._last_video_id is not None and \
                self._last_video_id != self.video_id:
            self.tracks.clear()
        self._last_video_id = self.video_id
        self.prune(frame_id)
        for pair in pairs:
            tid, obs, score = pair[0], pair[1], pair[2]
            aux = pair[3] if len(pair) > 3 else None
            if tid < 0:
                continue
            if tid in self.tracks:
                t = self.tracks[tid]
                t.update(obs, self.belief_alpha, aux)
                t.last_frame = frame_id
                t.add_weighted(obs["z"], obs["rel"], score)
            else:
                t = TrackSemState(tid, obs, self.prefix_mode)
                t.last_frame = frame_id
                self.tracks[tid] = t
            t = self.tracks[tid]
            admiss = self.admissibility(t)
            thr = self.decision_threshold(t.age)
            if obs["p_known"] < thr:
                t.novel_support += 1
            # semantic identity assignment (causal, after association);
            # observation stays active even when global commitment is gated.
            z = t.prefix()
            is_known = t.known_belief >= thr
            if is_known:
                t.novel_id = None
                t.candidate_id = None
                t.last_action = "KNOWN"
                continue
            best_n, gid = self.memory.best(z)
            validity_ok = self.validity(t, score, aux) >= \
                self.validity_threshold
            qualifies = self.commit_qualifies(t) and validity_ok
            obs_nid = obs.get("novel_id")
            obs_best = float(obs.get("best_novel", -1.0))
            # sticky active global identity (Phase 4I semantics): the
            # current effective novel id is kept even if the prefix no
            # longer argmax-matches it; cleared when the track is known.
            sticky = (isinstance(t.novel_id, int)
                      and self.memory.protos.get(t.novel_id) is not None)
            if sticky:
                self.branch_sticky += 1
                sem_id = t.novel_id
                t.novel_conf = max(best_n, 0.0)
                did_update = False
                if t.committed_sem_id is not None or qualifies:
                    self.memory.update(sem_id, z, t.reliability, admiss)
                    did_update = True
                if t.committed_sem_id is None and qualifies:
                    t.mark_committed(sem_id, "EXISTING_NOVEL", frame_id)
                    self.memory.commit_events.append({
                        "frame_id": int(frame_id),
                        "track_id": t.track_id,
                        "sem_id": sem_id, "action": "EXISTING_NOVEL",
                        "age": t.age, "support": t.novel_support,
                    })
                if self.provenance is not None:
                    self.provenance.log_reuse(
                        self.video_id, frame_id, sem_id,
                        (self.video_id, t.track_id), best_n,
                        selected=True, triggered_update=did_update)
                    if did_update:
                        self.provenance.log_update(
                            self.video_id, frame_id, sem_id,
                            (self.video_id, t.track_id),
                            self.memory.support.get(sem_id, 0), best_n, z)
                t.last_action = ("EXISTING_NOVEL"
                                 if t.committed_sem_id is not None
                                 else "EXISTING_NOVEL_PROVISIONAL")
                continue
            if isinstance(obs_nid, int) and \
                    obs_best >= self.theta_novel and \
                    self.memory.relative_ok(z, obs_best, obs_nid):
                self.branch_soft += 1
                if self._debug_branch:
                    print("SOFT", self.video_id, frame_id, tid,
                          obs_nid, round(obs_best, 3), flush=True)
                if t.committed_sem_id is None and self.should_defer(z, obs):
                    self._mark_unresolved(t)
                    continue
                # soft identity from existing global novel memory, used by
                # association immediately; update is gated by evidence.
                t.novel_id = obs_nid
                t.novel_conf = obs_best
                did_update = False
                if qualifies:
                    self.memory.update(obs_nid, z, t.reliability, admiss)
                    did_update = True
                    if t.committed_sem_id is None:
                        t.mark_committed(obs_nid, "EXISTING_NOVEL", frame_id)
                        self.memory.commit_events.append({
                            "frame_id": int(frame_id),
                            "track_id": t.track_id,
                            "sem_id": obs_nid, "action": "EXISTING_NOVEL",
                            "age": t.age, "support": t.novel_support,
                        })
                if self.provenance is not None:
                    self.provenance.log_reuse(
                        self.video_id, frame_id, obs_nid,
                        (self.video_id, t.track_id), best_n,
                        selected=True, triggered_update=did_update)
                    if did_update:
                        self.provenance.log_update(
                            self.video_id, frame_id, obs_nid,
                            (self.video_id, t.track_id),
                            self.memory.support.get(obs_nid, 0), best_n, z)
                t.last_action = ("EXISTING_NOVEL" if t.committed_sem_id
                                 is not None else "EXISTING_NOVEL_PROVISIONAL")
                continue
            # NEW_NOVEL branch: track-local provisional identity until the
            # admission gate is satisfied.
            self.branch_new += 1
            if self._debug_branch:
                print("NEW", self.video_id, frame_id, tid,
                      round(best_n, 3), flush=True)
            if t.committed_sem_id is None and self.should_defer(z, obs):
                self._mark_unresolved(t)
                continue
            if t.candidate_id is None:
                t.candidate_id = "L%d" % t.track_id
            t.novel_id = t.candidate_id
            if qualifies:
                sem_id, _ = self.memory.propose(z, t.reliability, admiss)
                self.memory.update(sem_id, z, t.reliability, admiss)
                t.novel_id = sem_id
                t.mark_committed(sem_id, "NEW_NOVEL", frame_id)
                # Phase 4I-compatible confidence for committed identities
                t.novel_conf = best_n if best_n >= 0 else 0.0
                self.memory.commit_events.append({
                    "frame_id": int(frame_id),
                    "track_id": t.track_id,
                    "sem_id": sem_id, "action": "NEW_NOVEL",
                    "age": t.age, "support": t.novel_support,
                })
                if self.provenance is not None:
                    self.provenance.log_birth(
                        self.video_id, frame_id, sem_id,
                        (self.video_id, t.track_id), t.age, t.age, score,
                        obs.get("p_known", 0.0),
                        obs.get("best_known", -1.0),
                        obs.get("known_margin", 0.0),
                        obs.get("best_novel", -1.0),
                        t.novel_conf, z, "NEW_NOVEL",
                        self.memory.support.get(sem_id, 0))
                t.last_action = "NEW_NOVEL"
            else:
                # provisional local confidence floor: same-track identity
                # is self-consistent (>=0.5), never shared across tracks
                t.novel_conf = max(best_n if best_n >= 0 else 0.0, 0.5)
                t.last_action = "PROVISIONAL_NOVEL"

    def post_association(self, frame_id, ids, bboxes, scores, det_obs,
                         feats, det_aux=None):
        """Update track semantic states and novel memory AFTER association.

        ids: torch long tensor aligned to raw detections; ids >= 0 are
        matched/new physical tracks, -1/-2 are not tracked.
        """
        pairs = []
        for idx, tid in enumerate(ids.tolist()):
            tid = int(tid)
            if tid < 0:
                continue
            score = float(scores[idx]) if idx < len(scores) else 1.0
            aux = det_aux[idx] if det_aux is not None and \
                idx < len(det_aux) else None
            pairs.append((tid, det_obs[idx], score, aux))
        self._apply_association(frame_id, pairs)

    def post_association_raw(self, frame_id, ids, bboxes, scores, det_obs,
                             raw_idx, det_aux=None):
        """Like post_association but det_obs is indexed by raw detection
        order while ids follow the post mask-nms order."""
        pairs = []
        for p, tid in enumerate(ids.tolist()):
            tid = int(tid)
            if tid < 0:
                continue
            ridx = int(raw_idx[p])
            score = float(scores[ridx]) if ridx < len(scores) else 1.0
            aux = det_aux[ridx] if det_aux is not None and \
                ridx < len(det_aux) else None
            pairs.append((tid, det_obs[ridx], score, aux))
        self._apply_association(frame_id, pairs)

    def log_row(self, frame_id, idx, det_obs, tid, score, bbox):
        o = det_obs[idx]
        kc = int(self.known_ids[int(np.argmax(o["class_dist"]))]) \
            if len(self.known_ids) else None
        t = self.tracks.get(tid) if tid >= 0 else None
        thr = self.decision_threshold(t.age) if t is not None else \
            self.decision_threshold(1)
        pred_known = o["p_known"] >= thr
        if pred_known:
            action = "KNOWN"
        elif t is not None and t.commit_action is not None and \
                t.commit_frame == int(frame_id):
            action = t.commit_action
        elif t is not None and t.resolution_state == "unresolved":
            action = "UNRESOLVED_NOVEL"
        elif t is not None and t.committed_sem_id is not None:
            action = "EXISTING_NOVEL"
        elif o.get("novel_id") is not None and \
                o.get("best_novel", -1.0) >= self.theta_novel:
            action = "EXISTING_NOVEL"
        else:
            action = "PROVISIONAL_NOVEL"
        effective_novel = None if pred_known else o.get("novel_id")
        if t is not None and not pred_known:
            effective_novel = t.novel_id
        global_nid = None
        mem_support = 0
        if isinstance(effective_novel, int) and \
                effective_novel in self.memory.protos:
            global_nid = effective_novel
            mem_support = self.memory.support.get(effective_novel, 0)
        return {
            "frame_id": int(frame_id),
            "det_idx": int(idx),
            "physical_track_id": int(tid) if tid >= 0 else None,
            "score": float(score),
            "bbox": [float(v) for v in bbox],
            "p_known": float(o["p_known"]),
            "decision_threshold": float(thr),
            "best_known": float(o["best_known"]),
            "best_novel": float(o["best_novel"]),
            "novel_id": effective_novel,
            "global_novel_id": global_nid,
            "novel_mem_support": int(mem_support),
            "known_class_id": kc,
            "semantic_id": ("K" + str(kc) if pred_known
                            else ("N" + str(effective_novel)
                                  if effective_novel is not None else "N?")),
            "semantic_action": action,
            "commit_state": ("committed" if t is not None and
                             t.committed_sem_id is not None
                             else "provisional"),
            "track_age": int(t.age) if t is not None else 1,
            "novel_support": int(t.novel_support) if t is not None else 0,
            "resolution_state": (t.resolution_state
                                 if t is not None else "resolved"),
            "validity": round(self.validity(t), 4) if t is not None else 1.0,
            "n_novel_memory": self.memory.size(),
        }


def build_semantic_manager(model, device, prefix_mode="P1", theta_novel=0.6,
                           class_temp=0.05, belief_alpha=0.5,
                           novel_update_rate=0.2, memo_tracklet_frames=10,
                           decision_threshold=0.5, decision_split_age=None,
                           commit_mode="M0", commit_min_age=2,
                           commit_min_support=2, provenance=None,
                           admissibility_mode="none",
                           admissibility_config=None,
                           matching_mode="absolute", margin_threshold=0.05,
                           entropy_threshold=1.6, deferral_mode="none",
                           defer_margin=0.10, defer_entropy=1.6,
                           defer_nk=0.25, defer_ambiguity_coef=None,
                           defer_ambiguity_intercept=0.0,
                           defer_ambiguity_threshold=0.5,
                           validity_mode="none", validity_config=None,
                           validity_threshold=0.03):
    """Build known prototypes from the frozen train-known split and return
    a SemanticStateManager.  Train-side only; no official data."""
    labels = load_train_labels()
    feats = load_frame_features("train_known_mean")
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in feats:
            by_class[int(c)].append(sid)
    protos, radii = {}, {}
    for c, ids in by_class.items():
        zs = []
        for sid in ids:
            z, _ = embed_track(model, feats[sid], device)
            zs.append(z)
        Z = np.stack(zs)
        p = _norm(Z.mean(axis=0).astype(np.float32))
        protos[int(c)] = p
        cos = Z @ p
        radii[int(c)] = float(np.percentile(1.0 - cos, 50).clip(min=0.02))
    return SemanticStateManager(
        model=model, known_protos=protos, radii=radii, device=device,
        prefix_mode=prefix_mode, theta_novel=theta_novel,
        class_temp=class_temp, belief_alpha=belief_alpha,
        novel_update_rate=novel_update_rate,
        memo_tracklet_frames=memo_tracklet_frames,
        decision_threshold=decision_threshold,
        decision_split_age=decision_split_age,
        commit_mode=commit_mode, commit_min_age=commit_min_age,
        commit_min_support=commit_min_support, provenance=provenance,
        admissibility_mode=admissibility_mode,
        admissibility_config=admissibility_config,
        matching_mode=matching_mode, margin_threshold=margin_threshold,
        entropy_threshold=entropy_threshold, deferral_mode=deferral_mode,
        defer_margin=defer_margin, defer_entropy=defer_entropy,
        defer_nk=defer_nk, defer_ambiguity_coef=defer_ambiguity_coef,
        defer_ambiguity_intercept=defer_ambiguity_intercept,
        defer_ambiguity_threshold=defer_ambiguity_threshold,
        validity_mode=validity_mode, validity_config=validity_config,
        validity_threshold=validity_threshold)
