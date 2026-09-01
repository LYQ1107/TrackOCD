"""Known/open-set branch and frozen raw baseline controllers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from src.iclr27_phase19r.runtime.state import StateMemory


class KnownOSR:
    def __init__(self, data: Any, tau_known: float | None = None):
        self.data = data
        self.prototypes = np.asarray(data.known_prototypes, np.float32)
        self.active = np.asarray(data.active_known_mask, bool)
        self.ids = list(data.supported_ids)
        self.tau_known = float(tau_known if tau_known is not None else self._calibrate())

    def _calibrate(self) -> float:
        # Fold-local calibration uses only legal supported-known fitting rows.
        scores = []
        for key, cat in self.data.track_category.items():
            if cat not in self.data.supported_set or cat in self.data.held_categories:
                continue
            if self.data.track_video[key] not in self.data.fit_videos:
                continue
            raw, _, _, _ = self.data.prefix(key)
            j = self.data.known_to_index.get(cat)
            if j is not None and self.active[j]:
                scores.append(float(raw @ self.prototypes[j]))
        if len(scores) < 8:
            return .38
        # Conservative lower quantile retains known safety rather than
        # selecting a threshold from held/public events.
        return float(np.clip(np.quantile(np.asarray(scores), .08), .25, .75))

    def score(self, raw: np.ndarray) -> tuple[int | None, float]:
        sims = self.prototypes @ raw
        sims = np.where(self.active, sims, -np.inf)
        j = int(np.argmax(sims))
        if not np.isfinite(sims[j]):
            return None, -1.0
        return self.ids[j], float(sims[j])

    def accepts(self, score: float) -> bool:
        return float(score) >= self.tau_known


class RawPersistentController:
    """B1 raw cosine controller with one persistent stream memory."""

    name = "raw cosine controller"

    def __init__(self, data: Any, *, deferred: bool = True, tau_ready: float | None = None,
                 tau_assign: float | None = None):
        self.data = data
        self.known = KnownOSR(data)
        self.deferred = bool(deferred)
        self.tau_ready = float(tau_ready if tau_ready is not None else .45)
        self.tau_assign = float(tau_assign if tau_assign is not None else .65)
        self.memory = StateMemory(max_states=16, max_anchors=8)

    def reset_stream(self) -> None:
        self.memory.reset()

    def _track(self, key: str, phase: str = "", eval_category: int | None = None) -> list[dict[str, Any]]:
        out = []
        for pos in range(len(self.data.track_rows[key])):
            raw, geom, quality, _ = self.data.prefix(key, pos)
            row = self.data.rows[self.data.track_rows[key][pos]]
            video = int(row["video_id"])
            candidates = self.memory.candidate_indices(video, key)
            action = "DEFER"; sid = None; conf = quality; state_idx = None
            if (not self.deferred) or quality >= self.tau_ready:
                known_id, known_score = self.known.score(raw)
                if known_id is not None and self.known.accepts(known_score):
                    action, sid, conf = "KNOWN", known_id, known_score
                else:
                    best = None
                    for j in candidates:
                        score = self._score_state(raw, self.memory.states[j])
                        if best is None or score > best[1]: best = (j, score)
                    if best is not None and best[1] >= self.tau_assign:
                        action, state_idx = "EXISTING", best[0]; sid = self.memory.states[best[0]].sid; conf = best[1]
                    else:
                        action, conf = "NEW", max(0.0, 1.0 - known_score)
            rec = self.memory.apply_action(action, _torch(raw), _torch(raw), video, key,
                                           state_index=state_idx if action == "EXISTING" else None,
                                           oracle_category=eval_category, quality=quality, confidence=float(conf), update_allowed=True)
            sid = rec["semantic_id"] if action != "KNOWN" else sid
            out.append({"row_key": row["row_key"], "tracklet_position": pos,
                        "phase": phase, "action": action, "semantic_id": sid,
                        "readiness": quality, "confidence": float(conf), "video": video,
                        "state_count": rec["state_count"], "candidate_count": len(candidates)})
        return out

    def _score_state(self, raw: np.ndarray, state: Any) -> float:
        return float(raw @ state.raw.detach().cpu().numpy())

    def process_track(self, key: str, phase: str = "", eval_category: int | None = None) -> list[dict[str, Any]]:
        return self._track(key, phase=phase, eval_category=eval_category)


class GaussianController(RawPersistentController):
    """Causal PCA/Ledoit-Wolf Gaussian state scorer (AGE-style adaptation)."""

    name = "Tracklet-AGE adaptation"

    def __init__(self, data: Any, **kwargs: Any):
        super().__init__(data, **kwargs)
        fit = data.raw[data.fit_rows]
        self.mean = fit.mean(0).astype(np.float32)
        # A deterministic compact PCA keeps the adaptation inexpensive.
        _, _, vt = np.linalg.svd(fit[:: max(1, len(fit) // 4096)] - self.mean, full_matrices=False)
        self.components = vt[: min(64, vt.shape[0])].astype(np.float32)
        proj = (fit[:: max(1, len(fit) // 4096)] - self.mean) @ self.components.T
        self.cov = LedoitWolf().fit(proj).covariance_.astype(np.float32)
        self.prec = np.linalg.pinv(self.cov).astype(np.float32)
        self.logdet = float(np.linalg.slogdet(self.cov)[1])
        # Calibrate the Gaussian EXISTING threshold from legal supported-known
        # same-category cross-video pairs only.  No held/public truth is used.
        same_scores = []
        for cat in sorted(getattr(data, "train_categories", [])):
            keys = [k for k in data.category_tracks.get(int(cat), []) if data.track_video[k] in data.fit_videos]
            if len(keys) < 2:
                continue
            base = keys[0]
            raw_base, _, _, _ = data.prefix(base)
            anchor = type("_Anchor", (), {"anchors": [raw_base], "raw": _torch(raw_base)})()
            for other in keys[1:3]:
                raw_other, _, _, _ = data.prefix(other)
                same_scores.append(self._score_state(raw_other, anchor))
        if same_scores:
            self.tau_assign = float(np.quantile(np.asarray(same_scores, np.float32), .10))

    def _score_state(self, raw: np.ndarray, state: Any) -> float:
        anchors = np.asarray(state.anchors if state.anchors else [state.raw.detach().cpu().numpy()], np.float32)
        p = (anchors - self.mean) @ self.components.T
        q = (raw - self.mean) @ self.components.T
        d = p.mean(0) - q
        ll = -.5 * float(d @ self.prec @ d) - .5 * self.logdet
        return ll


class TALONStyleController(RawPersistentController):
    """Margin-calibrated prototype-TTA adaptation, not exact TALON."""

    name = "TALON-style adaptation"

    def __init__(self, data: Any, **kwargs: Any):
        super().__init__(data, **kwargs)
        self.margin_threshold = .08

    def _score_state(self, raw: np.ndarray, state: Any) -> float:
        # A compact, causal margin calibration: diffuse states require a larger
        # raw-similarity margin.  The parent transition performs the bounded
        # prototype EMA (TTA) only after an accepted EXISTING decision.
        return float(super()._score_state(raw, state) - self.margin_threshold * min(float(state.dispersion), 1.0))

    def _track(self, key: str, phase: str = "", eval_category: int | None = None) -> list[dict[str, Any]]:
        return super()._track(key, phase=phase, eval_category=eval_category)


def _torch(x: np.ndarray):
    import torch
    return torch.from_numpy(np.asarray(x, np.float32))
