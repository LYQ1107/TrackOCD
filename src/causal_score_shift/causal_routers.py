from __future__ import annotations

import numpy as np


def mad(x):
    return float(np.median(np.abs(np.asarray(x) - np.median(np.asarray(x)))))


class CausalRouter:
    """Base causal router with the required online interface."""

    name = "base"

    def __init__(self, threshold=0.45):
        self.threshold = threshold
        self.video_id = None
        self.history = []  # (s1, margin, decision_is_known)

    def reset_video(self, video_id):
        self.video_id = video_id
        self.history = []

    def predict(self, state):
        raise NotImplementedError

    def update_after_prediction(self, state, is_known):
        self.history.append((state["s1"], state.get("margin", 0.0), is_known))

    def state_dict(self):
        return {"video_id": self.video_id, "history": self.history,
                "threshold": self.threshold}

    def load_state_dict(self, d):
        self.video_id = d["video_id"]
        self.history = d["history"]
        self.threshold = d["threshold"]


class C0Legacy(CausalRouter):
    name = "C0"

    def predict(self, state):
        return state["s1"] >= 0.45


class C1Global(CausalRouter):
    name = "C1"

    def __init__(self, threshold):
        super().__init__(threshold)

    def predict(self, state):
        return state["s1"] >= self.threshold


class C2Translation(CausalRouter):
    name = "C2"

    def __init__(self, threshold, ref_median, hc_score, hc_margin,
                 min_anchors=10, ema=0.25, max_shift=None):
        super().__init__(threshold)
        self.ref_median = ref_median
        self.hc_score = hc_score
        self.hc_margin = hc_margin
        self.min_anchors = min_anchors
        self.ema = ema
        self.max_shift = max_shift
        self.anchors = []
        self.shift_ema = 0.0

    def reset_video(self, video_id):
        super().reset_video(video_id)
        self.anchors = []
        self.shift_ema = 0.0

    def _shift(self):
        if not self.anchors:
            return 0.0
        sh = float(np.median(self.anchors)) - self.ref_median
        if self.max_shift is not None:
            sh = float(np.clip(sh, -self.max_shift, self.max_shift))
        self.shift_ema = self.ema * sh + (1 - self.ema) * self.shift_ema
        return self.shift_ema

    def predict(self, state):
        if len(self.anchors) < self.min_anchors:
            return state["s1"] >= self.threshold
        return (state["s1"] - self._shift()) >= self.threshold

    def update_after_prediction(self, state, is_known):
        super().update_after_prediction(state, is_known)
        if (is_known and state["s1"] >= self.hc_score
                and state.get("margin", 0.0) >= self.hc_margin):
            self.anchors.append(float(state["s1"]))

    def state_dict(self):
        d = super().state_dict()
        d["anchors"] = self.anchors
        d["shift_ema"] = self.shift_ema
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.anchors = d.get("anchors", [])
        self.shift_ema = d.get("shift_ema", 0.0)


class C3LocationScale(C2Translation):
    name = "C3"

    def __init__(self, threshold, ref_median, ref_mad, hc_score, hc_margin,
                 min_anchors=10, min_scale_anchors=10, ema=0.25,
                 max_shift=None, max_scale_ratio=3.0):
        super().__init__(threshold, ref_median, hc_score, hc_margin,
                         min_anchors, ema, max_shift)
        self.ref_mad = ref_mad
        self.min_scale_anchors = min_scale_anchors
        self.max_scale_ratio = max_scale_ratio

    def predict(self, state):
        if len(self.anchors) < self.min_anchors:
            return state["s1"] >= self.threshold
        loc = float(np.median(self.anchors))
        scale = mad(self.anchors)
        if len(self.anchors) < self.min_scale_anchors or scale < 1e-6:
            return (state["s1"] - self._shift()) >= self.threshold
        ratio = scale / max(self.ref_mad, 1e-9)
        ratio = float(np.clip(ratio, 1.0 / self.max_scale_ratio, self.max_scale_ratio))
        z = (state["s1"] - loc) / (scale + 1e-9)
        aligned = self.ref_median + z * (self.ref_mad * ratio)
        return aligned >= self.threshold


class C4Reliability(C2Translation):
    name = "C4"

    def __init__(self, threshold, ref_median, hc_score, hc_margin,
                 min_anchors=10, ema=0.25, max_shift=None, anchor_target=20):
        super().__init__(threshold, ref_median, hc_score, hc_margin,
                         min_anchors, ema, max_shift)
        self.anchor_target = anchor_target

    def predict(self, state):
        if not self.anchors:
            return state["s1"] >= self.threshold
        w = min(1.0, len(self.anchors) / self.anchor_target)
        raw_score = state["s1"]
        adapted = raw_score - self._shift()
        return (1 - w) * raw_score + w * adapted >= self.threshold


class C5AllTrack(CausalRouter):
    name = "C5"

    def __init__(self, threshold=0.45, min_tracks=10):
        super().__init__(threshold)
        self.min_tracks = min_tracks
        self.all_scores = []

    def reset_video(self, video_id):
        super().reset_video(video_id)
        self.all_scores = []

    def predict(self, state):
        if len(self.all_scores) < self.min_tracks:
            return state["s1"] >= self.threshold
        loc = float(np.median(self.all_scores))
        scale = mad(self.all_scores)
        if scale < 1e-6:
            return state["s1"] >= self.threshold
        z = (state["s1"] - loc) / scale
        return self.threshold <= 0.5 + 0.5 * np.tanh(z)

    def update_after_prediction(self, state, is_known):
        super().update_after_prediction(state, is_known)
        self.all_scores.append(float(state["s1"]))
