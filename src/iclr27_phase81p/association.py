"""Small Q0-anchored causal association model.

The module consumes only proposal/track geometry, score and causal appearance
summaries. Track-slot identifiers are runtime bookkeeping and never enter a
pair feature tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # allows audit scripts to import with system Python
    torch = None  # type: ignore
    nn = object  # type: ignore

PAIR_DIM = 16


def crop_descriptor(image_path: str, bbox: Sequence[float], size: Tuple[int, int] = (16, 16)) -> np.ndarray:
    """Return an 8-D causal RGB crop descriptor.

    This is deliberately category/ID agnostic: mean and standard deviation
    over a clipped crop are the only visual statistics. Missing images produce
    a deterministic zero descriptor and are counted by the caller.
    """
    try:
        from PIL import Image
        with Image.open(image_path).convert("RGB") as image:
            w, h = image.size
            x0, y0, x1, y1 = [float(x) for x in bbox]
            x0, x1 = max(0.0, min(w - 1.0, x0)), max(1.0, min(float(w), x1))
            y0, y1 = max(0.0, min(h - 1.0, y0)), max(1.0, min(float(h), y1))
            if x1 <= x0 or y1 <= y0:
                return np.zeros(8, dtype=np.float32)
            crop = image.crop((int(x0), int(y0), int(x1), int(y1))).resize(size)
            arr = np.asarray(crop, dtype=np.float32) / 255.0
            mean = arr.mean(axis=(0, 1)); std = arr.std(axis=(0, 1))
            gray = arr.mean(axis=2)
            return np.asarray([mean[0], mean[1], mean[2], std[0], std[1], std[2], float(gray[:8].mean()), float(gray[8:].mean())], dtype=np.float32)
    except Exception:
        return np.zeros(8, dtype=np.float32)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ab = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def pair_features(det: Dict[str, object], track: Dict[str, object], image_size: Tuple[float, float] = (640.0, 480.0), use_motion: bool = False) -> np.ndarray:
    """Build the registered 16-D causal pair vector.

    ``use_motion`` is an isolated Phase81P repair route.  It replaces the
    last-observation geometry with a constant-velocity causal prediction while
    preserving the vector width and all non-motion features.  The default is
    byte/shape compatible with the original geometry-only route.
    """
    db = np.asarray(det["bbox_xyxy"], dtype=np.float32)
    tb = np.asarray(track["last_bbox"], dtype=np.float32)
    iw, ih = float(image_size[0]), float(image_size[1])
    gap = max(0, min(32, int(det.get("frame_id", 0)) - int(track.get("last_frame", 0))))
    reference = tb
    if use_motion:
        velocity = np.asarray(track.get("velocity", np.zeros(4, dtype=np.float32)), dtype=np.float32)
        if velocity.shape == (4,):
            reference = tb + velocity * float(gap)
    dc = np.asarray([(db[0] + db[2]) / 2.0 / iw, (db[1] + db[3]) / 2.0 / ih])
    tc = np.asarray([(reference[0] + reference[2]) / 2.0 / iw, (reference[1] + reference[3]) / 2.0 / ih])
    dw, dh = max(1e-4, (db[2] - db[0]) / iw), max(1e-4, (db[3] - db[1]) / ih)
    tw, th = max(1e-4, (reference[2] - reference[0]) / iw), max(1e-4, (reference[3] - reference[1]) / ih)
    da = np.asarray(det.get("appearance", np.zeros(8, dtype=np.float32)), dtype=np.float32)
    ta = np.asarray(track.get("appearance_ema", np.zeros_like(da)), dtype=np.float32)
    denom = float(np.linalg.norm(da) * np.linalg.norm(ta))
    cos = float(np.dot(da, ta) / denom) if denom > 1e-8 else 0.0
    return np.asarray([
        cos, _box_iou(db, reference), float(dc[0] - tc[0]), float(dc[1] - tc[1]),
        float(np.log(dw / tw)), float(np.log(dh / th)),
        float(gap) / 8.0,
        float(min(32, int(track.get("age", 1))) / 32.0), float(min(8, int(track.get("miss_count", 0))) / 8.0),
        float(det.get("base_score", det.get("score", 0.0))), float(track.get("score_ema", 0.0)),
        float(track.get("association_ema", 0.0)), float(np.linalg.norm(da - ta)),
        float((db[2] - db[0]) * (db[3] - db[1]) / max(1.0, iw * ih)),
        float((reference[2] - reference[0]) * (reference[3] - reference[1]) / max(1.0, iw * ih)),
        float(track.get("hit_count", 0) / max(1, track.get("age", 1))),
    ], dtype=np.float32)


if torch is not None:
    class AssociationTransformer(nn.Module):
        """Compact one-layer temporal transformer and pair/new heads."""

        def __init__(self, pair_dim: int = PAIR_DIM, hidden: int = 128, heads: int = 4, history_len: int = 8):
            super().__init__()
            self.history_len = history_len
            self.input = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden), nn.GELU())
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=0.1, batch_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=1)
            self.pair_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            self.new_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        def forward(self, pair_history: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            if pair_history.ndim == 2:
                pair_history = pair_history.unsqueeze(0)
            encoded = self.temporal(self.input(pair_history))
            context = encoded[:, -1]
            return self.pair_head(context).squeeze(-1), self.new_head(context).squeeze(-1)

        def score_matrix(self, pair_matrix: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            if pair_matrix.ndim != 3:
                raise ValueError("pair_matrix must have shape [tracks,detections,PAIR_DIM]")
            tracks, dets, feats = pair_matrix.shape
            flat = pair_matrix.reshape(tracks * dets, 1, feats)
            logits, _ = self.forward(flat)
            logits = logits.reshape(tracks, dets)
            # A new/birth alternative is scored from each detection's mean pair context.
            _, new_logits = self.forward(pair_matrix.mean(dim=0).unsqueeze(1))
            return logits, new_logits

        def score_candidates(self, candidates: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Score listwise candidates shaped ``[batch, candidates, features]``."""
            if candidates.ndim != 3 or candidates.shape[-1] != PAIR_DIM:
                raise ValueError("candidates must have shape [batch,K,PAIR_DIM]")
            batch, count, feats = candidates.shape
            flat = candidates.reshape(batch * count, 1, feats)
            pair, _ = self.forward(flat)
            pair = pair.reshape(batch, count)
            _, new = self.forward(candidates.mean(dim=1, keepdim=True))
            return pair, new


    @dataclass
    class TrackState:
        last_bbox: np.ndarray
        last_frame: int
        appearance_ema: np.ndarray
        score_ema: float
        velocity: np.ndarray = None
        association_ema: float = 0.0
        age: int = 1
        miss_count: int = 0
        hit_count: int = 1
        physical_track_id: int = -1

        def __post_init__(self):
            if self.velocity is None:
                self.velocity = np.zeros(4, dtype=np.float32)

        def as_dict(self) -> Dict[str, object]:
            return {"last_bbox": self.last_bbox, "last_frame": self.last_frame, "appearance_ema": self.appearance_ema, "score_ema": self.score_ema, "velocity": self.velocity, "association_ema": self.association_ema, "age": self.age, "miss_count": self.miss_count, "hit_count": self.hit_count}


    class CausalAssociationRuntime:
        """One-to-one causal Hungarian association over Q0 detections."""

        def __init__(self, model: AssociationTransformer, device: str = "cpu", max_miss: int = 8, match_margin: float = 0.0, max_tracks: int = 256, use_motion: bool = False):
            from scipy.optimize import linear_sum_assignment
            self.model = model.to(device).eval(); self.device = device
            self.max_miss = int(max_miss); self.match_margin = float(match_margin); self.max_tracks = int(max_tracks); self.use_motion = bool(use_motion)
            self._hungarian = linear_sum_assignment; self.tracks: List[TrackState] = []; self.next_id = 0

        @torch.no_grad()
        def step(self, detections: Sequence[Dict[str, object]], frame_id: int, image_size: Tuple[float, float] = (640.0, 480.0)) -> List[Dict[str, object]]:
            dets = [dict(x) for x in detections]
            active = [t for t in self.tracks if t.miss_count <= self.max_miss]
            if active and dets:
                mat = np.stack([pair_features(d, t.as_dict(), image_size, use_motion=self.use_motion) for t in active for d in dets], axis=0)
                tensor = torch.from_numpy(mat).to(self.device).reshape(len(active), len(dets), PAIR_DIM)
                logits, new_logits = self.model.score_matrix(tensor)
                scores, births = logits.detach().cpu().numpy(), new_logits.detach().cpu().numpy()
                rows, cols = self._hungarian(-scores)
                matches = {(int(r), int(c)) for r, c in zip(rows, cols) if float(scores[r, c]) >= float(births[c]) + self.match_margin}
            else:
                scores = np.zeros((len(active), len(dets)), dtype=np.float32)
                births = np.asarray([float(d.get("base_score", d.get("score", 0.0))) for d in dets], dtype=np.float32)
                matches = set()
            by_det = {c: r for r, c in matches}; out: List[Dict[str, object]] = []
            for j, det in enumerate(dets):
                if j in by_det:
                    track = active[by_det[j]]; assoc_score = float(scores[by_det[j], j]); lifecycle = "continuation"; tid = track.physical_track_id
                    previous_bbox = track.last_bbox.copy(); dt = max(1, int(frame_id) - int(track.last_frame)); current_bbox = np.asarray(det["bbox_xyxy"], dtype=np.float32)
                    if self.use_motion:
                        delta = (current_bbox - previous_bbox) / float(dt); track.velocity = 0.8 * track.velocity + 0.2 * delta
                    track.last_bbox = current_bbox; track.last_frame = int(frame_id); track.age += 1; track.hit_count += 1; track.miss_count = 0
                    app = np.asarray(det.get("appearance", track.appearance_ema), dtype=np.float32); track.appearance_ema = 0.8 * track.appearance_ema + 0.2 * app
                    track.score_ema = 0.8 * track.score_ema + 0.2 * float(det.get("base_score", det.get("score", 0.0))); track.association_ema = 0.8 * track.association_ema + 0.2 * assoc_score
                else:
                    desc = np.asarray(det.get("appearance", np.zeros(8, dtype=np.float32)), dtype=np.float32)
                    track = TrackState(np.asarray(det["bbox_xyxy"], dtype=np.float32), int(frame_id), desc, float(det.get("base_score", det.get("score", 0.0))), physical_track_id=self.next_id)
                    self.next_id += 1; self.tracks.append(track); lifecycle = "birth"; tid = track.physical_track_id; assoc_score = float(births[j]) if j < len(births) else 0.0
                row = dict(det); row.update({"assigned_track_slot": int(tid), "physical_track_id": int(tid), "association_score": assoc_score, "assignment_candidate_rank": int(det.get("candidate_rank", j)), "lifecycle_action": lifecycle, "track_age": int(track.age), "miss_count": int(track.miss_count), "learned_match_score": assoc_score}); out.append(row)
            matched_indices = {r for r, _ in matches}
            for r, track in enumerate(active):
                if r not in matched_indices:
                    track.miss_count += 1; track.age += 1
            self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]
            if len(self.tracks) > self.max_tracks:
                # Fixed causal memory bound: retain recent, well-supported
                # tracks. This prevents a low-confidence model from creating
                # an unbounded candidate matrix; no proposal row is dropped.
                self.tracks.sort(key=lambda t: (t.miss_count, -t.hit_count, -t.score_ema, t.physical_track_id))
                self.tracks = self.tracks[:self.max_tracks]
            return out


    class CausalGeometryRuntime:
        """Model-free causal geometry/motion association control.

        This is a registered post-P1 family comparator, not a threshold sweep.
        It keeps every frozen Q0 proposal row and uses only a predicted box,
        centre distance, IoU, detector score and causal lifecycle state.  A
        global Hungarian assignment is accepted only for spatially plausible
        pairs; all remaining eligible detections become births.  No physical
        or semantic identifier is used in a feature tensor.
        """

        def __init__(self, max_miss: int = 8, max_tracks: int = 512, conservative: bool = False):
            from scipy.optimize import linear_sum_assignment
            self.max_miss = int(max_miss)
            self.max_tracks = int(max_tracks)
            self.conservative = bool(conservative)
            # One evidence-based repair after the first route: the initial
            # geometry comparator over-merged (403 vs Q0's 1026 tracks).  The
            # fixed tighter gates below are a single registered P2 contract,
            # not a sweep or held-event calibration.
            self.spatial_iou_gate = 0.05 if self.conservative else 0.02
            self.centre_gate = 0.12 if self.conservative else 0.20
            self._hungarian = linear_sum_assignment
            self.tracks: List[TrackState] = []
            self.next_id = 0

        def _score(self, det: Dict[str, object], track: TrackState, frame_id: int, image_size: Tuple[float, float]) -> Tuple[float, bool]:
            db = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            gap = max(0, min(32, int(frame_id) - int(track.last_frame)))
            ref = track.last_bbox + np.asarray(track.velocity, dtype=np.float32) * float(gap)
            iou = _box_iou(db, ref)
            iw, ih = float(image_size[0]), float(image_size[1])
            dc = np.asarray([(db[0] + db[2]) / (2.0 * iw), (db[1] + db[3]) / (2.0 * ih)])
            tc = np.asarray([(ref[0] + ref[2]) / (2.0 * iw), (ref[1] + ref[3]) / (2.0 * ih)])
            centre = float(np.linalg.norm(dc - tc))
            dw, dh = max(1e-4, (db[2] - db[0]) / iw), max(1e-4, (db[3] - db[1]) / ih)
            tw, th = max(1e-4, (ref[2] - ref[0]) / iw), max(1e-4, (ref[3] - ref[1]) / ih)
            shape = abs(float(np.log(dw / tw))) + abs(float(np.log(dh / th)))
            score = 2.0 * iou + float(np.exp(-centre / 0.08)) + 0.10 * float(det.get("base_score", 0.0)) + 0.05 * float(track.score_ema) - 0.05 * shape
            # Fixed spatial plausibility gate: IoU or a nearby predicted
            # centre is required; this is not selected from held events.
            plausible = bool(iou > self.spatial_iou_gate or centre <= self.centre_gate)
            return float(score), plausible

        def _score_matrix(self, dets: Sequence[Dict[str, object]], tracks: Sequence[TrackState], frame_id: int, image_size: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
            """Vectorized equivalent of ``_score`` for one frame."""
            if not tracks or not dets:
                return np.zeros((len(tracks), len(dets)), dtype=np.float32), np.zeros((len(tracks), len(dets)), dtype=bool)
            boxes = np.asarray([d["bbox_xyxy"] for d in dets], dtype=np.float32)
            last = np.asarray([t.last_bbox for t in tracks], dtype=np.float32)
            vel = np.asarray([t.velocity for t in tracks], dtype=np.float32)
            gaps = np.clip(np.asarray([int(frame_id) - int(t.last_frame) for t in tracks], dtype=np.float32), 0.0, 32.0)
            refs = last + vel * gaps[:, None]
            ix0 = np.maximum(refs[:, None, 0], boxes[None, :, 0]); iy0 = np.maximum(refs[:, None, 1], boxes[None, :, 1])
            ix1 = np.minimum(refs[:, None, 2], boxes[None, :, 2]); iy1 = np.minimum(refs[:, None, 3], boxes[None, :, 3])
            inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
            area_r = np.maximum(0.0, refs[:, 2] - refs[:, 0]) * np.maximum(0.0, refs[:, 3] - refs[:, 1])
            area_d = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
            union = area_r[:, None] + area_d[None, :] - inter
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            iw, ih = float(image_size[0]), float(image_size[1])
            dc = (boxes[:, :2] + boxes[:, 2:4]) / (2.0 * np.asarray([iw, ih], dtype=np.float32))
            tc = (refs[:, :2] + refs[:, 2:4]) / (2.0 * np.asarray([iw, ih], dtype=np.float32))
            centre = np.linalg.norm(dc[None, :, :] - tc[:, None, :], axis=-1)
            dw = np.maximum(1e-4, (boxes[:, 2] - boxes[:, 0]) / iw); dh = np.maximum(1e-4, (boxes[:, 3] - boxes[:, 1]) / ih)
            tw = np.maximum(1e-4, (refs[:, 2] - refs[:, 0]) / iw); th = np.maximum(1e-4, (refs[:, 3] - refs[:, 1]) / ih)
            shape = np.abs(np.log(dw[None, :] / tw[:, None])) + np.abs(np.log(dh[None, :] / th[:, None]))
            det_score = np.asarray([float(d.get("base_score", d.get("score", 0.0))) for d in dets], dtype=np.float32)
            tr_score = np.asarray([float(t.score_ema) for t in tracks], dtype=np.float32)
            score = 2.0 * iou + np.exp(-centre / 0.08) + 0.10 * det_score[None, :] + 0.05 * tr_score[:, None] - 0.05 * shape
            # Constructor-bound gates keep scalar and vector paths equivalent.
            plausible = (iou > self.spatial_iou_gate) | (centre <= self.centre_gate)
            return score.astype(np.float32), plausible

        def step(self, detections: Sequence[Dict[str, object]], frame_id: int, image_size: Tuple[float, float] = (640.0, 480.0)) -> List[Dict[str, object]]:
            dets = [dict(x) for x in detections]
            active = [t for t in self.tracks if t.miss_count <= self.max_miss]
            scores = np.zeros((len(active), len(dets)), dtype=np.float32)
            plausible = np.zeros_like(scores, dtype=bool)
            if active and dets:
                scores, plausible = self._score_matrix(dets, active, frame_id, image_size)
                rows, cols = self._hungarian(-scores)
                matches = {(int(r), int(c)) for r, c in zip(rows, cols) if plausible[r, c]}
            else:
                matches = set()
            by_det = {c: r for r, c in matches}
            out: List[Dict[str, object]] = []
            for j, det in enumerate(dets):
                if j in by_det:
                    track = active[by_det[j]]
                    assoc_score = float(scores[by_det[j], j])
                    previous_bbox = track.last_bbox.copy(); dt = max(1, int(frame_id) - int(track.last_frame)); current_bbox = np.asarray(det["bbox_xyxy"], dtype=np.float32)
                    delta = (current_bbox - previous_bbox) / float(dt)
                    track.velocity = 0.8 * np.asarray(track.velocity, dtype=np.float32) + 0.2 * delta
                    track.last_bbox = current_bbox; track.last_frame = int(frame_id); track.age += 1; track.hit_count += 1; track.miss_count = 0
                    track.score_ema = 0.8 * track.score_ema + 0.2 * float(det.get("base_score", det.get("score", 0.0)))
                    track.association_ema = 0.8 * track.association_ema + 0.2 * assoc_score
                    lifecycle, tid = "continuation", track.physical_track_id
                else:
                    track = TrackState(np.asarray(det["bbox_xyxy"], dtype=np.float32), int(frame_id), np.zeros(8, dtype=np.float32), float(det.get("base_score", det.get("score", 0.0))), physical_track_id=self.next_id)
                    self.next_id += 1; self.tracks.append(track); lifecycle, tid, assoc_score = "birth", track.physical_track_id, float(det.get("base_score", 0.0))
                row = dict(det); row.update({"assigned_track_slot": int(tid), "physical_track_id": int(tid), "association_score": assoc_score, "assignment_candidate_rank": int(det.get("candidate_rank", j)), "lifecycle_action": lifecycle, "track_age": int(track.age), "miss_count": int(track.miss_count), "learned_match_score": assoc_score, "geometry_match_score": assoc_score})
                out.append(row)
            matched_indices = {r for r, _ in matches}
            for r, track in enumerate(active):
                if r not in matched_indices:
                    track.miss_count += 1; track.age += 1
            self.tracks = [t for t in self.tracks if t.miss_count <= self.max_miss]
            if len(self.tracks) > self.max_tracks:
                self.tracks.sort(key=lambda t: (t.miss_count, -t.hit_count, -t.score_ema, t.physical_track_id))
                self.tracks = self.tracks[:self.max_tracks]
            return out
