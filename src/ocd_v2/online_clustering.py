"""Online, class-count-free clustering baselines and the OCD-v2
candidate-mature multi-prototype memory."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def cos(a, b):
    return float(np.dot(a, b))


class OnlineSphericalKMeans:
    """Continuous-cosine online k-means (single prototype per category)."""

    name = "spherical_kmeans"

    def __init__(self, attach_thr=0.6, ema=0.2, seed_offset=0):
        self.attach_thr = attach_thr
        self.ema = ema
        self.centers = {}
        self.counts = defaultdict(int)
        self.next_id = 100000 + seed_offset
        self.log = []

    def predict_one(self, x, sample_id, stream_order, num_frames=1, video_id=None):
        best_sim, best_id = -1.0, None
        for cid, c in self.centers.items():
            s = cos(x, c)
            if s > best_sim:
                best_sim, best_id = s, cid
        if best_sim >= self.attach_thr:
            self.centers[best_id] = (1 - self.ema) * self.centers[best_id] + self.ema * x
            self.centers[best_id] /= np.linalg.norm(self.centers[best_id]) + 1e-12
            self.counts[best_id] += 1
            vid = best_id
        else:
            vid = self.next_id
            self.next_id += 1
            self.centers[vid] = x.copy()
            self.counts[vid] = 1
        self.log.append(
            {"stream_order": stream_order, "sample_id": sample_id, "virtual_category_id": vid}
        )
        return vid

    def memory_stats(self):
        statuses = list(self.status.values())
        return {
            "categories": len(self.centers),
            "candidate": sum(s == "candidate" for s in statuses),
            "mature": sum(s == "mature" for s in statuses),
            "prototypes": len(self.centers),
            "prototypes_per_category_mean": 1.0 if self.centers else 0.0,
            "singleton_candidates": sum(
                1 for cid, cnt in self.counts.items() if cnt == 1 and self.status.get(cid) == "candidate"
            ),
        }


class OnlineDPMeans(OnlineSphericalKMeans):
    """Cosine DP-means: identical assign-or-create rule with a calibrated
    threshold (kept as a separate named baseline)."""

    name = "dpmeans"


class CandidateBuffer(OnlineSphericalKMeans):
    """Dual-threshold candidate buffer.

    - sim >= attach_thr: assign to (and mature) an existing category.
    - sim <= create_thr: create a candidate category.
    - in between: assign to the best existing candidate (never a mature-only
      hard assignment); the category remains candidate.
    """

    name = "candidate_buffer"

    def __init__(self, attach_thr=0.65, create_thr=0.45, ema=0.2, seed_offset=0, maturity_tracks=2):
        super().__init__(attach_thr=attach_thr, ema=ema, seed_offset=seed_offset)
        self.create_thr = create_thr
        self.maturity_tracks = maturity_tracks
        self.status = {}

    def predict_one(self, x, sample_id, stream_order, num_frames=1, video_id=None):
        best_sim, best_id = -1.0, None
        for cid, c in self.centers.items():
            s = cos(x, c)
            if s > best_sim:
                best_sim, best_id = s, cid
        if best_sim >= self.attach_thr:
            vid = best_id
            self.centers[vid] = (1 - self.ema) * self.centers[vid] + self.ema * x
            self.centers[vid] /= np.linalg.norm(self.centers[vid]) + 1e-12
            self.counts[vid] += 1
            if self.counts[vid] >= self.maturity_tracks:
                self.status[vid] = "mature"
            else:
                self.status.setdefault(vid, "candidate")
        elif best_sim <= self.create_thr or best_id is None:
            vid = self.next_id
            self.next_id += 1
            self.centers[vid] = x.copy()
            self.counts[vid] = 1
            self.status[vid] = "candidate"
        else:
            # uncertain band: attach to best candidate (existing category),
            # keep it candidate until maturity is reached
            vid = best_id
            self.centers[vid] = (1 - self.ema * 0.5) * self.centers[vid] + self.ema * 0.5 * x
            self.centers[vid] /= np.linalg.norm(self.centers[vid]) + 1e-12
            self.counts[vid] += 1
            if self.counts[vid] >= self.maturity_tracks:
                self.status[vid] = "mature"
            else:
                self.status[vid] = "candidate"
        self.log.append(
            {"stream_order": stream_order, "sample_id": sample_id, "virtual_category_id": vid}
        )
        return vid


class MultiPrototypeMemory:
    """OCD-v2 candidate-mature multi-prototype memory.

    Each global category can hold up to `max_proto` local prototypes. Category
    score is the max cosine over prototypes. Prototype updates use
    confidence-weighted EMA; a new local prototype is added only when the
    category already has support (>=2 tracks or >=2 videos) and the current
    sample is dissimilar to all existing prototypes but still above the
    `new_proto_thr` category score.
    """

    name = "ocd_v2_multiprototype"

    def __init__(
        self,
        attach_thr=0.62,
        create_thr=0.45,
        new_proto_thr=0.55,
        max_proto=4,
        ema=0.25,
        maturity_tracks=2,
        seed_offset=0,
    ):
        self.attach_thr = attach_thr
        self.create_thr = create_thr
        self.new_proto_thr = new_proto_thr
        self.max_proto = max_proto
        self.ema = ema
        self.maturity_tracks = maturity_tracks
        self.next_id = 200000 + seed_offset
        self.categories = {}
        self.log = []

    def _cat_score(self, x, cid):
        return max(cos(x, p) for p in self.categories[cid]["prototypes"])

    def predict_one(self, x, sample_id, stream_order, num_frames=1, video_id=None, score=1.0):
        best_sim, best_id = -1.0, None
        for cid in self.categories:
            s = self._cat_score(x, cid)
            if s > best_sim:
                best_sim, best_id = s, cid

        if best_sim >= self.attach_thr and best_id is not None:
            vid = best_id
            cat = self.categories[vid]
            # update nearest prototype with confidence-weighted EMA
            idx = int(np.argmax([cos(x, p) for p in cat["prototypes"]]))
            conf = best_sim
            w = self.ema * (0.5 + 0.5 * conf)
            cat["prototypes"][idx] = (1 - w) * cat["prototypes"][idx] + w * x
            cat["prototypes"][idx] /= np.linalg.norm(cat["prototypes"][idx]) + 1e-12
            cat["track_count"] += 1
            cat["confidence"] = 0.9 * cat["confidence"] + 0.1 * conf
            if video_id is not None:
                cat["videos"].add(video_id)
            cat["last_update"] = stream_order
            # add a local prototype if the category is already supported
            if (
                len(cat["prototypes"]) < self.max_proto
                and best_sim < self.new_proto_thr
                and (cat["track_count"] >= 2 or len(cat["videos"]) >= 2)
            ):
                cat["prototypes"].append(x.copy())
        elif best_sim <= self.create_thr or best_id is None:
            vid = self.next_id
            self.next_id += 1
            self.categories[vid] = {
                "prototypes": [x.copy()],
                "track_count": 1,
                "videos": {video_id} if video_id is not None else set(),
                "confidence": float(best_sim) if best_sim > 0 else 0.0,
                "last_update": stream_order,
                "status": "candidate",
            }
        else:
            # uncertain band: soft attach to best existing category
            vid = best_id
            cat = self.categories[vid]
            idx = int(np.argmax([cos(x, p) for p in cat["prototypes"]]))
            w = self.ema * 0.25
            cat["prototypes"][idx] = (1 - w) * cat["prototypes"][idx] + w * x
            cat["prototypes"][idx] /= np.linalg.norm(cat["prototypes"][idx]) + 1e-12
            cat["track_count"] += 1
            cat["confidence"] = 0.95 * cat["confidence"] + 0.05 * best_sim
            if video_id is not None:
                cat["videos"].add(video_id)
            cat["last_update"] = stream_order

        cat = self.categories[vid]
        if cat["status"] != "mature":
            if cat["track_count"] >= self.maturity_tracks and cat["confidence"] >= self.attach_thr:
                cat["status"] = "mature"
        self.log.append(
            {"stream_order": stream_order, "sample_id": sample_id, "virtual_category_id": vid}
        )
        return vid

    def memory_stats(self):
        statuses = [c["status"] for c in self.categories.values()]
        proto_counts = [len(c["prototypes"]) for c in self.categories.values()]
        return {
            "categories": len(self.categories),
            "candidate": sum(s == "candidate" for s in statuses),
            "mature": sum(s == "mature" for s in statuses),
            "prototypes": sum(proto_counts),
            "prototypes_per_category_mean": float(np.mean(proto_counts)) if proto_counts else 0.0,
            "singleton_candidates": sum(
                1 for c in self.categories.values() if c["track_count"] == 1 and c["status"] == "candidate"
            ),
        }
