"""GT-pilot adapters; the recovered B1/B2 source remains byte-identical.

Only visual PrefixView and legal Known prototypes reach these policies.
No category supervision, GT role/matches or evaluator mapping is accepted.
"""
from __future__ import annotations

from collections import Counter
from copy import copy

from src.trackocd_core.features import PrefixView
from src.trackocd_v2.methods.dpmeans import OnlineDPMeans
from src.trackocd_v2.methods.nearest_prototype import NearestPrototype


class FrameSnapshotNearestVoting(NearestPrototype):
    """Frame nearest against pre-track memory, vote, then one causal update.

    Provisional NEWs never enter memory or leak unvoted tokens. This is not
    a claim of a continuous frame-online discovery controller.
    """

    def step_prefix(self, view: PrefixView) -> dict:
        votes, first = Counter(), {}
        for vector in view.visual:
            snapshot = copy(self)
            snapshot._anonymous = self._anonymous.copy()
            action = snapshot.step(vector)
            choice = (action["kind"], action.get("category_id") if action["kind"] == "KNOWN"
                      else action.get("token") if action["kind"] == "EXISTING" else None)
            votes[choice] += 1
            first.setdefault(choice, action)
        choice = max(votes, key=votes.get)  # Insertion order = earliest chronological vote.
        action = dict(first[choice])
        mean = view.weighted_mean()
        if action["kind"] == "EXISTING":
            self._update(action["token"], mean)
        elif action["kind"] == "NEW":
            token = f"A:{self._next_id}"
            self._next_id += 1
            self._anonymous[token] = (mean, 1)
            action["token"] = token
        action["winning_frame_votes"] = votes[choice]
        return action


def make_baseline(name: str, prototypes: dict, thresholds: dict):
    if name == "B0_frame_snapshot_vote":
        return FrameSnapshotNearestVoting(known_prototypes=prototypes,
                                          tau_known=thresholds["known_cosine_threshold"],
                                          tau_existing=thresholds["existing_cosine_threshold"])
    if name == "B1_track_nearest":
        return NearestPrototype(known_prototypes=prototypes,
                                tau_known=thresholds["known_cosine_threshold"],
                                tau_existing=thresholds["existing_cosine_threshold"])
    if name == "B2_track_dpmeans":
        return OnlineDPMeans(known_prototypes=prototypes,
                             tau_known=thresholds["dpmeans_known_distance_threshold"],
                             lambda_distance=thresholds["dpmeans_lambda_distance"])
    raise ValueError("No registered/compatible baseline: " + name)


def decide_prefix(model, view: PrefixView) -> dict:
    if isinstance(model, FrameSnapshotNearestVoting):
        return model.step_prefix(view)
    return model.step(view.weighted_mean())


def anonymous_memory_size(model) -> int:
    return len(model._centroids if isinstance(model, OnlineDPMeans) else model._anonymous)
