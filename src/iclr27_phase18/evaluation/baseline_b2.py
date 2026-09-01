"""B2: category-disjoint learned DINOv2 tracklet pair scorer.

The state machine is exactly B1's causal prototype controller.  Only the
cross-video tracklet similarity is replaced by a fitted logistic pair scorer;
fit pairs come from allowed training roles and never from held/calibration
categories or held videos.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.iclr27_phase18.evaluation.baseline_b1 import (
    B1Policy, atomic_json, event_metrics, known_metrics, load_data,
    load_jsonl, reliability_metrics, simulate_event, threshold_curve,
)


def pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return np.asarray([
        float(a @ b),
        float(np.mean(np.abs(a - b))),
        float(np.linalg.norm(a - b)),
        float(np.mean(a * b)),
    ], np.float32)


class B2Policy(B1Policy):
    """B1 controller with a learned, fold-local pair probability."""

    def __init__(self, data: dict[str, Any], fold: dict[str, Any]):
        self.pair_model: LogisticRegression | None = None
        self.pair_fit_summary: dict[str, Any] = {}
        super().__init__(data, fold)

    def _fit_tracks(self) -> list[dict[str, Any]]:
        held_videos = set(self.fold["strict_excluded_videos"])
        excluded = set(self.fold["held_categories"]) | set(self.fold["nested_calibration_categories"])
        tracks = []
        for t in self.tracklets.values():
            c = int(t["label_category_gt_only"])
            if t["label_role_gt_only"] != "novel" or int(t["first_reliable_prefix_index_gt_only"]) < 0:
                continue
            if int(t["video_id"]) in held_videos or c in excluded:
                continue
            if not all(self._allowed_fit_row(int(i)) for i in t["row_indices"]):
                continue
            tracks.append(t)
        return sorted(tracks, key=lambda x: x["tracklet_key"])

    def _fit_pair_model(self) -> None:
        tracks = self._fit_tracks()
        embeddings = {t["tracklet_key"]: self.aggregate(t["row_indices"]) for t in tracks}
        positives: list[tuple[dict[str, Any], dict[str, Any]]] = []
        negatives: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                if int(a["video_id"]) == int(b["video_id"]):
                    continue
                if int(a["label_category_gt_only"]) == int(b["label_category_gt_only"]):
                    positives.append((a, b))
                else:
                    negatives.append((a, b))
        # Use every available positive and an equal deterministic negative set;
        # this is complete over the rare positive population without letting
        # the much larger negative cross-product dominate the fit.
        rng = np.random.default_rng(1800 + int(self.fold["fold"]))
        if len(negatives) > len(positives):
            negatives = [negatives[int(i)] for i in rng.choice(len(negatives), len(positives), replace=False)]
        pairs = positives + negatives
        x = np.asarray([pair_features(embeddings[a["tracklet_key"]], embeddings[b["tracklet_key"]]) for a, b in pairs], np.float32)
        y = np.asarray([1] * len(positives) + [0] * len(negatives), np.int64)
        if len(set(y.tolist())) < 2:
            raise RuntimeError(f"B2 fold {self.fold['fold']} has no positive/negative pair diversity")
        self.pair_model = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=1800 + int(self.fold["fold"]))
        self.pair_model.fit(x, y)
        self.pair_fit_summary = {
            "fit_tracklets": len(tracks), "fit_categories": len({int(t["label_category_gt_only"]) for t in tracks}),
            "positive_pairs": len(positives), "negative_pairs": len(negatives),
            "held_categories_excluded": True, "nested_calibration_categories_excluded": True,
            "held_videos_excluded": True,
        }

    def _calibrate_pair(self) -> tuple[float, list[dict[str, float]]]:
        self._fit_pair_model()
        tracks = self._cal_tracklets()
        emb = [self.aggregate(t["row_indices"]) for t in tracks]
        scores, labels = [], []
        for i, a in enumerate(tracks):
            for j in range(i + 1, len(tracks)):
                b = tracks[j]
                if int(a["video_id"]) == int(b["video_id"]):
                    continue
                scores.append(self.pair_score(emb[i], emb[j]))
                labels.append(int(a["label_category_gt_only"]) == int(b["label_category_gt_only"]))
        return threshold_curve(np.asarray(scores, np.float32), np.asarray(labels, bool), "f1")

    def pair_score(self, query: np.ndarray, prototype: np.ndarray) -> float:
        assert self.pair_model is not None
        f = pair_features(query, prototype)[None]
        return float(self.pair_model.predict_proba(f)[0, 1])


def run() -> dict[str, Any]:
    data = load_data()
    policies = {int(f["fold"]): B2Policy(data, f) for f in data["folds"]["folds"]}
    events = data["positives"] + data["negatives"]
    records = [simulate_event(policies[int(e["fold"])], e) for e in events]
    held_scores, held_labels, thresholds, seen = [], [], [], set()
    for event in data["positives"]:
        policy = policies[int(event["fold"])]
        for i in data["tracklets"][event["target_tracklet_key"]]["row_indices"]:
            if int(i) in seen:
                continue
            seen.add(int(i)); held_scores.append(float(data["readiness"][int(i)])); held_labels.append(bool(data["rows"][int(i)]["reliable"])); thresholds.append(policy.tau_ready)
    known = [known_metrics(policies[f]) for f in sorted(policies)]
    result = {
        "protocol": "trackocd_iclr27_phase18_B2_learned_dinov2_tracklet_pair_scorer",
        "representation": "L2(0.8 * DINOv2 CLS + 0.2 * ROI), causal readiness-weighted tracklet aggregation",
        "pair_model": "fold-local balanced logistic regression on [cosine, mean_abs_difference, l2_difference, mean_product]",
        "held_gt_used_as_model_input": False, "physical_id_used_as_semantic_value": False,
        "fit_summary": [policies[f].pair_fit_summary for f in sorted(policies)],
        "calibration": [{"fold": policies[f].fold["fold"], "tau_ready": policies[f].tau_ready, "tau_pair": policies[f].tau_pair, "tau_known": policies[f].tau_known, "pair_selected": policies[f].pair_curve[0], "selected_from_nested_calibration_only": True} for f in sorted(policies)],
        "metrics": event_metrics(records),
        "known": {"per_fold": known, "micro_accuracy_mean": float(np.mean([x["micro_accuracy"] for x in known])), "category_macro_accuracy_mean": float(np.mean([x["category_macro_accuracy"] for x in known])), "closed_top1_mean": float(np.mean([x["closed_top1"] for x in known]))},
        "reliability": reliability_metrics(held_scores, held_labels, thresholds),
        "event_records": records,
    }
    atomic_json(data_path := (B2Policy.__mro__[1].__module__ and __import__('pathlib').Path(__file__).resolve().parents[3] / 'outputs/iclr27_phase18/eval/b2_prereg_baseline.json'), result)
    print(json.dumps({"fit_summary": result["fit_summary"], "metrics": result["metrics"], "known": result["known"], "reliability": result["reliability"]}, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
