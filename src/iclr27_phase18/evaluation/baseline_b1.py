"""B1: frozen DINOv2 causal tracklet prototype baseline.

The held-fold path never reads GT category/IoU as an input.  GT fields are
kept in a separate evaluator-side record for fixed denominators and scoring.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/iclr27_phase18"
ROWS_PATH = ROOT / "data/iclr27_phase18/sources/public_rows_corrected.csv"
FEATURE_PATH = ROOT / "data/iclr27_phase18/sources/public_dinov2_cls_roi.npz"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_data() -> dict[str, Any]:
    with ROWS_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = list(csv.DictReader((OUT / "manifests/row_aligned_tracklet_manifest.csv").open(newline="")))
    assert len(rows) == len(manifest) == 43423
    z = np.load(FEATURE_PATH, mmap_mode="r")
    cls = np.asarray(z["cls"], np.float32)
    roi = np.asarray(z["roi"], np.float32)
    d2_index = np.asarray([int(x["dinov2_index"]) for x in manifest], np.int64)
    assert all(r["row_key"] == m["row_key"] for r, m in zip(rows, manifest))
    feature = l2(.8 * cls[d2_index] + .2 * roi[d2_index])
    tracklets = {x["tracklet_key"]: x for x in load_jsonl(OUT / "manifests/tracklet_manifest.jsonl")}
    folds = json.loads((OUT / "manifests/fold_manifest.json").read_text())
    positives = load_jsonl(OUT / "episodes/identifiable_positive_events.jsonl")
    negatives = load_jsonl(OUT / "episodes/identifiable_negative_events.jsonl")
    supported_known = json.loads((ROOT / "data/iclr27_phase18/sources/supported_known_ids.json").read_text())
    for i, r in enumerate(rows):
        r["idx"] = i
        r["video_i"] = int(r["video_id"])
        r["track_i"] = int(r["track_id"])
        r["cat_i"] = int(r["gt_category_id_common"])
        r["iou_f"] = float(r["row_iou"])
        r["reliable"] = r["assigned"] == "1" and r["iou_f"] >= .5
    readiness = np.asarray([
        .62 * float(r["score"])
        + .23 * float(r["causal_box_stability_iou"])
        + .15 * min(1.0, math.log1p(int(r["causal_prefix_count"])) / math.log(5.0))
        for r in rows
    ], np.float32)
    return {
        "rows": rows, "feature": feature, "tracklets": tracklets, "folds": folds,
        "positives": positives, "negatives": negatives,
        "supported_known": [int(x) for x in supported_known], "readiness": readiness,
    }


def threshold_curve(scores: np.ndarray, labels: np.ndarray, objective: str = "f1") -> tuple[float, list[dict[str, float]]]:
    assert len(scores) and len(set(labels.tolist())) == 2
    grid = np.unique(np.concatenate([np.linspace(float(scores.min()), float(scores.max()), 81), np.quantile(scores, np.linspace(0, 1, 101))]))
    curve = []
    for t in grid:
        p = scores >= t
        tp = int((p & labels).sum()); fp = int((p & ~labels).sum())
        fn = int((~p & labels).sum()); tn = int((~p & ~labels).sum())
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        bal = .5 * (recall + tn / max(tn + fp, 1))
        value = f1 if objective == "f1" else bal
        curve.append({"threshold": float(t), "precision": precision, "recall": recall,
                      "f1": f1, "balanced_accuracy": bal, "objective": value})
    curve.sort(key=lambda x: (-x["objective"], -x["precision"], x["threshold"]))
    return curve[0]["threshold"], curve


class B1Policy:
    def __init__(self, data: dict[str, Any], fold: dict[str, Any]):
        self.data = data
        self.rows = data["rows"]
        self.feature = data["feature"]
        self.tracklets = data["tracklets"]
        self.readiness = data["readiness"]
        self.fold = fold
        self.held_videos = set(fold["strict_excluded_videos"])
        self.held_cats = set(fold["held_categories"])
        self.cal_cats = set(fold["nested_calibration_categories"])
        self.known_ids = data["supported_known"]
        self.known_proto = self._known_prototypes()
        self.tau_ready, self.ready_curve = self._calibrate_readiness()
        self.tau_pair, self.pair_curve = self._calibrate_pair()
        self.tau_known, self.known_curve = self._calibrate_known()

    def _allowed_fit_row(self, i: int) -> bool:
        r = self.rows[i]
        return (r["role17"] in {"known_bank", "novel_correspondence_train"}
                and r["video_i"] not in self.held_videos
                and r["cat_i"] not in self.held_cats | self.cal_cats)

    def _known_prototypes(self) -> dict[int, np.ndarray]:
        by: dict[int, list[np.ndarray]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            if (self._allowed_fit_row(i) and r["gt_role_common"] == "supported_known"
                    and r["reliable"] and r["cat_i"] in self.known_ids):
                by[r["cat_i"]].append(self.feature[i])
        return {c: l2(np.mean(v, axis=0, keepdims=True))[0] for c, v in by.items()}

    def _cal_tracklets(self) -> list[dict[str, Any]]:
        return [x for x in self.tracklets.values()
                if x["label_category_gt_only"] in self.cal_cats
                and x["video_id"] not in self.held_videos
                and x["first_reliable_prefix_index_gt_only"] >= 0]

    def _calibrate_readiness(self) -> tuple[float, list[dict[str, float]]]:
        indices = sorted({i for t in self._cal_tracklets() for i in t["row_indices"]})
        # Add role-frozen calibration false positives without using held videos.
        indices += [i for i, r in enumerate(self.rows)
                    if r["role17"] in {"known_calibration", "novel_calibration"}
                    and r["gt_role_common"] == "fp" and r["video_i"] not in self.held_videos]
        indices = np.asarray(sorted(set(indices)), np.int64)
        labels = np.asarray([self.rows[int(i)]["reliable"] for i in indices], bool)
        return threshold_curve(self.readiness[indices], labels, "f1")

    def aggregate(self, indices: list[int], upto: int | None = None) -> np.ndarray:
        use = indices if upto is None else indices[: upto + 1]
        w = np.maximum(self.readiness[use], .02)
        return l2(np.average(self.feature[use], axis=0, weights=w)[None])[0]

    def _calibrate_pair(self) -> tuple[float, list[dict[str, float]]]:
        tracks = self._cal_tracklets()
        emb = [self.aggregate(t["row_indices"]) for t in tracks]
        scores, labels = [], []
        for i, a in enumerate(tracks):
            for j in range(i + 1, len(tracks)):
                b = tracks[j]
                if a["video_id"] == b["video_id"]:
                    continue
                scores.append(float(emb[i] @ emb[j]))
                labels.append(a["label_category_gt_only"] == b["label_category_gt_only"])
        return threshold_curve(np.asarray(scores, np.float32), np.asarray(labels, bool), "f1")

    def _calibrate_known(self) -> tuple[float, list[dict[str, float]]]:
        cats = sorted(self.known_proto)
        proto = np.asarray([self.known_proto[c] for c in cats], np.float32)
        scores, labels = [], []
        for i, r in enumerate(self.rows):
            if r["video_i"] in self.held_videos or not r["reliable"]:
                continue
            is_cal_known = r["role17"] == "known_calibration" and r["gt_role_common"] == "supported_known"
            is_nested_novel = r["cat_i"] in self.cal_cats and r["gt_role_common"] == "novel"
            if not (is_cal_known or is_nested_novel):
                continue
            sims = proto @ self.feature[i]
            j = int(np.argmax(sims))
            scores.append(float(sims[j]))
            labels.append(bool(is_cal_known and cats[j] == r["cat_i"]))
        return threshold_curve(np.asarray(scores, np.float32), np.asarray(labels, bool), "balanced")

    def known(self, query: np.ndarray) -> tuple[int, float]:
        cats = sorted(self.known_proto)
        proto = np.asarray([self.known_proto[c] for c in cats], np.float32)
        sims = proto @ query
        j = int(np.argmax(sims))
        return cats[j], float(sims[j])

    def pair_score(self, query: np.ndarray, prototype: np.ndarray) -> float:
        """Pair similarity hook; B1 is the frozen cosine scorer."""
        return float(query @ prototype)

    def calibration_summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold["fold"], "tau_ready": self.tau_ready,
            "tau_pair": self.tau_pair, "tau_known": self.tau_known,
            "known_prototype_categories": len(self.known_proto),
            "readiness_selected": self.ready_curve[0],
            "pair_selected": self.pair_curve[0],
            "known_selected": self.known_curve[0],
            "selected_from_nested_calibration_only": True,
        }


def simulate_event(policy: B1Policy, event: dict[str, Any]) -> dict[str, Any]:
    data = policy.data; tracks = policy.tracklets; rows = policy.rows
    memory: dict[int, dict[str, Any]] = {}
    local: dict[str, int] = {}
    next_sid = 100000
    merge_count = 0

    def process_track(track_key: str, evaluator_category: int, phase: str) -> list[dict[str, Any]]:
        nonlocal next_sid, merge_count
        t = tracks[track_key]; decisions = []
        for pos, idx in enumerate(t["row_indices"]):
            q = policy.aggregate(t["row_indices"], pos)
            ready_score = float(policy.readiness[idx]); ready = ready_score >= policy.tau_ready
            action = "DEFER"; sid = None; evidence = "local_buffer_only"
            if not ready and track_key in local:
                sid = local[track_key]; action = "EXISTING_NOVEL"; evidence = "inherited_local_belief"
            elif ready:
                known_id, known_score = policy.known(q)
                if known_score >= policy.tau_known:
                    action, sid, evidence = "KNOWN", known_id, "known_prototype"
                else:
                    candidates = [(s, m) for s, m in memory.items()
                                  if m["birth_tracklet"] != track_key and m["birth_video"] != t["video_id"]]
                    best_sid, best_sim = None, -2.0
                    for s, m in candidates:
                        p = l2(np.mean(m["anchors"], axis=0, keepdims=True))[0]
                        sim = float(policy.pair_score(q, p))
                        if sim > best_sim:
                            best_sid, best_sim = s, sim
                    if best_sid is not None and best_sim >= policy.tau_pair:
                        previous = local.get(track_key)
                        sid = best_sid; action = "EXISTING_NOVEL"; evidence = "different_video_prototype"
                        if previous is not None and previous != sid:
                            memory[previous]["aliased_to"] = sid
                            merge_count += 1
                        local[track_key] = sid
                    elif track_key in local:
                        sid = local[track_key]; action = "EXISTING_NOVEL"; evidence = "retained_local_belief"
                    else:
                        sid = next_sid; next_sid += 1
                        memory[sid] = {
                            "anchors": [], "birth_tracklet": track_key, "birth_video": t["video_id"],
                            "eval_category_not_model_input": evaluator_category, "aliased_to": None,
                        }
                        local[track_key] = sid
                        action, evidence = "NEW_NOVEL", "novel_birth"
                if action in {"NEW_NOVEL", "EXISTING_NOVEL"} and sid in memory:
                    memory[sid]["anchors"].append(q.copy())
                    memory[sid]["anchors"] = memory[sid]["anchors"][-8:]
            decisions.append({
                "row_key": rows[idx]["row_key"], "tracklet_position": pos, "phase": phase,
                "action": action, "semantic_id": sid, "readiness_score": ready_score,
                "predicted_ready": ready, "evidence": evidence,
            })
        return decisions

    source_eval_cat = (event.get("category_gt_denominator_only")
                       if event["kind"] == "positive_existing"
                       else event["distractor_category_gt_denominator_only"])
    source_decisions = []
    for key in event["source_tracklet_keys"]:
        source_decisions.extend(process_track(key, int(source_eval_cat), "source"))
    target_cat = int(event.get("category_gt_denominator_only", event.get("target_category_gt_denominator_only")))
    target = process_track(event["target_tracklet_key"], target_cat, "target")
    prefix = int(event["target_first_reliable_prefix_index_gt_only"])

    def correct_existing(d: dict[str, Any]) -> bool:
        if d["action"] != "EXISTING_NOVEL":
            return False
        m = memory.get(int(d["semantic_id"]))
        return bool(m and m["eval_category_not_model_input"] == target_cat
                    and m["birth_video"] != int(event["target_video"])
                    and m["birth_tracklet"] != event["target_tracklet_key"])

    post = target[prefix:]
    first_commit_offset = next((i for i, d in enumerate(post) if d["action"] != "DEFER"), None)
    first_commit = post[first_commit_offset] if first_commit_offset is not None else None
    correct_offsets = [i for i, d in enumerate(post) if correct_existing(d)]
    premature = [d for d in target[:prefix] if d["action"] != "DEFER"]
    return {
        "event_key": event["event_key"], "kind": event["kind"], "fold": event["fold"],
        "target_category_gt_evaluator_only": target_cat,
        "source_decisions": source_decisions, "target_decisions": target,
        "first_commit_after_prefix": first_commit,
        "first_commit_offset": first_commit_offset,
        "first_commit_correct_existing": bool(first_commit and correct_existing(first_commit)),
        "post_prefix_correct_existing_rows": len(correct_offsets),
        "post_prefix_rows": len(post),
        "time_to_correct_commit": min(correct_offsets) if correct_offsets else None,
        "pre_prefix_rows": prefix,
        "pre_prefix_defer_rows": prefix - len(premature),
        "premature_commit": bool(premature),
        "unresolved_after_prefix": first_commit is None,
        "state_count": len(memory), "merge_count": merge_count,
        "duplicate_target_births": sum(
            m["eval_category_not_model_input"] == target_cat and m["birth_video"] == int(event["target_video"])
            for m in memory.values()
        ),
    }


def event_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [x for x in records if x["kind"] == "positive_existing"]
    neg = [x for x in records if x["kind"] == "negative_new"]
    all_commits = [x for x in records if x["first_commit_after_prefix"] is not None]
    existing = [x for x in all_commits if x["first_commit_after_prefix"]["action"] == "EXISTING_NOVEL"]
    correct_existing = [x for x in existing if x["first_commit_correct_existing"]]
    false_merge = [x for x in neg if x["first_commit_after_prefix"] is not None
                   and x["first_commit_after_prefix"]["action"] == "EXISTING_NOVEL"]
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for x in pos:
        by_cat[x["target_category_gt_evaluator_only"]].append(x)
        event = x["event_key"].split(":")
        target_video = int(next(v[2:] for v in event if v.startswith("tv")))
        by_video[target_video].append(x)
    latency = [x["time_to_correct_commit"] for x in pos if x["time_to_correct_commit"] is not None]
    return {
        "commit_ct": {
            "correct": sum(x["first_commit_correct_existing"] for x in pos),
            "eligible": len(pos),
            "recall": sum(x["first_commit_correct_existing"] for x in pos) / max(len(pos), 1),
        },
        "post_prefix_ct": {
            "correct_rows": sum(x["post_prefix_correct_existing_rows"] for x in pos),
            "rows": sum(x["post_prefix_rows"] for x in pos),
            "recall": sum(x["post_prefix_correct_existing_rows"] for x in pos) / max(sum(x["post_prefix_rows"] for x in pos), 1),
        },
        "existing_precision": len(correct_existing) / max(len(existing), 1),
        "first_commit_existing_count": len(existing),
        "negative_false_merge_rate": len(false_merge) / max(len(neg), 1),
        "negative_false_merges": len(false_merge),
        "mean_time_to_correct_commit": float(np.mean(latency)) if latency else None,
        "median_time_to_correct_commit": float(np.median(latency)) if latency else None,
        "pre_prefix_defer_rate": sum(x["pre_prefix_defer_rows"] for x in records) / max(sum(x["pre_prefix_rows"] for x in records), 1),
        "premature_commit_event_rate": sum(x["premature_commit"] for x in records) / max(len(records), 1),
        "unresolved_event_rate": sum(x["unresolved_after_prefix"] for x in records) / max(len(records), 1),
        "mean_state_count": float(np.mean([x["state_count"] for x in records])),
        "merge_count": sum(x["merge_count"] for x in records),
        "duplicate_target_births": sum(x["duplicate_target_births"] for x in records),
        "correct_categories": sum(any(x["first_commit_correct_existing"] for x in v) for v in by_cat.values()),
        "correct_target_videos": sum(any(x["first_commit_correct_existing"] for x in v) for v in by_video.values()),
        "by_category": {str(c): {"correct": sum(x["first_commit_correct_existing"] for x in v), "eligible": len(v),
                                  "recall": sum(x["first_commit_correct_existing"] for x in v) / len(v)} for c, v in sorted(by_cat.items())},
        "by_target_video": {str(c): {"correct": sum(x["first_commit_correct_existing"] for x in v), "eligible": len(v),
                                    "recall": sum(x["first_commit_correct_existing"] for x in v) / len(v)} for c, v in sorted(by_video.items())},
    }


def known_metrics(policy: B1Policy) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(policy.rows):
        if r["role17"] == "known_audit" and r["gt_role_common"] == "supported_known":
            groups[f"v{r['video_i']}:p{r['track_i']}"].append(i)
    correct, top1, labels, scores = [], [], [], []
    by: dict[int, list[int]] = defaultdict(list)
    for key, idx in sorted(groups.items()):
        idx.sort(key=lambda i: int(policy.rows[i]["event_rank"]))
        local_known = None
        for pos, i in enumerate(idx):
            q = policy.aggregate(idx, pos)
            pred, sim = policy.known(q)
            is_ready = policy.readiness[i] >= policy.tau_ready
            if is_ready and sim >= policy.tau_known:
                local_known = pred
            action_pred = local_known if local_known is not None else -1
            ok = int(action_pred == policy.rows[i]["cat_i"])
            closed = int(pred == policy.rows[i]["cat_i"])
            correct.append(ok); top1.append(closed); by[policy.rows[i]["cat_i"]].append(ok)
            labels.append(policy.rows[i]["reliable"]); scores.append(float(policy.readiness[i]))
    return {
        "rows": len(correct), "micro_accuracy": float(np.mean(correct)),
        "category_macro_accuracy": float(np.mean([np.mean(v) for v in by.values()])),
        "closed_top1": float(np.mean(top1)), "categories": len(by),
        "by_category": {str(c): {"correct": sum(v), "rows": len(v), "accuracy": float(np.mean(v))} for c, v in sorted(by.items())},
    }


def reliability_metrics(scores: list[float], labels: list[bool], thresholds: list[float]) -> dict[str, Any]:
    s = np.asarray(scores, np.float32); y = np.asarray(labels, bool)
    # Fold-specific thresholds are already applied for precision/recall.
    pred = np.asarray([score >= threshold for score, threshold in zip(scores, thresholds)], bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    return {
        "rows": len(y), "positive_rows": int(y.sum()),
        "auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
        "precision_at_fold_threshold": tp / max(tp + fp, 1),
        "recall_at_fold_threshold": tp / max(tp + fn, 1),
    }


def run() -> dict[str, Any]:
    data = load_data()
    policies = {f["fold"]: B1Policy(data, f) for f in data["folds"]["folds"]}
    records = []
    for event in data["positives"] + data["negatives"]:
        records.append(simulate_event(policies[event["fold"]], event))
    held_scores, held_labels, held_thresholds, seen_rows = [], [], [], set()
    for event in data["positives"]:
        policy = policies[event["fold"]]
        track = data["tracklets"][event["target_tracklet_key"]]
        for i in track["row_indices"]:
            if i in seen_rows:
                continue
            seen_rows.add(i); held_scores.append(float(data["readiness"][i]))
            held_labels.append(bool(data["rows"][i]["reliable"])); held_thresholds.append(policy.tau_ready)
    known = [known_metrics(policies[f]) for f in sorted(policies)]
    result = {
        "protocol": "trackocd_iclr27_phase18_B1_dinov2_causal_tracklet_prototype",
        "representation": "L2(0.8 * DINOv2 CLS + 0.2 * ROI)",
        "readiness_formula": "0.62*proposal_score + 0.23*causal_box_stability_iou + 0.15*min(1,log1p(prefix_count)/log(5))",
        "held_gt_used_as_model_input": False,
        "physical_id_used_as_semantic_value": False,
        "calibration": [policies[f].calibration_summary() for f in sorted(policies)],
        "metrics": event_metrics(records),
        "known": {
            "per_fold": known,
            "micro_accuracy_mean": float(np.mean([x["micro_accuracy"] for x in known])),
            "category_macro_accuracy_mean": float(np.mean([x["category_macro_accuracy"] for x in known])),
            "closed_top1_mean": float(np.mean([x["closed_top1"] for x in known])),
        },
        "reliability": reliability_metrics(held_scores, held_labels, held_thresholds),
        "event_records": records,
    }
    atomic_json(OUT / "eval/b1_prereg_baseline.json", result)
    print(json.dumps({"calibration": result["calibration"], "metrics": result["metrics"],
                      "known": result["known"], "reliability": result["reliability"]}, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
