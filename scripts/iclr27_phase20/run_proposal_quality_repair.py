#!/usr/bin/env python
"""One bounded Phase20 proposal-quality repair diagnostic.

The repair is deliberately small: a fold-local logistic quality head is
trained on public TRAIN rows only, using causal proposal/geometry fields.  It
does not create boxes, use GT-tight crops, alter the evaluator, or inspect
DEV+/Q1.  We report both the proxy quality-head coverage and the unchanged
true-IoU ceiling, so a better classifier cannot be mistaken for new
observations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OUT = ROOT / "outputs/iclr27_phase20"
PREFIXES = (1, 2, 4, 8, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def feature(r: dict[str, str]) -> list[float]:
    # All fields are available at the current causal row.  No category, track
    # identity, semantic ID, future frame, or GT-tight coordinate is included.
    names = ["score", "box_width_norm", "box_height_norm", "box_area_norm",
             "box_aspect_log", "border_left_norm", "border_top_norm",
             "border_right_norm", "border_bottom_norm", "causal_prefix_count",
             "causal_prefix_age_norm", "causal_box_stability_iou", "track_temporal_iou"]
    out = []
    for n in names:
        try: out.append(float(r.get(n, 0.0)))
        except (TypeError, ValueError): out.append(0.0)
    return out


def reliable(r: dict[str, str]) -> int:
    try: iou = float(r.get("row_iou", 0.0))
    except (TypeError, ValueError): iou = 0.0
    return int(str(r.get("assigned", "0")) == "1" and iou >= .5)


def main() -> None:
    with SRC.open(newline="") as f: rows = list(csv.DictReader(f))
    fold_manifest = json.loads((OUT / "manifests/fold_manifest.json").read_text())
    obs_events = json.loads((OUT / "audit/observability_events.json").read_text())
    positives = [r for r in obs_events if r["kind"] == "positive_existing" and int(r["causal_prefix_requested"]) == 1]
    pos_by_key = {str(r["event_key"]): r for r in positives}
    # Track rows in causal order.
    by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows: by_track[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(r)
    for k in by_track: by_track[k].sort(key=lambda r: (int(r.get("event_rank", 0)), int(r.get("frame_id", 0))))

    fold_results: dict[str, Any] = {}
    all_true = []; all_proxy = []; all_labels = []
    for fr in fold_manifest["folds"]:
        fold = int(fr["fold"]); fit_videos = {int(v) for v in fr.get("fit_videos", [])}; val_videos = {int(v) for v in fr.get("validation_videos", [])}
        # Fit only on rows from fold fit videos.  TRAIN metadata labels are
        # legal for this proposal/objectness repair; held event rows are never
        # used for fitting or selecting a classifier.
        fit_rows = [r for r in rows if int(r["video_id"]) in fit_videos and int(r.get("gt_category_id_common", -1)) >= 0]
        val_rows = [r for r in rows if int(r["video_id"]) in val_videos and int(r.get("gt_category_id_common", -1)) >= 0]
        X = np.asarray([feature(r) for r in fit_rows], np.float32); y = np.asarray([reliable(r) for r in fit_rows], np.int64)
        # A constant-label fold is legal but cannot fit a logistic model; use
        # the empirical rate and document it instead of silently dropping it.
        if len(np.unique(y)) < 2:
            rate = float(y.mean()) if len(y) else 0.0
            model = None
            val_prob = np.full(len(val_rows), rate, dtype=np.float64)
        else:
            model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=20260828)
            model.fit(X, y)
            val_prob = model.predict_proba(np.asarray([feature(r) for r in val_rows], np.float32))[:, 1]
        val_y = np.asarray([reliable(r) for r in val_rows], np.int64)
        try: val_auc = float(roc_auc_score(val_y, val_prob))
        except ValueError: val_auc = None
        try: val_ap = float(average_precision_score(val_y, val_prob))
        except ValueError: val_ap = None
        # Apply the frozen 0.5 quality decision to the event rows.  This is a
        # proxy only: true reliable status remains the IoU-based contract.
        event_rows = []
        for key, e in pos_by_key.items():
            if int(e["fold"]) != fold: continue
            sk, tk = str(e["source_tracklet_key"]), str(e["target_tracklet_key"])
            sr, tr = by_track.get(sk, []), by_track.get(tk, [])
            source_probs = model.predict_proba(np.asarray([feature(r) for r in sr], np.float32))[:, 1] if model is not None and sr else np.zeros(len(sr))
            target_probs = model.predict_proba(np.asarray([feature(r) for r in tr], np.float32))[:, 1] if model is not None and tr else np.zeros(len(tr))
            true_source = any(reliable(r) for r in sr); true_target = any(reliable(r) for r in tr[:16])
            proxy_source = bool(np.any(source_probs >= .5)); proxy_target = bool(np.any(target_probs[:16] >= .5))
            event_rows.append({"event_key": key, "fold": fold, "true_source_reliable": true_source, "true_target_reliable_prefix16": true_target,
                               "proxy_source_quality_prefix": proxy_source, "proxy_target_quality_prefix16": proxy_target,
                               "true_ceiling": bool(true_source and true_target), "proxy_ceiling": bool(proxy_source and proxy_target),
                               "source_rows": len(sr), "target_rows": len(tr)})
            all_true.append(int(true_source and true_target)); all_proxy.append(int(proxy_source and proxy_target)); all_labels.append(1)
        fold_results[str(fold)] = {"fit_rows": len(fit_rows), "validation_rows": len(val_rows), "fit_reliable_rate": float(y.mean()) if len(y) else 0.0,
                                   "validation_reliable_rate": float(val_y.mean()) if len(val_y) else 0.0,
                                   "validation_roc_auc": val_auc, "validation_pr_auc": val_ap,
                                   "quality_threshold": .5, "event_count": len(event_rows),
                                   "event_true_ceiling": int(sum(x["true_ceiling"] for x in event_rows)),
                                   "event_proxy_ceiling": int(sum(x["proxy_ceiling"] for x in event_rows)),
                                   "event_rows": event_rows}

    true_count = int(sum(all_true)); proxy_count = int(sum(all_proxy)); denom = len(all_true)
    result = {"protocol": "trackocd_iclr27_phase20_stage0_proposal_quality_repair",
              "source_rows": len(rows), "source_rows_path": str(SRC), "source_rows_sha256": sha256(SRC),
              "feature_fields": ["score", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log", "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_prefix_count", "causal_prefix_age_norm", "causal_box_stability_iou", "track_temporal_iou"],
              "method": "fold-local logistic proposal-quality head; threshold=0.5; public TRAIN rows only",
              "folds": fold_results, "positive_event_denominator": denom,
              "true_iou_ceiling_at_prefix16": {"correct": true_count, "eligible": denom, "recall": true_count / max(denom, 1)},
              "quality_proxy_ceiling_at_prefix16": {"correct": proxy_count, "eligible": denom, "recall": proxy_count / max(denom, 1)},
              "proposal_presence_changed": False, "true_iou_changed": False,
              "interpretation": "quality head can reprioritize existing rows but cannot create a proposal or alter stored IoU; true ceiling is the decision quantity",
              "repair_gate_pass": bool(true_count / max(denom, 1) >= .5),
              "labels_used": "public TRAIN category/video metadata only", "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}
    atomic_json(OUT / "audit/proposal_quality_repair.json", result)
    atomic_json(OUT / "completion/proposal_quality_repair.done", {"stage": "proposal_quality_repair", "true_ceiling": true_count, "denominator": denom, "repair_gate_pass": result["repair_gate_pass"]})
    print(json.dumps({"true_ceiling": [true_count, denom], "proxy_ceiling": [proxy_count, denom], "repair_gate_pass": result["repair_gate_pass"]}, indent=2))


if __name__ == "__main__": main()
