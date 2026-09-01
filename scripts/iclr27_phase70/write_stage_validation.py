#!/usr/bin/env python3
"""Write the immutable Phase70 checkpoint validation gate artifact.

This is a post-hoc, TRAIN/validation-only report writer.  It never reads the
held-event labels as model input and deliberately records retrieval/controller
stages as blocked when the physical proposal sanity gate fails.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/iclr27_phase70/validation/joint_d_repair1"
OUT = BASE / "step_5000_metrics.json"
DONE = BASE / "step_5000.done"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    aggregate_path = BASE / "validation_aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    q0_recall_path = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/proposal_recall.json"
    q0_trackeval_path = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval_aggregate.json"
    q0_recall = json.loads(q0_recall_path.read_text())
    q0_trackeval = json.loads(q0_trackeval_path.read_text())

    checkpoint_steps = {0: 4000, 1: 4000, 2: 5000, 3: 4000}
    checkpoints = []
    for fold, steps in checkpoint_steps.items():
        path = ROOT / f"outputs/iclr27_phase70/checkpoints/joint_d_repair1_f{fold}/checkpoint.pth"
        checkpoints.append({
            "fold": fold,
            "steps": steps,
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256(path) if path.is_file() else None,
            "completion_marker": str(ROOT / f"outputs/iclr27_phase70/completion/joint_d_repair1_f{fold}.done"),
        })

    folds = []
    for row in aggregate["folds"]:
        fold = int(row["fold"])
        metric = {
            "fold": fold,
            "checkpoint_steps": checkpoint_steps[fold],
            "prediction": row["prediction"],
            "gt_rows": row["gt_rows"],
            "proposal_top20_recall": row["top20_recall"],
            "proposal_top20_mean_best_iou": row["top20_mean_best_iou"],
            "proposal_top20_median_best_iou": row["top20_median_best_iou"],
            "track_continuity_proxy": {
                "mean_reliable_fraction": row["track_continuity_proxy"]["mean_reliable_fraction"],
                "median_reliable_fraction": row["track_continuity_proxy"]["median_reliable_fraction"],
                "tracks_reliable_at_least_once": row["track_continuity_proxy"]["tracks_reliable_at_least_once"],
            },
            "trackeval_macro": row["trackeval_macro"],
            "trackeval_count_sums": row["trackeval_count_sums"],
            "trackeval_count_weighted": row["trackeval_count_weighted"],
            "trackeval_summary_count": row["trackeval_summary_count"],
        }
        folds.append(metric)

    q0_top20 = q0_recall["recall"]["topk"]["20"]
    q0_macro = q0_trackeval["macro"]
    learned = aggregate["fold_mean"]
    # The user's safety gate is a validation-only stop rule.  It is evaluated
    # before any semantic/controller metric is allowed to be interpreted.
    proposal_pass = learned["top20_recall_iou05"] >= q0_top20["thresholds"]["0.5"]["recall"]
    metric_pairs = {
        "HOTA": "trackeval_macro_HOTA",
        "DetA": "trackeval_macro_DetA",
        "AssA": "trackeval_macro_AssA",
        "IDF1": "trackeval_macro_IDF1",
    }
    mot_non_degraded = all(learned[new_key] >= q0_macro[old_key] for old_key, new_key in metric_pairs.items())
    sanity = {
        "q0_reference_top20_recall_iou05": q0_top20["thresholds"]["0.5"]["recall"],
        "candidate_top20_recall_iou05": learned["top20_recall_iou05"],
        "proposal_top20_iou05_non_degraded": proposal_pass,
        "q0_reference_trackeval": {
            "HOTA": q0_macro["HOTA"],
            "DetA": q0_macro["DetA"],
            "AssA": q0_macro["AssA"],
            "IDF1": q0_macro["IDF1"],
        },
        "candidate_trackeval": {
            "HOTA": learned["trackeval_macro_HOTA"],
            "DetA": learned["trackeval_macro_DetA"],
            "AssA": learned["trackeval_macro_AssA"],
            "IDF1": learned["trackeval_macro_IDF1"],
        },
        "mot_non_degraded": mot_non_degraded,
        "passed": bool(proposal_pass and mot_non_degraded),
        "decision": "STOP_PHASE70_JOINT_REPAIR1_BEFORE_CAUSAL_OCD",
        "reasons": [
            "proposal top20 IoU>=0.5 recall is far below the Q0 reference",
            "HOTA/DetA/AssA/IDF1 are degraded relative to Q0",
            "validation sanity failure blocks semantic retrieval/controller interpretation",
        ],
    }
    artifact = {
        "phase": 70,
        "tag": "joint_d_repair1",
        "step_label": 5000,
        "protocol": aggregate["protocol"],
        "validation_split": "TRAIN-disjoint validation annotations; post-hoc scoring only",
        "causal_prefixes": [1, 2, 4, 8, 16],
        "checkpoint_steps": checkpoint_steps,
        "checkpoints": checkpoints,
        "folds": folds,
        "aggregate": aggregate["fold_mean"],
        "sanity_gate": sanity,
        "retrieval": {"status": "NOT_RUN", "blocked_by": "proposal_mot_sanity_gate"},
        "controller": {"status": "NOT_RUN", "blocked_by": "proposal_mot_sanity_gate"},
        "sealed": {"status": "NOT_RUN", "reason": "no valid frozen MOT candidate"},
        "labels_used_for_model": False,
        "held_event_gt_used_for_model": False,
        "sealed_public_q1_accessed": False,
        "references": {
            "q0_proposal_recall": str(q0_recall_path),
            "q0_trackeval": str(q0_trackeval_path),
            "aggregate": str(aggregate_path),
        },
    }
    atomic_json(OUT, artifact)
    marker = {
        "phase": 70,
        "tag": "joint_d_repair1",
        "step_label": 5000,
        "status": "done",
        "artifact": str(OUT),
        "artifact_sha256": sha256(OUT),
        "decision": sanity["decision"],
    }
    atomic_json(DONE, marker)
    print(json.dumps({"artifact": str(OUT), "done": str(DONE), "sanity": sanity}, indent=2))


if __name__ == "__main__":
    main()
