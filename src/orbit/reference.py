"""Unified TrackOCD-Ref reproduction (D0: DINOv2 mean + B2, thr=0.45)."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import load_train_known, build_prototypes
from src.orbit.protocol import load_mean_features, load_stream, load_gt, subset_ids
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def run_ref(proto, subset, stream):
    val_feats = load_mean_features("gt_tracks_mean")
    tr_feats, labels = load_train_known("dinov2")
    protos = build_prototypes(tr_feats, labels, set(labels.values()))
    mem = B2Memory(protos, threshold=0.45)
    rows = load_stream(proto, stream)
    preds = []
    for i, r in enumerate(rows):
        vid, kind = mem.predict_one(val_feats[r["sample_id"]], r["sample_id"], i)
        preds.append(emit(r["sample_id"], i, kind,
                          vid if kind == "known" else None,
                          vid if kind == "novel" else None))
    gt = load_gt(proto)
    ev = TrackOCDEvaluator(gt)
    res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
    return res, preds


def main():
    streams = ["main_seed1027", "main_seed1028", "main_seed1029"]
    rows = []
    all_res = []
    for stream in streams:
        res, preds = run_ref("pure", "full", stream)
        all_res.append(res)
        rows.append({
            "protocol": "pure", "subset": "full", "seed": stream,
            "all_track_acc": res["all_track_acc"],
            "overall_known_acc": res["overall_known_acc"],
            "route_aware_novel_acc": res["route_aware_novel_acc"],
            "conditional_novel_acc": res["conditional_novel_acc"],
            "novel_routing_recall": res["novel_routing_recall"],
            "novel_only_nmi": res["novel_only_nmi"],
            "novel_only_ari": res["novel_only_ari"],
            "predicted_novel_count": res["predicted_novel_count"],
            "novel_count_abs_error": res["novel_count_abs_error"],
        })
        out = ROOT / "runs" / "orbit" / f"ref_{stream}.json"
        out.write_text(json.dumps({**res, "prediction_log": preds}, indent=1, default=str))
    # mean row
    mean_row = {"protocol": "pure", "subset": "full", "seed": "mean",
                "all_track_acc": statistics.mean(r["all_track_acc"] for r in rows),
                "overall_known_acc": statistics.mean(r["overall_known_acc"] for r in rows),
                "route_aware_novel_acc": statistics.mean(r["route_aware_novel_acc"] for r in rows),
                "conditional_novel_acc": statistics.mean(r["conditional_novel_acc"] for r in rows),
                "novel_routing_recall": statistics.mean(r["novel_routing_recall"] for r in rows),
                "novel_only_nmi": statistics.mean(r["novel_only_nmi"] for r in rows),
                "novel_only_ari": statistics.mean(r["novel_only_ari"] for r in rows),
                "predicted_novel_count": statistics.mean(r["predicted_novel_count"] for r in rows),
                "novel_count_abs_error": statistics.mean(r["novel_count_abs_error"] for r in rows)}
    rows.append(mean_row)
    out_csv = ROOT / "outputs" / "orbit" / "reference_reproduction.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(mean_row, indent=1))


if __name__ == "__main__":
    main()
