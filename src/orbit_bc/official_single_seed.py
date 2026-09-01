"""Official validation single-seed ORBIT-BC evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import load_model, build_known
from src.orbit.protocol import load_frame_features, load_train_labels, load_stream, load_gt, subset_ids
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.orbit_bc.evaluate_bc import run_stream_bc


def run(threshold):
    device = "cuda"
    model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=device)
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    protos, radii = build_known(model, train_feats, train_labels, set(train_labels.values()), device)
    rows = load_stream("pure", "main_seed1027")
    feats = load_frame_features("gt_tracks_mean")
    preds, mem = run_stream_bc(model, rows, feats, protos, radii, device,
                               birth_threshold=threshold)
    ev = TrackOCDEvaluator(load_gt("pure"))
    res = ev.evaluate(preds)
    res["birth_threshold"] = threshold
    res["memory_size"] = mem.memory_stats() if hasattr(mem, "memory_stats") else None
    return res, preds


def main():
    rows_out = []
    for thr in [0.55, 0.45]:
        res, preds = run(thr)
        row = {
            "birth_threshold": thr,
            "all_acc": res["all_track_acc"],
            "known_acc": res["overall_known_acc"],
            "rn_acc": res["route_aware_novel_acc"],
            "cond_novel_acc": res["conditional_novel_acc"],
            "routing_recall": res["novel_routing_recall"],
            "nmi": res["novel_only_nmi"],
            "ari": res["novel_only_ari"],
            "count_error": res["novel_count_abs_error"],
            "predicted_novel_count": res["predicted_novel_count"],
        }
        rows_out.append(row)
        out = ROOT / "outputs/orbit_bc/results/orbit_bc_seed1027.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows([row])
        (ROOT / f"runs/orbit_bc/orbit_bc_{thr}_seed1027.json").write_text(
            json.dumps({**res, "prediction_log": preds}, indent=1, default=str))
        print(row, flush=True)


if __name__ == "__main__":
    main()
