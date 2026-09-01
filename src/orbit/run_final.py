"""Run final ORBIT (D1 best available) across protocols/subsets/seeds."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from src.orbit.evaluate import load_model, evaluate_official

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
CHECKPOINT = ROOT / "runs" / "orbit" / "model_D1_b128_g0.3" / "model.pth"


def main():
    model, _ = load_model(CHECKPOINT, device="cuda")
    streams = ["main_seed1027", "main_seed1028", "main_seed1029"]
    rows = []
    for proto in ["pure", "ov_assisted"]:
        for subset in ["full", "repeated", "balanced"]:
            seed_rows = []
            for stream in streams:
                res, preds = evaluate_official(model, proto, subset, stream, "cuda", mode="joint")
                row = {
                    "method": "ORBIT-D1", "protocol": proto, "subset": subset,
                    "seed": stream,
                    "all_track_acc": res["all_track_acc"],
                    "overall_known_acc": res["overall_known_acc"],
                    "route_aware_novel_acc": res["route_aware_novel_acc"],
                    "conditional_novel_acc": res["conditional_novel_acc"],
                    "novel_routing_recall": res["novel_routing_recall"],
                    "novel_only_nmi": res["novel_only_nmi"],
                    "novel_only_ari": res["novel_only_ari"],
                    "predicted_novel_count": res["predicted_novel_count"],
                    "novel_count_abs_error": res["novel_count_abs_error"],
                }
                rows.append(row)
                seed_rows.append(row)
                out = ROOT / "runs" / "orbit" / f"final_D1_{proto}_{subset}_{stream}.json"
                out.write_text(json.dumps({**res, "prediction_log": preds}, indent=1, default=str))
            mean_row = dict(seed_rows[0])
            mean_row["seed"] = "mean"
            for k in ["all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                      "conditional_novel_acc", "novel_routing_recall", "novel_only_nmi",
                      "novel_only_ari", "predicted_novel_count", "novel_count_abs_error"]:
                mean_row[k] = statistics.mean(r[k] for r in seed_rows)
            rows.append(mean_row)
            print(proto, subset, {k: round(mean_row[k], 4) for k in
                  ["all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                   "conditional_novel_acc", "novel_routing_recall"]}, flush=True)
    out_csv = ROOT / "outputs" / "orbit" / "tables" / "gt_track_main.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
