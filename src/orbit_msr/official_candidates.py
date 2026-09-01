"""Official Pure Full seed1027 for frozen ORBIT-MSR candidates."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch

from src.orbit_msr.evaluate import (
    load_msr_model,
    evaluate_official,
    evaluate_split,
    mechanism_rates,
)
from src.iclr27_phase4d.long_stream import active_bucket, stage_of

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
CANDIDATES = [
    ("candidate_1", "runs/orbit_msr/msr_nr2/model.pth", 0.5, 0.45),
    ("candidate_2", "runs/orbit_msr/msr_c2/model.pth", 0.5, 0.45),
]


def main():
    for name, path, gt, rt in CANDIDATES:
        model, ck = load_msr_model(ROOT / path)
        logs, gt_rows = evaluate_official(model, ck, "cuda", gate_thr=gt,
                                          reuse_thr=rt)
        res, ev = evaluate_split(logs, gt_rows)
        mr = mechanism_rates(logs, res["hungarian_assignment"])
        row = {
            "candidate": name, "gate_threshold": gt, "reuse_threshold": rt,
            "all_acc": res["all_track_acc"],
            "known_acc": res["overall_known_acc"],
            "rn_acc": res["route_aware_novel_acc"],
            "cond_novel_acc": res["conditional_novel_acc"],
            "routing_recall": res["novel_routing_recall"],
            "nmi": res["novel_only_nmi"], "ari": res["novel_only_ari"],
            "count_error": res["novel_count_abs_error"],
            "predicted_novel_count": res["predicted_novel_count"],
            "known_to_novel": mr["known_to_novel"],
            "novel_to_known": mr["novel_to_known"],
            "repeated_false_birth": mr["repeated_false_birth"],
            "wrong_existing": mr["wrong_existing"],
            "first_merge": mr["first_merge"],
        }
        out_dir = ROOT / "outputs" / "orbit_msr" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{name}_seed1027.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        (ROOT / f"runs/orbit_msr/{name}_seed1027.json").write_text(
            json.dumps({**res, "decomposition": mr, "prediction_log":
                        [{"sample_id": l["sample_id"],
                          "arrival_index": l["arrival_index"],
                          "predicted_action": l["predicted_action"],
                          "predicted_known_id": l["predicted_known_id"],
                          "predicted_virtual_novel_id": l["predicted_virtual_novel_id"],
                          "active_novel_prototypes": l["active_novel_prototypes"],
                          "prototype_support": l["prototype_support"],
                          "best_known_similarity": l["best_known_similarity"],
                          "best_novel_similarity": l["best_novel_similarity"],
                          "true_role": l["true_role"],
                          "true_class": l["true_class"],
                          "first_occurrence": l["first_occurrence"],
                          "stage": l["stage"]} for l in logs]},
                       indent=1, default=str))
        print(json.dumps(row, indent=1), flush=True)
        # per-bucket and per-stage breakdown
        b_rows = []
        for bucket in ["0-32", "33-128", "129-256", "257+"]:
            r = _summarize(name, logs, gt_rows, bucket,
                           lambda l, b=bucket:
                           active_bucket(l["active_novel_prototypes"]) == b)
            if r:
                b_rows.append(r)
        for stage in ["early", "middle", "late"]:
            r = _summarize(name, logs, gt_rows, stage,
                           lambda l, s=stage: l["stage"] == s)
            if r:
                b_rows.append(r)
        with open(out_dir / f"{name}_breakdown.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(b_rows[0].keys()))
            w.writeheader()
            w.writerows(b_rows)


def _summarize(name, logs, gt_rows, scope, select):
    from src.orbit_msr.evaluate import summarize
    return summarize(name, logs, gt_rows, scope, select)


if __name__ == "__main__":
    main()
