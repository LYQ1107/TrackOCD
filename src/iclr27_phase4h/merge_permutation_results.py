"""Merge permutation track logs into summary + root-cause probe tables."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_gt
from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_msr.evaluate import mechanism_rates


def read_logs(path):
    return list(csv.DictReader(open(path)))


def metrics(logs, gt):
    aug = []
    for l in logs:
        l2 = dict(l)
        l2["true_role"] = ("supported_known" if l["role"] == "known"
                           else "novel")
        l2["true_class"] = l["class"]
        aug.append(l2)
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    mr = mechanism_rates(aug, res["hungarian_assignment"])
    return {
        "all_acc": res["all_track_acc"],
        "known_acc": res["overall_known_acc"],
        "rn_acc": res["route_aware_novel_acc"],
        "cond_novel_acc": res["conditional_novel_acc"],
        "routing_recall": res["novel_routing_recall"],
        "nmi": res["novel_only_nmi"],
        "ari": res["novel_only_ari"],
        "count_error": res["novel_count_abs_error"],
        "predicted_novel_count": res["predicted_novel_count"],
        "known_to_novel": mr["known_to_novel"],
        "novel_to_known": mr["novel_to_known"],
        "repeated_false_birth": mr["repeated_false_birth"],
        "wrong_existing": mr["wrong_existing"],
        "first_merge": mr["first_merge"],
        "final_memory_size": max((int(l["memory_size"]) for l in logs),
                                 default=0),
        "mean_memory_size": float(np.mean(
            [int(l["memory_size"]) for l in logs])),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    log_dir = ROOT / "outputs/iclr27_phase4h/audit/permutation_track_logs"
    out_dir = ROOT / "outputs/iclr27_phase4h/audit"
    gt = load_gt("pure")
    res_rows = []
    cls_rows = []
    track_rows = []
    for p in sorted(log_dir.glob("*.csv")):
        if "smoke" in p.name:
            continue
        tag = p.stem
        if tag.startswith("P1") or tag.startswith("P2"):
            mode, seed = tag.split("_")
        else:
            mode, seed = tag, "0"
        logs = read_logs(p)
        m = metrics(logs, gt)
        res_rows.append({"mode": mode, "seed": seed, **m})
        by_class = defaultdict(list)
        for l in logs:
            by_class[l["class"]].append(l)
            track_rows.append({
                "mode": mode, "seed": seed,
                "sample_id": l["sample_id"], "class": l["class"],
                "role": l["role"], "arrival_index": int(l["arrival_index"]),
                "memory_size": int(l["memory_size"]),
                "gate_prob": float(l["gate_prob"]),
                "predicted_action": l["predicted_action"],
                "best_known_sim": float(l["best_known_sim"]),
            })
        for c, ls in by_class.items():
            role = ls[0]["role"]
            cls_rows.append({
                "mode": mode, "seed": seed, "class": c, "role": role,
                "count": len(ls),
                "n2k_rate": (sum(1 for l in ls
                                 if l["predicted_action"] == "KNOWN") / len(ls)
                             if role == "novel" else ""),
                "k2n_rate": (sum(1 for l in ls
                                 if l["predicted_action"] != "KNOWN") / len(ls)
                             if role == "known" else ""),
                "first_arrival": min(int(l["arrival_index"]) for l in ls),
                "mean_arrival": float(np.mean(
                    [int(l["arrival_index"]) for l in ls])),
            })
        print(tag, "rn", round(m["rn_acc"], 4), "n2k",
              round(m["novel_to_known"], 4), "ari", round(m["ari"], 4),
              "mem", m["final_memory_size"], flush=True)
    res_rows.sort(key=lambda r: (r["mode"], str(r["seed"])))
    cls_rows.sort(key=lambda r: (r["mode"], str(r["seed"]), r["class"]))
    write_csv(out_dir / "permutation_results.csv", res_rows)
    write_csv(out_dir / "permutation_class_results.csv", cls_rows)
    write_csv(out_dir / "permutation_track_dataset.csv", track_rows)
    print("saved merged", len(res_rows), "runs,", len(cls_rows), "class rows,",
          len(track_rows), "tracks")


if __name__ == "__main__":
    main()
