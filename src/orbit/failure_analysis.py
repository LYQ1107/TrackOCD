"""Failure analysis for ORBIT-D1 on official Pure Full seed1027."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.orbit.protocol import load_gt

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    res = json.load(open(ROOT / "runs/orbit/final_D1_pure_full_main_seed1027.json"))
    preds = {p["sample_id"]: p for p in res["prediction_log"]}
    gt = load_gt("pure")
    rows = []
    for g in gt:
        if g["protocol_role"] == "distractor":
            continue
        p = preds.get(g["sample_id"], {})
        rows.append((g, p))
    known = [(g, p) for g, p in rows if g["protocol_role"] in ("supported_known", "zero_shot_known")]
    novel = [(g, p) for g, p in rows if g["protocol_role"] == "novel"]
    conf = Counter()
    for g, p in novel:
        t = p.get("prediction_type", "unresolved")
        conf[(g["protocol_role"], t)] += 1
    action_conf = Counter()
    for g, p in novel:
        if p.get("prediction_type") == "known":
            action_conf["novel->KNOWN"] += 1
        elif p.get("prediction_type") == "novel":
            action_conf["novel->NOVEL"] += 1
        else:
            action_conf["novel->UNRESOLVED"] += 1
    # first vs repeated novel
    first_seen = set()
    first_stats = {"first": 0, "repeated": 0, "first_correct_route": 0, "repeated_correct_route": 0}
    for g, p in sorted(novel, key=lambda x: int(x[1].get("stream_order", 0))):
        c = g["ground_truth_category_id"]
        is_first = c not in first_seen
        first_seen.add(c)
        routed = p.get("prediction_type") == "novel"
        if is_first:
            first_stats["first"] += 1
            first_stats["first_correct_route"] += int(routed)
        else:
            first_stats["repeated"] += 1
            first_stats["repeated_correct_route"] += int(routed)
    # head/mid/tail by GT category frequency
    cat_count = Counter(g["ground_truth_category_id"] for g, _ in novel)
    order = sorted(cat_count)
    n = len(order)
    bands = {"head": set(order[: max(1, n // 3)]),
             "mid": set(order[n // 3: 2 * n // 3]),
             "tail": set(order[2 * n // 3:])}
    band_stats = {}
    for name, cats in bands.items():
        sub = [g for g, p in novel if g["ground_truth_category_id"] in cats]
        routed = [g for g, p in novel if g["ground_truth_category_id"] in cats and p.get("prediction_type") == "novel"]
        band_stats[name] = {"n": len(sub), "routing_recall": len(routed) / max(len(sub), 1)}
    # write CSVs
    out = ROOT / "outputs" / "orbit" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "action_confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT", "PredictedAction", "Count"])
        for k, v in action_conf.items():
            w.writerow([k.split("->")[0], k.split("->")[1], v])
    with open(out / "first_vs_repeated_novel.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Occurrence", "Tracks", "RoutingRecall"])
        for k in ("first", "repeated"):
            w.writerow([k, first_stats[k], first_stats[f"{k}_correct_route"] / max(first_stats[k], 1)])
    with open(out / "failure_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Metric", "Value"])
        w.writerow(["known", "acc", res["overall_known_acc"]])
        w.writerow(["novel", "route_aware_acc", res["route_aware_novel_acc"]])
        w.writerow(["novel", "routing_recall", res["novel_routing_recall"]])
        for name, s in band_stats.items():
            w.writerow([f"novel_{name}_class", "routing_recall", s["routing_recall"]])
    print(json.dumps({"known": len(known), "novel": len(novel), "action_conf": dict(action_conf),
                      "first_stats": first_stats, "band_stats": band_stats}, indent=1))


if __name__ == "__main__":
    main()
