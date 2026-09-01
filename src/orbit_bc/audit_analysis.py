"""Analyze ORBIT-D1 decision logs: action confusion, cluster birth, geometry."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_logs(path):
    return list(csv.DictReader(open(path)))


def true_action(log):
    if log["true_role"] in ("supported_known", "zero_shot_known"):
        return "KNOWN"
    return "NEW_NOVEL" if log["first_occurrence"] == "True" else "EXISTING_NOVEL"


def main():
    logs = load_logs(ROOT / "outputs/orbit_bc/audit/per_track_decisions_val_seed1027.csv")
    actions = ["KNOWN", "EXISTING_NOVEL", "NEW_NOVEL"]
    conf = Counter()
    for l in logs:
        conf[(true_action(l), l["predicted_action"])] += 1
    conf_rows = []
    for t in actions:
        for p in actions:
            conf_rows.append({"true": t, "pred": p, "count": conf[(t, p)]})
    out = ROOT / "outputs/orbit_bc/audit"
    with open(out / "action_confusion_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["true", "pred", "count"]); w.writeheader(); w.writerows(conf_rows)

    # per true novel class birth stats
    by_class = defaultdict(list)
    for l in logs:
        if l["true_role"] == "novel":
            by_class[l["true_class"]].append(l)
    rows = []
    for c, ls in sorted(by_class.items()):
        vids = [l["predicted_virtual_novel_id"] for l in ls if l["predicted_virtual_novel_id"] not in ("None", "")]
        clusters = set(vids)
        cnt = Counter(vids)
        max_cover = (max(cnt.values()) / len(ls)) if (ls and cnt) else 0
        rows.append({
            "true_class": c, "tracks": len(ls),
            "first_pos": min(int(l["arrival_index"]) for l in ls),
            "virtual_clusters": len(clusters),
            "max_cluster_coverage": max_cover,
            "fragmentation_ratio": len(clusters) / len(ls) if ls else 0,
        })
    with open(out / "per_class_birth_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    repeated_novel = [l for l in logs if l["true_role"] == "novel" and l["first_occurrence"] == "False"]
    repeated_false_birth = sum(1 for l in repeated_novel if l["predicted_action"] == "NEW_NOVEL")
    known_logs = [l for l in logs if l["true_role"] in ("supported_known", "zero_shot_known")]
    known_to_new = sum(1 for l in known_logs if l["predicted_action"] == "NEW_NOVEL")
    first_novel = [l for l in logs if l["true_role"] == "novel" and l["first_occurrence"] == "True"]
    first_to_existing = sum(1 for l in first_novel if l["predicted_action"] == "EXISTING_NOVEL")
    stats = {
        "repeated_novel_tracks": len(repeated_novel),
        "repeated_false_birth_rate": repeated_false_birth / max(len(repeated_novel), 1),
        "known_to_new_rate": known_to_new / max(len(known_logs), 1),
        "first_novel_to_existing_rate": first_to_existing / max(len(first_novel), 1),
        "mean_virtual_clusters_per_true_novel_class": float(np.mean([r["virtual_clusters"] for r in rows])),
    }
    (out / "root_cause_decision.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
