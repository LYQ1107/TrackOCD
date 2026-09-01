"""Phase 4J candidate comparison table (J0/J1/J2...)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.iclr27_phase4j.trackeval_metrics import flat


TRACK_METRICS = ["HOTA", "DetA", "AssA", "LocA", "IDF1", "MOTA", "IDSW",
                 "Frag"]
SEM_METRICS = [
    "routing_accuracy", "k2n_rate_known_denom", "n2k_rate_novel_denom",
    "known_class_accuracy", "semantic_id_switch_rate", "novel_consistency",
    "commit_coverage_known", "commit_coverage_novel", "commit_coverage_fp",
    "commit_coverage_len2_fp", "commit_latency_mean_novel",
    "commit_latency_median_novel", "commit_latency_p90_novel",
    "commit_latency_mean_fp", "commit_latency_median_fp",
    "commit_latency_p90_fp", "fp_novel_observation_rate",
    "fp_global_memory_admission_rate", "fp_stable_novel_birth_rate",
    "global_novel_memory_size", "novel_ids_created", "gt_novel_reuse_tracks",
    "fp_birth_later_used_by_gt_novel",
    "known_tracks_committed_to_novel",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "iclr27_phase4j" /
                    "compare_candidates.csv")
    args = ap.parse_args()
    rows = []
    for tag in args.tags:
        row = {"tag": tag}
        te_json = ROOT / "outputs" / "iclr27_phase4j" / "trackeval" / \
            tag / "trackeval.json"
        if te_json.exists():
            f = flat(te_json)[tag]
            for k in TRACK_METRICS:
                row[k] = round(float(f[k]), 4)
        se_csv = ROOT / "outputs" / "iclr27_phase4j" / "audit" / \
            f"semantic_eval_{tag}.csv"
        if se_csv.exists():
            with open(se_csv) as fh:
                d = next(csv.DictReader(fh))
            for k in SEM_METRICS:
                row[k] = d.get(k, "")
        fr_csv = ROOT / "outputs" / "iclr27_phase4j" / "audit" / \
            f"fragment_{tag}.csv"
        if fr_csv.exists():
            fr = list(csv.DictReader(open(fr_csv)))
            frag = [r for r in fr if int(r.get("fragmented", 0))]
            n_cons = sum(1 for r in frag
                         if r.get("semantic_consistent") == "1")
            n_gcons = sum(1 for r in frag
                          if r.get("global_novel_consistent") == "1")
            row["fragment_semantic_consistent"] = round(
                n_cons / max(len(frag), 1), 4)
            row["fragment_global_novel_consistent"] = round(
                n_gcons / max(len(frag), 1), 4)
            row["fragmented_gt_tracks"] = len(frag)
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        keys = ["tag"] + TRACK_METRICS + SEM_METRICS + [
            "fragment_semantic_consistent", "fragment_global_novel_consistent",
            "fragmented_gt_tracks"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
