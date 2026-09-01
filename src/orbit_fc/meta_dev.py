"""Meta-dev model selection for ORBIT-FC configs (F1/F2/F3)."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import torch

from src.orbit_fc.evaluate import load_fc_model, evaluate_proxy

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "orbit_fc" / "meta_dev"
OUT.mkdir(parents=True, exist_ok=True)


def error_rates(logs):
    known = [l for l in logs if l["true_role"] == "supported_known"]
    known_to_novel = sum(1 for l in known if l["predicted_action"] != "KNOWN")
    novel = [l for l in logs if l["true_role"] == "novel"]
    repeated = [l for l in novel if not l["first_occurrence"]]
    first = [l for l in novel if l["first_occurrence"]]
    repeated_false_birth = sum(1 for l in repeated if l["predicted_action"] == "NEW_NOVEL")
    first_merge = sum(1 for l in first if l["predicted_action"] == "EXISTING_NOVEL")
    return {
        "known_to_novel_error": known_to_novel / max(len(known), 1),
        "repeated_false_birth": repeated_false_birth / max(len(repeated), 1),
        "first_occurrence_merge": first_merge / max(len(first), 1),
        "known_tracks": len(known),
        "novel_tracks": len(novel),
    }


def main():
    device = "cuda"
    results = []
    for variant, ck_path in [("F1", "runs/orbit_fc/fc_F1/model.pth"),
                             ("F2", "runs/orbit_fc/fc_F2/model.pth")]:
        model, ck = load_fc_model(ROOT / ck_path, device=device)
        for gate_thr in [0.5, 0.55]:
            for reuse_thr in [0.5, 0.45]:
                res, preds, logs = evaluate_proxy(model, ck, device,
                                                  gate_thr=gate_thr,
                                                  reuse_thr=reuse_thr)
                er = error_rates(logs)
                row = {
                    "variant": variant, "gate_threshold": gate_thr,
                    "reuse_threshold": reuse_thr,
                    "all_acc": round(res["all_track_acc"], 4),
                    "known_acc": round(res["overall_known_acc"], 4),
                    "rn_acc": round(res["route_aware_novel_acc"], 4),
                    "cond_novel_acc": round(res["conditional_novel_acc"], 4),
                    "routing_recall": round(res["novel_routing_recall"], 4),
                    "nmi": round(res["novel_only_nmi"], 4),
                    "ari": round(res["novel_only_ari"], 4),
                    "count_error": res["novel_count_abs_error"],
                    "predicted_novel_count": res["predicted_novel_count"],
                    "known_to_novel_error": round(er["known_to_novel_error"], 4),
                    "repeated_false_birth": round(er["repeated_false_birth"], 4),
                    "first_occurrence_merge": round(er["first_occurrence_merge"], 4),
                }
                results.append(row)
                print(row, flush=True)
    with open(OUT / "config_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    # error tradeoff view
    trade_rows = []
    for r in results:
        trade_rows.append({
            "variant": r["variant"], "gate_threshold": r["gate_threshold"],
            "reuse_threshold": r["reuse_threshold"],
            "known_acc": r["known_acc"], "rn_acc": r["rn_acc"],
            "cond_novel_acc": r["cond_novel_acc"], "count_error": r["count_error"],
            "known_to_novel_error": r["known_to_novel_error"],
            "repeated_false_birth": r["repeated_false_birth"],
            "first_occurrence_merge": r["first_occurrence_merge"],
            "ari": r["ari"],
        })
    with open(OUT / "error_tradeoff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trade_rows[0].keys()))
        w.writeheader()
        w.writerows(trade_rows)


if __name__ == "__main__":
    main()
