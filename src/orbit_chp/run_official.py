"""Run frozen CHP candidates on official Pure Full seed1027 (diagnostics)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit_msr.evaluate import mechanism_rates
from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_mdc.evaluate_mdc import evaluate_official_mdc, load_mdc_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    device = "cuda"
    model, ck = load_mdc_model(args.checkpoint, device)
    logs, gt = evaluate_official_mdc(
        model, ck, device, args.gate_threshold, args.compat_threshold,
        args.compat_margin)
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    mr = mechanism_rates(logs, res["hungarian_assignment"])
    row = {
        "candidate": args.name,
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
    }
    print(json.dumps(row, indent=1))
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            with open(out, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writerow(row)
        else:
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
