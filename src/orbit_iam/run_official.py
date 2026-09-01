"""Run a frozen ORBIT-IAM candidate on official Pure Full seed1027.

Appends the result row to outputs/orbit_iam/results/official_comparison.csv
and writes per-candidate CSV outputs/orbit_iam/results/<name>_seed1027.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_msr.evaluate import mechanism_rates
from src.orbit_iam.evaluate_iam import evaluate_official_iam, load_iam_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate_name", required=True)
    ap.add_argument("--compat_threshold", type=float, default=0.5)
    ap.add_argument("--compat_margin", type=float, default=0.02)
    ap.add_argument("--device", default="cuda:8")
    args = ap.parse_args()

    model, ck = load_iam_model(ROOT / args.checkpoint, args.device)
    logs, gt = evaluate_official_iam(model, ck, args.device,
                                     gate_thr=0.5,
                                     compat_thr=args.compat_threshold,
                                     compat_margin=args.compat_margin)
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    mr = mechanism_rates(logs, res["hungarian_assignment"])
    row = {
        "candidate": args.candidate_name,
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
        "compat_threshold": args.compat_threshold,
        "compat_margin": args.compat_margin,
    }
    print(json.dumps(row, indent=1), flush=True)

    out_dir = ROOT / "outputs/orbit_iam/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.candidate_name}_seed1027.csv").write_text(
        csv_text([row]))
    comp_path = out_dir / "official_comparison.csv"
    rows = []
    if comp_path.exists():
        rows = list(csv.DictReader(comp_path.open()))
    rows = [r for r in rows if r.get("candidate") != args.candidate_name]
    rows.append(row)
    fn = []
    for r in rows:
        for k in r:
            if k not in fn:
                fn.append(k)
    comp_path.write_text(csv_text(rows, fn))
    print("saved", out_dir)


def csv_text(rows, fieldnames=None):
    if not rows:
        return ""
    out = []
    import io
    buf = io.StringIO()
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    main()
