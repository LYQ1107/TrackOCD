"""Run a frozen ORBIT-MSRouting candidate on official Pure Full seed1027."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_msrouting.evaluate_msrouting import (
    bucket_rows,
    evaluate_official_msrouting,
    load_msrouting_checkpoint,
    result_row,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def csv_text(rows, fieldnames=None):
    if not rows:
        return ""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate_name", required=True)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trajectory_csv", default=None)
    args = ap.parse_args()
    model, ck = load_msrouting_checkpoint(ROOT / args.checkpoint,
                                          args.device)
    logs, gt = evaluate_official_msrouting(
        model, ck, args.device, gate_thr=args.gate_threshold,
        compat_thr=args.compat_threshold, compat_margin=args.compat_margin)
    row = result_row(logs, gt, args.candidate_name)
    print(json.dumps(row, indent=1), flush=True)
    out_dir = ROOT / "outputs/orbit_msrouting/results"
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
    buckets = bucket_rows(logs, gt, args.candidate_name)
    if buckets:
        (out_dir / f"{args.candidate_name}_buckets_seed1027.csv").write_text(
            csv_text(buckets))
    if args.trajectory_csv:
        path = ROOT / args.trajectory_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(csv_text(logs))
    print("saved", out_dir)


if __name__ == "__main__":
    main()
