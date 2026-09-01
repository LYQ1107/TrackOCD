"""Run a frozen ORBIT-MDC candidate on official Pure Full seed1027."""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict

from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_mdc.evaluate_mdc import (
    evaluate_official_mdc,
    load_mdc_model,
    result_row,
)

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def memory_stats(logs):
    by_vid = defaultdict(list)
    for l in logs:
        if l["predicted_action"] in ("EXISTING_NOVEL", "NEW_NOVEL"):
            by_vid[l["predicted_virtual_novel_id"]].append(l)
    hubs = sum(1 for ls in by_vid.values()
               if len({l["class"] for l in ls}) >= 2)
    known_origin = sum(1 for ls in by_vid.values()
                       if ls[0]["role"] == "known")
    return {"hub_prototype_count": hubs,
            "known_origin_prototype_count": known_origin,
            "predicted_prototype_count": len(by_vid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--candidate_name", required=True)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--birth_threshold", type=float, default=0.5)
    ap.add_argument("--policy", choices=["auto", "compat", "birth"],
                    default="auto")
    ap.add_argument("--quarantine_mode", type=int, default=0)
    ap.add_argument("--quarantine_support_thr", type=int, default=3)
    ap.add_argument("--quarantine_dispersion_thr", type=float, default=0.3)
    ap.add_argument("--quarantine_coef", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:8")
    args = ap.parse_args()
    model, ck = load_mdc_model(ROOT + "/" + args.checkpoint, args.device)
    logs, gt = evaluate_official_mdc(
        model, ck, args.device, gate_thr=0.5,
        compat_thr=args.compat_threshold,
        compat_margin=args.compat_margin,
        birth_thr=args.birth_threshold, policy=args.policy,
        quarantine_mode=args.quarantine_mode,
        quarantine_support_thr=args.quarantine_support_thr,
        quarantine_dispersion_thr=args.quarantine_dispersion_thr,
        quarantine_coef=args.quarantine_coef)
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    row = result_row(logs, gt, args.candidate_name)
    row.update(memory_stats(logs))
    row.update({"compat_threshold": args.compat_threshold,
                "compat_margin": args.compat_margin,
                "birth_threshold": args.birth_threshold,
                "policy": args.policy,
                "quarantine_mode": args.quarantine_mode,
                "quarantine_support_thr": args.quarantine_support_thr,
                "quarantine_dispersion_thr":
                    args.quarantine_dispersion_thr,
                "quarantine_coef": args.quarantine_coef})
    print(json.dumps(row, indent=1), flush=True)
    out_dir = ROOT + "/outputs/orbit_mdc/results"
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(out_dir) / f"{args.candidate_name}_seed1027.csv").write_text(
        csv_text([row]))
    comp_path = pathlib.Path(out_dir) / "official_comparison.csv"
    rows = []
    if comp_path.exists():
        rows = list(csv.DictReader(comp_path.open()))
    rows = [r for r in rows if r.get("candidate") != args.candidate_name]
    rows.append(row)
    comp_path.write_text(csv_text(rows))
    print("saved", out_dir)


def csv_text(rows):
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


if __name__ == "__main__":
    main()
