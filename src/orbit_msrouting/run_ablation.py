"""Single-seed diagnostic ablation for ORBIT-MSRouting (official seed1027).

Diagnostics only; never used to modify frozen configurations.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_msrouting.evaluate_msrouting import (
    bucket_rows,
    evaluate_official_msrouting,
    load_msrouting_checkpoint,
    result_row,
)

ROOT = pathlib.Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--g1", default="runs/orbit_msrouting/msrouting_g1/model.pth")
    ap.add_argument("--g2", default="runs/orbit_msrouting/msrouting_g2/model.pth")
    ap.add_argument("--g1_memonly",
                    default="runs/orbit_msrouting/msrouting_g1_memonly/model.pth")
    args = ap.parse_args()
    out = ROOT / "outputs/orbit_msrouting/results/ablation_seed1027.csv"
    rows = []

    def run(name, ckpt, gate_mode=None, zero_state=False):
        p = ROOT / ckpt
        if not p.exists():
            print("SKIP missing", ckpt, flush=True)
            return
        model, ck = load_msrouting_checkpoint(str(p), args.device,
                                              gate_mode=gate_mode)
        logs, gt = evaluate_official_msrouting(
            model, ck, args.device, gate_thr=0.5, compat_thr=0.45,
            compat_margin=0.05, zero_state=zero_state)
        r = result_row(logs, gt, name)
        rows.append(r)
        print(name, r["rn_acc"], r["ari"], r["novel_to_known"], flush=True)

    # A0 = Phase 4F M2 baseline (G0)
    run("A0_M2_G0", "runs/orbit_mdc/mdc_m2/model.pth")
    # A1 = G1 without memory state at eval (zero state inputs)
    run("A1_G1_no_state", args.g1, gate_mode="G1", zero_state=True)
    # A2 = G1 memory-size only (separately trained checkpoint)
    run("A2_G1_mem_only", args.g1_memonly)
    # A3 = G1 full selected state
    run("A3_G1_full", args.g1)
    # A4 = G2 residual correction removed (load G2 checkpoint as G0)
    run("A4_G2_no_residual", args.g2, gate_mode="G0")
    # A5 = G2 full
    run("A5_G2_full", args.g2)

    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("saved", out)


if __name__ == "__main__":
    main()
