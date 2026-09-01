"""Single-seed diagnostic ablation for ORBIT-MDC (official seed1027).

Diagnostics only; never used to modify frozen configurations.
"""
from __future__ import annotations

import csv
import pathlib

from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_msr.evaluate import (
    evaluate_official,
    load_msr_model,
    mechanism_rates,
)
from src.orbit_mdc.evaluate_mdc import (
    evaluate_official_mdc,
    load_mdc_model,
    result_row,
)

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def main():
    out = pathlib.Path(ROOT) / "outputs/orbit_mdc/results/ablation_seed1027.csv"
    rows = []
    # A0 = C1 baseline
    model, ck = load_msr_model(ROOT + "/runs/orbit_msr/msr_nr2/model.pth")
    logs, gt = evaluate_official(model, ck, "cuda", 0.5, 0.45)
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    mr = mechanism_rates(logs, res["hungarian_assignment"])
    r = result_row(logs, gt, "A0_C1")
    rows.append(r)
    print("A0", r["rn_acc"], flush=True)
    # MDC variants
    for name, ckpt, policy, qmode, extra in [
        ("A4_full_mdc", "runs/orbit_mdc/mdc_m2/model.pth", "birth", 0, None),
        ("A1_teacher_forced", "runs/orbit_mdc/mdc_a1_teacher/model.pth",
         "auto", 0, None),
        ("A2_no_realband", "runs/orbit_mdc/mdc_a2_norealand/model.pth",
         "auto", 0, None),
        ("A3_no_joint_gate", "runs/orbit_mdc/mdc_a3_nogate/model.pth",
         "auto", 0, None),
        ("A5_no_quarantine", "runs/orbit_mdc/mdc_m2/model.pth",
         "birth", 0, None),
    ]:
        p = pathlib.Path(ROOT) / ckpt
        if not p.exists():
            print("SKIP missing", ckpt, flush=True)
            continue
        model, ck = load_mdc_model(str(p), "cuda")
        logs, gt = evaluate_official_mdc(model, ck, "cuda", 0.5, 0.45, 0.05,
                                         0.5, policy, qmode)
        r = result_row(logs, gt, name)
        rows.append(r)
        print(name, r["rn_acc"], r["ari"], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
