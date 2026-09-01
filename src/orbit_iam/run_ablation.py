"""Single-seed diagnostic ablation for ORBIT-IAM (official seed1027).

Runs after Candidate A is frozen; results are diagnostics only and are not
used to modify any frozen configuration.  A0=C1, A1=IAM without hard
negatives, A2=IAM without confidence, A3=IAM without margin, A4=Full IAM.
"""
from __future__ import annotations

import csv
import json

from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.orbit_msr.evaluate import (
    load_msr_model,
    evaluate_official,
    mechanism_rates,
)
from src.orbit_iam.evaluate_iam import evaluate_official_iam, load_iam_model

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"


def row_for(name, res, mr, thr=None, margin=None):
    r = {
        "ablation": name,
        "known_acc": res["overall_known_acc"],
        "rn_acc": res["route_aware_novel_acc"],
        "cond_novel_acc": res["conditional_novel_acc"],
        "routing_recall": res["novel_routing_recall"],
        "nmi": res["novel_only_nmi"],
        "ari": res["novel_only_ari"],
        "count_error": res["novel_count_abs_error"],
        "known_to_novel": mr["known_to_novel"],
        "novel_to_known": mr["novel_to_known"],
        "repeated_false_birth": mr["repeated_false_birth"],
        "wrong_existing": mr["wrong_existing"],
        "first_merge": mr["first_merge"],
    }
    if thr is not None:
        r["compat_threshold"] = thr
        r["compat_margin"] = margin
    return r


def main():
    import pathlib
    out = pathlib.Path(ROOT) / "outputs/orbit_iam/results/ablation_seed1027.csv"
    rows = []

    # A0: C1 baseline
    model, ck = load_msr_model(ROOT + "/runs/orbit_msr/msr_nr2/model.pth")
    logs, gt = evaluate_official(model, ck, "cuda", 0.5, 0.45)
    res, _ = assignment_from_preds(emit_preds(logs), gt)
    mr = mechanism_rates(logs, res["hungarian_assignment"])
    rows.append(row_for("A0_C1", res, mr))
    print("A0 done", rows[-1]["rn_acc"], flush=True)

    for name, ckpt, thr, margin in [
        ("A1_no_hardneg", "runs/orbit_iam/iam_a1_v3/model.pth", 0.45, 0.05),
        ("A2_no_conf", "runs/orbit_iam/iam_i1_v3/model.pth", 0.45, 0.05),
        ("A3_no_margin", "runs/orbit_iam/iam_a3_v3/model.pth", 0.45, 0.05),
        ("A4_full", "runs/orbit_iam/iam_i3_v3/model.pth", 0.45, 0.05),
    ]:
        model, ck = load_iam_model(ROOT + "/" + ckpt, "cuda")
        logs, gt = evaluate_official_iam(model, ck, "cuda", 0.5, thr, margin)
        res, _ = assignment_from_preds(emit_preds(logs), gt)
        mr = mechanism_rates(logs, res["hungarian_assignment"])
        rows.append(row_for(name, res, mr, thr, margin))
        print(name, "done", rows[-1]["rn_acc"], flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fn = []
    for r in rows:
        for k in r:
            if k not in fn:
                fn.append(k)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("saved", out)


if __name__ == "__main__":
    main()
