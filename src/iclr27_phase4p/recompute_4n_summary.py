"""Recompute Phase 4N headline object-validity numbers on the FIXED
(frame-aware) detection populations.  Old Phase 4N numbers were computed
on the det_local_id-truncated populations and are invalid."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from iclr27_phase4p.trajectory_objectness_audit import (
    auroc, auprc, load_pop, role_of,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4p" / "audit" / \
    "phase4n_recomputed_summary.csv"


def main():
    rows = []
    for mode, path in (
            ("dev", ROOT / "outputs" / "iclr27_phase4p" / "audit" /
             "detection_population_dev_fixed.csv"),
            ("heldout", ROOT / "outputs" / "iclr27_phase4p" / "audit" /
             "detection_population_heldout_corrected_fixed.csv")):
        pop = load_pop(path, mode)
        n_known = sum(1 for r in pop if role_of(r) == "KNOWN")
        n_novel = sum(1 for r in pop if role_of(r) == "NOVEL")
        n_fp = sum(1 for r in pop if role_of(r) == "FP")
        n_valid = n_known + n_novel
        # valid vs FP (static detector score)
        y = np.asarray([role_of(r) in ("KNOWN", "NOVEL") for r in pop])
        s = np.asarray([r["score"] for r in pop])
        auroc_vf = auroc(y, s)
        auprc_vf = auprc(y, s)
        # novel vs FP (static)
        y2 = np.asarray([role_of(r) == "NOVEL" for r in pop])
        auroc_nf = auroc(y2, s)
        auprc_nf = auprc(y2, s)
        # novel vs persistent FP, causal trajectory LR (reuse audit result)
        rows.append({
            "mode": mode, "rows": len(pop), "frames_known": n_known,
            "novel": n_novel, "fp": n_fp,
            "valid_precision_baseline": round(
                n_valid / max(len(pop), 1), 4),
            "valid_vs_FP_score_AUROC": round(auroc_vf, 4)
            if auroc_vf else "",
            "valid_vs_FP_score_AUPRC": round(auprc_vf, 4)
            if auprc_vf else "",
            "novel_vs_FP_score_AUROC": round(auroc_nf, 4)
            if auroc_nf else "",
            "novel_vs_FP_score_AUPRC": round(auprc_nf, 4)
            if auprc_nf else "",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("PHASE4N_RECOMPUTED_SUMMARY_DONE")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
