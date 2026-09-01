"""Router evidence separability audit on real-stream episodic samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

from src.iclr27_phase4u.data import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(ROOT / args.samples)
    X, y, ep = d["X"], d["y"], d["ep_idx"]
    mask = y <= 1
    X, y, ep = X[mask], y[mask], ep[mask]
    te_mask = ep >= int(0.8 * (int(ep.max()) + 1))
    Xtr, ytr = X[~te_mask], y[~te_mask]
    Xte, yte = X[te_mask], y[te_mask]

    def report(name, score_tr, score_te):
        fpr, tpr, _ = roc_curve(yte, score_te)
        fpr95 = None
        idx = np.where(tpr >= 0.95)[0]
        if len(idx):
            fpr95 = float(fpr[idx[0]])
        return {
            "evidence": name,
            "train_auroc": round(float(roc_auc_score(ytr, score_tr)), 4),
            "heldout_auroc": round(float(roc_auc_score(yte, score_te)), 4),
            "heldout_auprc": round(float(average_precision_score(yte, score_te)), 4),
            "fpr_at_95tpr": (round(fpr95, 4) if fpr95 is not None else None),
        }

    cols = {
        "known_top1": 0, "known_margin": 1, "known_entropy": 2,
        "known_energy": 3, "novel_max": 4, "novel_margin": 5,
        "novel_new_logit": 6, "novel_logK": 7, "disagreement": 8,
    }
    results = []
    # known-only evidence
    results.append(report("known_energy_only", Xtr[:, cols["known_energy"]],
                          Xte[:, cols["known_energy"]]))
    results.append(report("known_top1_only", Xtr[:, cols["known_top1"]],
                          Xte[:, cols["known_top1"]]))
    # novel-only evidence
    results.append(report("novel_max_only", Xtr[:, cols["novel_max"]],
                          Xte[:, cols["novel_max"]]))
    results.append(report("novel_new_logit_only", Xtr[:, cols["novel_new_logit"]],
                          Xte[:, cols["novel_new_logit"]]))
    # linear probe on subsets
    for sel, name in [([0, 1, 2, 3], "known_ev_linear"),
                      ([4, 5, 6, 7], "novel_ev_linear"),
                      ([0, 1, 2, 3, 4, 5, 6, 7, 8], "dual_ev_linear"),
                      (list(range(15)), "dual_plus_q_linear")]:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr[:, sel], ytr)
        results.append(report(name, clf.predict_proba(Xtr[:, sel])[:, 1],
                              clf.predict_proba(Xte[:, sel])[:, 1]))

    # joint distribution summary (held-out)
    out = {
        "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
        "results": results,
        "joint_stats": {
            "known_novel_known_energy_mean": float(np.mean(Xte[yte == 1, 3])),
            "novel_known_energy_mean": float(np.mean(Xte[yte == 0, 3])),
            "known_novel_max_mean": float(np.mean(Xte[yte == 1, 4])),
            "novel_novel_max_mean": float(np.mean(Xte[yte == 0, 4])),
        },
    }
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
