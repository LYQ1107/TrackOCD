"""Frozen-evidence separability audit for the CHP failure case.

Answers: in the frozen DINO / M2-adapted space, can train-side known tracks
be separated from held-out (meta-dev) tracks by best-known similarity to the
38-class pool prototypes?  If yes, representation evidence is sufficient and
the CHP failure is a training-signal/optimization problem; if the AUROC is
close to chance, REPRESENTATION_LIMITATION_SUPPORTED.

Offline audit only: no metric, threshold, or tier boundary is used for any
training decision.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_chp.split import load_chp_split
from src.orbit_mdc.evaluate_mdc import load_mdc_model


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def _auroc(scores, y):
    # y in {0,1}; score higher => more "known"
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos - 1) / 2)
                 / (n_pos * n_neg))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_chp_split()
    pool = set(split["episode_pool"])
    heldout = set(split["heldout"])
    labels = load_train_labels()
    all_feats = load_frame_features("train_known_mean")
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)

    model, ck = load_mdc_model(
        str(ROOT / "runs" / "orbit_mdc" / "mdc_m2" / "model.pth"), device)
    model.eval()

    def embed(sids):
        out = {}
        with torch.no_grad():
            for sid in sids:
                x = torch.as_tensor(
                    all_feats[sid][:8], dtype=torch.float32,
                    device=device).unsqueeze(0)
                mask = torch.ones(1, x.shape[1], dtype=torch.bool,
                                  device=device)
                z = model.aggregate(x, mask)["z"][0].cpu().numpy()
                out[sid] = _norm(z.astype(np.float32))
        return out

    pool_sids = [sid for c in pool for sid in by_class[c]]
    held_sids = [sid for c in heldout for sid in by_class[c]]
    z = embed(pool_sids + held_sids)

    # pool prototypes from adapted embeddings (train-side only)
    protos = {}
    for c in pool:
        zs = np.stack([z[sid] for sid in by_class[c]])
        protos[c] = _norm(zs.mean(axis=0).astype(np.float32))
    P = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    cids = sorted(protos)

    rows = []
    for sid, c in labels.items():
        if sid not in z:
            continue
        if int(c) not in pool and int(c) not in heldout:
            continue
        role = "known" if int(c) in pool else "novel"
        ks = P @ z[sid]
        order = np.argsort(ks)[::-1]
        best = float(ks[order[0]])
        second = float(ks[order[1]]) if len(order) >= 2 else best
        rows.append({
            "sample_id": sid, "class": c, "role": role,
            "best_known_cos": best, "known_margin": best - second,
            "nearest_class": cids[int(order[0])],
        })

    scores = np.array([r["best_known_cos"] for r in rows])
    y = np.array([1 if r["role"] == "known" else 0 for r in rows])
    novel_idx = np.where(y == 0)[0]
    hard_cut = float(np.percentile(scores[novel_idx], 66))
    hard_novel = np.array([s >= hard_cut for s in scores]) & (y == 0)
    easy_novel = np.array([s < hard_cut for s in scores]) & (y == 0)
    y_known = np.ones(int(y.sum()))

    summary = {
        "metric": "adapted best-known cosine to 38-class pool prototypes",
        "n_known": int(y.sum()),
        "n_novel": int((y == 0).sum()),
        "auroc_known_vs_novel": _auroc(scores, y),
        "auroc_known_vs_hard_novel": _auroc(
            np.concatenate([scores[y == 1], scores[hard_novel]]),
            np.concatenate([y_known, np.zeros(int(hard_novel.sum()))])),
        "auroc_known_vs_easy_novel": _auroc(
            np.concatenate([scores[y == 1], scores[easy_novel]]),
            np.concatenate([y_known, np.zeros(int(easy_novel.sum()))])),
        "hard_novel_cut": hard_cut,
        "mean_best_known_known": float(scores[y == 1].mean()),
        "mean_best_known_novel": float(scores[y == 0].mean()),
        "mean_best_known_hard_novel": float(scores[hard_novel].mean()),
    }

    # class-center overlap: nearest-other-pool-center cosine
    novel_center_cos = []
    for c in heldout:
        zs = np.stack([z[sid] for sid in by_class[c]])
        center = _norm(zs.mean(axis=0).astype(np.float32))
        novel_center_cos.append((c, float(np.max(P @ center))))

    def nearest_other(class_centers):
        vals = []
        for c in class_centers:
            idx = [i for i, cc in enumerate(cids) if cc != c]
            vals.append(float(np.max(P[idx] @ class_centers[c])))
        return np.array(vals)

    nk = nearest_other({c: protos[c] for c in pool})
    nc = np.array([v for _, v in novel_center_cos])
    summary.update({
        "n_pool_classes": len(pool),
        "n_heldout_classes": len(heldout),
        "known_class_center_nearest_mean": float(nk.mean()),
        "novel_class_center_nearest_mean": float(nc.mean()),
        "novel_centers_above_known_median": float(
            (nc > np.median(nk)).mean()),
    })

    out_dir = ROOT / "outputs" / "iclr27_phase4h" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "representation_separability.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "representation_separability_summary.json",
              "w") as f:
        json.dump({k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in summary.items()}, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
