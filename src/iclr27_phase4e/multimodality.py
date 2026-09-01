"""Phase 4E multi-modality gate audit.

Offline, GT-labeled diagnostic over:
  1. real meta-dev novel classes (63 tracks, 10 classes);
  2. official Pure Full novel classes (843 tracks, 219 classes);
  3. synthetic long-stream novel classes (null control: single-center by
     construction).

All analysis uses C1's adapted track representation and is train-side /
offline only; no GT enters any test-time decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_train_labels,
    meta_classes,
)
from src.orbit_msr.evaluate import load_msr_model
from src.iclr27_phase4d.long_stream import load_stream_cache


def embed_tracks(model, feats, sids, device, batch=512):
    zs = {}
    for start in range(0, len(sids), batch):
        chunk = sids[start:start + batch]
        max_t = max(len(feats[s]) for s in chunk)
        x = np.zeros((len(chunk), max_t, 768), dtype=np.float32)
        m = np.zeros((len(chunk), max_t), dtype=bool)
        for i, s in enumerate(chunk):
            f = feats[s]
            x[i, :len(f)] = f
            m[i, :len(f)] = True
        with torch.no_grad():
            out = model.aggregate(
                torch.as_tensor(x, device=device),
                torch.as_tensor(m, device=device))
        for i, s in enumerate(chunk):
            zs[s] = out["z"][i].cpu().numpy().astype(np.float32)
    return zs


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def kmeans_cosine(X, k, seed=0, n_init=20, max_iter=100):
    """Spherical k-means on unit vectors; SSE in 1-cos space."""
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    rng = np.random.RandomState(seed)
    best = None
    for _ in range(n_init):
        idx = rng.choice(len(X), size=k, replace=False)
        centers = X[idx].copy()
        for _ in range(max_iter):
            sims = X @ centers.T
            assign = sims.argmax(axis=1)
            new = np.zeros_like(centers)
            for j in range(k):
                mask = assign == j
                if mask.sum():
                    new[j] = X[mask].mean(axis=0)
                    new[j] = _norm(new[j])
                else:
                    new[j] = centers[j]
            if np.allclose(new, centers, atol=1e-7):
                centers = new
                break
            centers = new
        sims = (X @ centers.T).max(axis=1)
        sse = float(np.sum(1.0 - sims))
        if best is None or sse < best[0]:
            best = (sse, centers.copy(), assign.copy())
    return best


def class_diagnostics(zs, labels):
    """Per-class single vs multi center diagnostics."""
    classes = sorted(set(labels))
    rows = []
    for c in classes:
        idx = [i for i, l in enumerate(labels) if l == c]
        X = np.stack([zs[i] for i in idx])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        n = len(idx)
        c1 = _norm(X.mean(axis=0))
        within = 1.0 - X @ c1
        sse1 = float(within.sum())
        r = {"class": int(c), "n": n,
             "sse1": sse1, "within_mean": float(within.mean()),
             "within_p95": float(np.percentile(within, 95)),
             "within_max": float(within.max())}
        if n >= 3:
            sse2, cen2, a2 = kmeans_cosine(X, 2)
            sse3, cen3, a3 = kmeans_cosine(X, 3)
            r["sse2"] = sse2
            r["sse3"] = sse3
            r["reduction12"] = (sse1 - sse2) / max(sse1, 1e-9)
            r["reduction23"] = (sse2 - sse3) / max(sse2, 1e-9)
            r["two_center_sim"] = float(np.dot(_norm(cen2[0]), _norm(cen2[1])))
            r["bic1"] = n * math.log(sse1 / n + 1e-12) + 1 * math.log(n)
            r["bic2"] = n * math.log(sse2 / n + 1e-12) + 2 * math.log(n)
            r["bic3"] = n * math.log(sse3 / n + 1e-12) + 3 * math.log(n)
            try:
                from sklearn.metrics import silhouette_score
                r["sil1"] = float(silhouette_score(
                    X, np.zeros(n), metric="cosine")) if n > 2 else float("nan")
                r["sil2"] = float(silhouette_score(X, a2, metric="cosine"))
            except Exception:
                r["sil1"] = r["sil2"] = float("nan")
        else:
            for k in ["sse2", "sse3", "reduction12", "reduction23",
                      "two_center_sim", "bic1", "bic2", "bic3", "sil1", "sil2"]:
                r[k] = float("nan")
        rows.append(r)
    return rows


def cross_class_overlap(zs, labels, class_rows):
    """Per-class within vs nearest-different-sample distance overlap."""
    X = np.stack([_norm(z) for z in zs])
    lab = np.array(labels, dtype=int)
    out = []
    for r in class_rows:
        c = r["class"]
        mask = lab == c
        others = ~mask
        if mask.sum() == 0 or others.sum() == 0:
            r["cross_nn_mean"] = r["cross_nn_min"] = r["overlap_rate"] = float("nan")
            out.append(r)
            continue
        d_within = 1.0 - X[mask] @ X[mask].T
        d_cross = 1.0 - X[mask] @ X[others].T
        cross_min = d_cross.min(axis=1)
        self_min = np.where(d_within > 1e-9, d_within, np.inf).min(axis=1)
        r["cross_nn_mean"] = float(cross_min.mean())
        r["cross_nn_min"] = float(cross_min.min())
        r["overlap_rate"] = float((self_min > cross_min).mean())
        out.append(r)
    return out


def aggregate(rows, dataset, subset_n=None):
    rs = [r for r in rows if subset_n is None or r["n"] >= subset_n]
    if not rs:
        return None
    def mean(k):
        vals = [r[k] for r in rs if r.get(k) is not None
                and not (isinstance(r[k], float) and math.isnan(r[k]))]
        return float(np.mean(vals)) if vals else float("nan")
    flagged = [r for r in rs if r.get("multi_modal")]
    return {
        "dataset": dataset, "subset": "all" if subset_n is None else f"n>={subset_n}",
        "classes": len(rs), "mean_n": float(np.mean([r["n"] for r in rs])),
        "sse1_mean": mean("sse1"), "sse2_mean": mean("sse2"),
        "sse3_mean": mean("sse3"), "reduction12_mean": mean("reduction12"),
        "reduction23_mean": mean("reduction23"), "sil1_mean": mean("sil1"),
        "sil2_mean": mean("sil2"), "within_mean": mean("within_mean"),
        "within_p95_mean": mean("within_p95"), "within_max_mean": mean("within_max"),
        "cross_nn_mean": mean("cross_nn_mean"), "overlap_rate_mean": mean("overlap_rate"),
        "multi_modal_classes": len(flagged),
        "multi_modal_rate": len(flagged) / len(rs),
        "bic_prefer_2": sum(1 for r in rs if r.get("bic2", float("nan")) < r.get("bic1", float("inf"))),
        "bic_prefer_3": sum(1 for r in rs if r.get("bic3", float("nan")) < r.get("bic2", float("inf"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    model, ck = load_msr_model(ROOT / "runs/orbit_msr/msr_nr2/model.pth",
                               device=args.device)
    out_dir = ROOT / "outputs" / "iclr27_phase4e" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}

    # real meta-dev
    labels = load_train_labels()
    feats = load_frame_features("train_known_mean")
    dev_classes = meta_classes("meta_dev_classes")
    dev_sids = [sid for sid, c in labels.items() if c in dev_classes and sid in feats]
    zs = embed_tracks(model, feats, dev_sids, args.device)
    datasets["real_meta_dev"] = ([zs[s] for s in dev_sids],
                                 [int(labels[s]) for s in dev_sids])

    # official novel
    gt = load_gt("pure")
    feats_o = {sid: f[:8] for sid, f in
               load_frame_features("gt_tracks_mean").items()}
    nov = [g for g in gt if g["protocol_role"] == "novel" and g["sample_id"] in feats_o]
    zs_o = embed_tracks(model, feats_o, [g["sample_id"] for g in nov], args.device)
    datasets["official_novel"] = ([zs_o[g["sample_id"]] for g in nov],
                                  [int(g["ground_truth_category_id"]) for g in nov])

    # synthetic long-stream (null control)
    rows, gt_rows, feats_ls, _ = load_stream_cache()
    syn_sids = [r["sample_id"] for r in rows
                if r["role"] == "novel" and int(r["class"]) >= 1000000
                and r["sample_id"] in feats_ls]
    zs_s = embed_tracks(model, feats_ls, syn_sids, args.device)
    cls_map = {r["sample_id"]: int(r["class"]) for r in rows if r["role"] == "novel"}
    datasets["synthetic_long"] = ([zs_s[s] for s in syn_sids],
                                  [cls_map[s] for s in syn_sids])

    all_class_rows = []
    agg_rows = []
    for ds, (zs, lab) in datasets.items():
        rows_c = class_diagnostics(zs, lab)
        rows_c = cross_class_overlap(zs, lab, rows_c)
        for r in rows_c:
            r["dataset"] = ds
            # multi-modal flag: meaningful 2-center gain, separated centers,
            # and silhouette not worse
            r["multi_modal"] = bool(
                r.get("reduction12") is not None
                and not (isinstance(r.get("reduction12"), float)
                         and math.isnan(r.get("reduction12", float("nan"))))
                and r["reduction12"] >= 0.15
                and r.get("two_center_sim", 1.0) < 0.95
                and (math.isnan(r.get("sil2", float("nan")))
                     or r.get("sil1", 0.0) <= r.get("sil2", 0.0)))
        all_class_rows.extend(rows_c)
        a1 = aggregate(rows_c, ds)
        a2 = aggregate(rows_c, ds, subset_n=8)
        if a1:
            agg_rows.append(a1)
        if a2:
            agg_rows.append(a2)
        print(json.dumps(a1, indent=1), flush=True)

    with open(out_dir / "multimodality_by_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_class_rows[0].keys()))
        w.writeheader()
        w.writerows(all_class_rows)
    with open(out_dir / "single_vs_multi_center_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print("saved multimodality CSVs", flush=True)


if __name__ == "__main__":
    main()
