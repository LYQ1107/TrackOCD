"""Train-side real similarity distribution + real-band hard negatives.

All statistics are computed from the frozen train-known set (48 classes,
train_known_mean frame features).  Official-validation statistics are never
used to choose the band.  The band is derived a priori from the train-side
distribution (documented) and can be overridden explicitly in training.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_msr.evaluate import embed_many, load_msr_model


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_class_geometry(zs, by_class, classes):
    protos = {}
    radii = {}
    disp = {}
    for c in classes:
        arr = np.stack([zs[s] for s in by_class[c]]).astype(np.float32)
        p = arr.mean(axis=0)
        p = p / (np.linalg.norm(p) + 1e-12)
        protos[c] = p
        cos = arr @ p
        radii[c] = float(np.percentile(1.0 - cos, 50))
        disp[c] = float(np.mean(1.0 - cos))
    return protos, radii, disp


def analyze(checkpoint, out_dir, sample_cap=4000, seed=2026):
    feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in feats:
            by_class[int(c)].append(sid)
    classes = sorted(by_class)
    all_sids = [s for c in classes for s in by_class[c]]
    model, ck = load_msr_model(ROOT / checkpoint, "cuda")
    zs, rels = embed_many(model, feats, all_sids, "cuda")
    protos, radii, disp = build_class_geometry(zs, by_class, classes)
    P = np.stack([protos[c] for c in classes]).astype(np.float32)

    rng = np.random.RandomState(seed)
    pos_sims = []
    neg_sims = []
    topk_sims = []
    intra = []
    for c in classes:
        arr = np.stack([zs[s] for s in by_class[c]]).astype(np.float32)
        cos = arr @ protos[c]
        pos_sims.extend(float(x) for x in cos)
        # nearest different-class prototype similarity
        other = [protos[d] for d in classes if d != c]
        Po = np.stack(other).astype(np.float32)
        sims = arr @ Po.T
        best_other = sims.max(axis=1)
        neg_sims.extend(float(x) for x in best_other)
        # top-4 wrong-prototype similarity
        order = np.argsort(sims, axis=1)[:, ::-1][:, :4]
        for i, o in enumerate(order):
            topk_sims.extend(float(sims[i, j]) for j in o)
    # random inter-class pair sample
    n_pairs = min(sample_cap, 200000)
    for _ in range(n_pairs):
        a, b = rng.choice(len(classes), size=2, replace=False)
        s1 = by_class[classes[a]][rng.randint(len(by_class[classes[a]]))]
        s2 = by_class[classes[b]][rng.randint(len(by_class[classes[b]]))]
        intra.append(float(np.dot(zs[s1], zs[s2])))
    intra = np.asarray(intra)
    pos = np.asarray(pos_sims)
    neg = np.asarray(neg_sims)
    topk = np.asarray(topk_sims)

    def q(a):
        return [float(x) for x in np.percentile(a, [5, 10, 25, 50, 75, 90, 95])]

    dist_rows = [{
        "set": "train_known",
        "n_tracks": len(all_sids),
        "n_classes": len(classes),
        "metric": m,
        "n_samples": len(v),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p05": q(v)[0], "p10": q(v)[1], "p25": q(v)[2],
        "p50": q(v)[3], "p75": q(v)[4], "p90": q(v)[5], "p95": q(v)[6],
    } for m, v in [("intra_class_similarity", pos),
                   ("nearest_different_class_similarity", neg),
                   ("top4_wrong_prototype_similarity", topk),
                   ("random_inter_class_pair_similarity", intra)]]
    _write_csv(out_dir / "real_similarity_distribution.csv", dist_rows)

    # band proposal: hard negatives live between nearest-different-class p10
    # and intra-class p50, clipped to [0.30, 0.90]; record the chosen band.
    band_lo = float(max(neg_q := q(neg)[1], 0.30))
    band_hi = float(min(q(pos)[3], 0.90))
    if band_hi <= band_lo:
        band_lo, band_hi = 0.50, 0.80
    band_rows = [{
        "set": "train_known",
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_source": ("train-side distribution only; "
                        "p10 nearest-different-class .. p50 intra-class"),
        "n_candidate_prototype_pairs": int(((P[:, None, :] @ P[None, :, :])
                                            > band_lo).sum()
                                           - len(classes)),
    }]
    _write_csv(out_dir / "hard_negative_band.csv", band_rows)
    (out_dir / "real_band_config.json").write_text(json.dumps({
        "checkpoint": checkpoint,
        "band": [band_lo, band_hi],
        "bands": {"nearest_different_p10": q(neg)[1],
                  "intra_p50": q(pos)[3]},
    }, indent=1))
    print("band:", band_lo, band_hi, flush=True)
    for r in dist_rows:
        print(r, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="runs/orbit_msr/msr_nr2/model.pth")
    ap.add_argument("--out_dir",
                    default="outputs/iclr27_phase4f/training")
    args = ap.parse_args()
    analyze(args.checkpoint, ROOT / args.out_dir)


if __name__ == "__main__":
    main()
