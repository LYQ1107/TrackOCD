"""Train-side real similarity distribution audit (Phase 4F).

Computes, in the frozen C1-adapted representation of train-known tracks:
same-class positive similarity, nearest-wrong negative similarity, top-k
wrong-prototype similarity, and the overlap bands that define real hard
negatives.  Official validation data is never used here.
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_msr.evaluate import embed_many, load_msr_model


def main():
    import pathlib
    out = pathlib.Path(f"{ROOT}/outputs/iclr27_phase4f/training")
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_msr_model(f"{ROOT}/runs/orbit_msr/msr_nr2/model.pth",
                               "cuda")
    feats = {sid: f[:8] for sid, f in
             load_frame_features("train_known_mean").items()}
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in feats:
            by_class[int(c)].append(sid)
    classes = sorted(by_class)
    sids = [sid for c in classes for sid in by_class[c]]
    zs, _ = embed_many(model, feats, sids, "cuda")
    protos = {}
    for c in classes:
        z = np.stack([zs[sid] for sid in by_class[c]])
        p = z.mean(axis=0)
        protos[c] = p / (np.linalg.norm(p) + 1e-12)
    P = np.stack([protos[c] for c in classes]).astype(np.float32)
    rows = []
    for c in classes:
        for sid in by_class[c]:
            z = zs[sid]
            sims = P @ z
            order = np.argsort(sims)[::-1]
            cidx = classes.index(c)
            same = float(sims[cidx])
            wrong = [float(sims[o]) for o in order if int(o) != cidx]
            rows.append({
                "class": c, "sample_id": sid,
                "positive_sim": same,
                "nearest_wrong_sim": wrong[0] if wrong else -1.0,
                "second_wrong_sim": wrong[1] if len(wrong) >= 2 else -1.0,
                "top5_wrong_mean": float(np.mean(wrong[:5])) if wrong else -1.0,
                "margin": same - wrong[0] if wrong else 0.0,
            })
    with open(out / "real_similarity_distribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pos = np.array([r["positive_sim"] for r in rows])
    nw = np.array([r["nearest_wrong_sim"] for r in rows])
    t5 = np.array([r["top5_wrong_mean"] for r in rows])
    n_high = int((nw >= 0.5).sum())
    n_band = int(((nw >= 0.5) & (nw <= 0.8)).sum())
    band_rows = [
        {"quantity": "positive_sim", "p05": float(np.percentile(pos, 5)),
         "p25": float(np.percentile(pos, 25)), "p50": float(np.percentile(pos, 50)),
         "p75": float(np.percentile(pos, 75)), "p95": float(np.percentile(pos, 95)),
         "mean": float(pos.mean())},
        {"quantity": "nearest_wrong_sim", "p05": float(np.percentile(nw, 5)),
         "p25": float(np.percentile(nw, 25)), "p50": float(np.percentile(nw, 50)),
         "p75": float(np.percentile(nw, 75)), "p95": float(np.percentile(nw, 95)),
         "mean": float(nw.mean())},
        {"quantity": "top5_wrong_mean", "p05": float(np.percentile(t5, 5)),
         "p25": float(np.percentile(t5, 25)), "p50": float(np.percentile(t5, 50)),
         "p75": float(np.percentile(t5, 75)), "p95": float(np.percentile(t5, 95)),
         "mean": float(t5.mean())},
        {"quantity": "nearest_wrong>=0.5", "count": int(n_high),
         "share": float(n_high / len(rows))},
        {"quantity": "nearest_wrong in [0.5,0.8]", "count": int(n_band),
         "share": float(n_band / len(rows))},
        {"quantity": "n_tracks", "count": len(rows)},
    ]
    with open(out / "hard_negative_band.csv", "w", newline="") as f:
        fn = []
        for r in band_rows:
            for k in r:
                if k not in fn:
                    fn.append(k)
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(band_rows)
    print("tracks", len(rows), "nearest_wrong>=0.5:", n_high,
          "in [0.5,0.8]:", n_band)
    print("pos quantiles:", np.percentile(pos, [5, 25, 50, 75, 95]).round(3))
    print("nw quantiles:", np.percentile(nw, [5, 25, 50, 75, 95]).round(3))


if __name__ == "__main__":
    main()
