from __future__ import annotations

import numpy as np

from src.evaluation.metrics import hungarian_acc
from src.ocd_v2.common import proxy_split, build_prototypes


def calibrate_b2_threshold(feats, labels, seed=1027):
    """Independent B2/router threshold calibration on the train-known proxy,
    using exactly the official TrackOCD B2 grid [0.45, 0.50, ..., 0.95] and
    the proxy-novel Hungarian ACC objective (src/ocd/online_ncm.py).
    Returns (best_threshold, curve). No val labels used."""
    pk, pn = proxy_split(labels, seed=seed)
    ids = sorted(s for s, c in labels.items() if c in pn and s in feats)
    y = np.array([labels[s] for s in ids])
    protos = build_prototypes(feats, labels, pk)
    curve = []
    best = (0.45, -1.0)
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds = []
        novel = {}
        counts = {}
        nid = 100000
        for s in ids:
            x = feats[s]
            x = x / (np.linalg.norm(x) + 1e-12)
            best_k, best_s = None, -1.0
            for cid, p in protos.items():
                sim = float(np.dot(x, p))
                if sim > best_s:
                    best_s, best_k = sim, cid
            if best_s >= thr:
                preds.append(best_k)
                continue
            best_n, best_ns = None, -1.0
            for cid, c in novel.items():
                sim = float(np.dot(x, c))
                if sim > best_ns:
                    best_ns, best_n = sim, cid
            if best_ns >= thr:
                novel[best_n] = (novel[best_n] * counts[best_n] + x) / (counts[best_n] + 1)
                novel[best_n] /= np.linalg.norm(novel[best_n]) + 1e-12
                counts[best_n] += 1
                preds.append(best_n)
            else:
                novel[nid] = x.copy()
                counts[nid] = 1
                preds.append(nid)
                nid += 1
        pv = np.array(preds)
        uniq = sorted(set(int(v) for v in pv))
        remap = {v: i for i, v in enumerate(uniq)}
        pv = np.array([remap[int(v)] for v in pv])
        acc = hungarian_acc(y, pv)[0]
        curve.append({"threshold": float(thr), "proxy_novel_acc": float(acc)})
        if acc > best[1]:
            best = (float(thr), acc)
    return best[0], curve
