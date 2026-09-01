"""Compare original DINO track-mean vs ORBIT-D1 adapter geometry."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import load_model, embed_track
from src.orbit.protocol import load_frame_features, load_gt, load_stream, load_mean_features


def mean_cos(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = np.asarray(a); b = np.asarray(b)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((a @ b.T).mean())


def main():
    device = "cuda"
    model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=device)
    feats = load_frame_features("gt_tracks_mean")
    means = load_mean_features("gt_tracks_mean")
    gt = load_gt("pure")
    gt_by_sid = {g["sample_id"]: g for g in gt}
    sids = [r["sample_id"] for r in load_stream("pure", "main_seed1027") if r["sample_id"] in feats]
    orbit = {}
    for sid in sids:
        orbit[sid], _ = embed_track(model, feats[sid], device)
    by_class = defaultdict(list)
    for sid in sids:
        g = gt_by_sid[sid]
        if g["protocol_role"] == "distractor":
            continue
        by_class[(g["protocol_role"], g["ground_truth_category_id"])].append(sid)
    known_ids = [sid for (role, c), ls in by_class.items() if role in ("supported_known", "zero_shot_known") for sid in ls]
    novel_ids = [sid for (role, c), ls in by_class.items() if role == "novel" for sid in ls]
    known_classes = [(c, ls) for (role, c), ls in by_class.items() if role in ("supported_known", "zero_shot_known")]
    novel_classes = [(c, ls) for (role, c), ls in by_class.items() if role == "novel"]
    rows = []
    for space_name, space in [("dino_mean", means), ("orbit_d1", orbit)]:
        def class_means(classes):
            out = {}
            for c, ls in classes:
                out[c] = np.mean([space[s] for s in ls[:40]], axis=0)
            return out
        km = class_means(known_classes); nm = class_means(novel_classes)
        intra_k = np.mean([mean_cos([space[s] for s in ls[:30]], [space[s] for s in ls[:30]]) for c, ls in known_classes if len(ls) > 1])
        intra_n = np.mean([mean_cos([space[s] for s in ls[:30]], [space[s] for s in ls[:30]]) for c, ls in novel_classes if len(ls) > 1])
        inter_k = np.mean([mean_cos([km[a]], [km[b]]) for i, a in enumerate(km) for b in list(km)[i+1:i+11]])
        inter_n = np.mean([mean_cos([nm[a]], [nm[b]]) for i, a in enumerate(nm) for b in list(nm)[i+1:i+11]])
        kn_nn = np.mean([max(np.dot(space[s], km[c]) for c in km) for s in known_ids[:500]])
        rows.append({
            "space": space_name,
            "known_intra": intra_k, "known_inter": inter_k,
            "novel_intra": intra_n, "novel_inter": inter_n,
            "known_to_novel_nn": kn_nn,
        })
    out = ROOT / "outputs/orbit_bc/audit/geometry_before_after.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(rows)


if __name__ == "__main__":
    main()
