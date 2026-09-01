"""Build the active-known prototype bank (TRAIN-only 48 categories).

Prototypes are mean R3-TSR final states over real Q1 train tracklets.
The bank is indexed by sorted supported-known category ids. Episode
builders slice it to episode-known categories only (genuine OOV).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4t.episodes import RealStreamStore
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase4v.evidence import load_known_branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    store = RealStreamStore.__new__(RealStreamStore)
    rows = list(csv.DictReader(open(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
        r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
        r["q_phys"] = json.loads(r["q_phys"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["gt_role"] = r["gt_role"]; r["gt_category_id"] = int(r["gt_category_id"])
        r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
        r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
        r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
    feats = np.load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz")["feats"]
    store.rows = rows
    store.feats = feats
    store.row_index = {id(r): i for i, r in enumerate(rows)}
    from src.iclr27_phase4t.stream_data import build_tracklets
    store.tracklets = build_tracklets(rows)

    ktsr, _ = load_known_branch(args.device)
    cat_list = sorted(known_ids())
    cat_index = {c: i for i, c in enumerate(cat_list)}
    sums = np.zeros((len(cat_list), 256), dtype=np.float64)
    counts = np.zeros(len(cat_list), dtype=np.int64)
    with torch.no_grad():
        for key, tl in store.tracklets.items():
            c = tl["gt_category_id"]
            if tl["role"] != "known" or c not in cat_index:
                continue
            z, q = store.tracklet_seq(key)
            zt = torch.from_numpy(z).to(args.device)
            qt = torch.from_numpy(q).to(args.device)
            states = ktsr.embed_sequence(zt, qt)
            s = states[-1].cpu().numpy()
            sums[cat_index[c]] += s
            counts[cat_index[c]] += 1
    protos = sums / np.maximum(counts[:, None], 1)
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    out = ROOT / "outputs/iclr27_phase4w/active_universe"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "active_protos.npz",
                        protos=protos.astype(np.float32),
                        cat_ids=np.asarray(cat_list, dtype=np.int64))
    (out / "proto_counts.json").write_text(json.dumps(
        {str(c): int(counts[i]) for i, c in enumerate(cat_list)}, indent=2))
    # fixed frozen random projection of s_k (seeded, deterministic)
    rng = np.random.default_rng(20260815)
    proj = rng.standard_normal((256, 32)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12
    np.savez_compressed(out / "sk_proj.npz", proj=proj)
    print("saved prototypes", protos.shape, "categories", len(cat_list))
    print("counts min/max", int(counts.min()), int(counts.max()))


if __name__ == "__main__":
    main()
