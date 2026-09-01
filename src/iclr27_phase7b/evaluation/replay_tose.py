"""Strict-causal TOSE replay for Q1 DEV / locked heldout streams.

Reads a frozen physical-stream proposals CSV + DINOv2 feats and recomputes
only sem_action / sem_sid (plus diagnostic knownness columns). The memory is
global across the whole stream in chronological order; every row gets an
immediate immutable decision; no future/GT information is used.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase7b.model.explainability import (
    EMAMemory,
    TOSEHead,
    TOSELinearHead,
    TrackState,
    base_feature_names,
    tose_step,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def load_stats():
    return dict(np.load(
        ROOT / "outputs/iclr27_phase7b/assets/known_stats.npz"))


def replay_track_ema(rows, z_all, alpha=0.30):
    ema = {}
    h_all = np.zeros_like(z_all)
    for i in range(len(rows)):
        key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
        e = ema.get(key)
        z = z_all[i]
        if e is None:
            e = z.copy()
        else:
            e = (1 - alpha) * e + alpha * z
            e /= (np.linalg.norm(e) + 1e-12)
        ema[key] = e
        h_all[i] = e
    return h_all


def precompute(h_all, anchors, stats):
    asims = h_all @ anchors.T
    diff = h_all[:, None, :] - stats["mu"][None, :, :]
    mahal2 = np.sum(diff * diff / stats["sigma2"][None, :, :], axis=2)
    loglik = -0.5 * (mahal2 + stats["logdet"][None, :])
    return asims.astype(np.float32), loglik.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--tse-ckpt",
                    default="outputs/iclr27_phase6c/training/tse_main/checkpoint.pth")
    ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-dist", action="store_true", default=True)
    ap.add_argument("--use-traj", action="store_true", default=True)
    ap.add_argument("--no-dist", action="store_true")
    ap.add_argument("--frame-level", action="store_true")
    ap.add_argument("--classifier-conf", action="store_true")
    ap.add_argument("--known-tau", type=float, default=-1.0)
    ap.add_argument("--head-type", choices=["mlp", "linear"], default="mlp")
    args = ap.parse_args()
    if args.no_dist:
        args.use_dist = False
    if args.frame_level:
        args.use_traj = False
    if args.classifier_conf:
        args.use_dist = False
        args.use_traj = False

    dev = torch.device(args.device)
    model, anchors, known_ids = load_tse(dev, args.tse_ckpt)
    stats = load_stats()
    with open(ROOT / args.proposals) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    feats = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    assert len(rows) == len(feats)
    z_all = project(dev, model, feats)
    h_all = replay_track_ema(rows, z_all)
    asims_all, loglik_all = precompute(h_all, anchors, stats)

    base_names = base_feature_names(args.use_dist, args.use_traj)
    if args.head_type == "linear":
        policy = TOSELinearHead(
            base_dim=len(base_names), use_dist=args.use_dist,
            use_traj=args.use_traj).to(dev)
    else:
        policy = TOSEHead(base_dim=len(base_names)).to(dev)
    ck = torch.load(ROOT / args.head_ckpt, map_location=dev,
                    weights_only=False)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    visible_mask = np.ones(len(known_ids), dtype=bool)
    mem = EMAMemory(dim=128)
    track_stats = {}
    chrono = sorted(
        rows,
        key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                       int(r.get("proposal_local_id") or 0),
                       int(r["track_id"])),
    )
    row_index = {id(r): i for i, r in enumerate(rows)}
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    sem_kscore = [""] * len(rows)
    sem_kmahal = [""] * len(rows)
    sem_ksim = [""] * len(rows)
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackState()
            track_stats[key] = ts
        res = tose_step(
            policy, z_all[i], mem, anchors, visible_mask, known_ids, stats,
            ts, int(r["frame_id"]), key,
            use_dist=args.use_dist, use_traj=args.use_traj,
            known_tau=args.known_tau if args.known_tau >= 0 else None,
            asims_row=asims_all[i], loglik_row=loglik_all[i])
        sem_action[i] = ["known", "existing", "new"][res["decision"]]
        sem_sid[i] = str(res["sid"]) if res["sid"] is not None else ""
        sem_kscore[i] = f"{res['kscore']:.6f}"
        sem_kmahal[i] = f"{res['kmahal']:.6f}"
        sem_ksim[i] = f"{res['ksim']:.6f}"

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["sem_kscore", "sem_kmahal", "sem_ksim"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + extra)
        w.writeheader()
        for i, r in enumerate(rows):
            r = dict(r)
            r["sem_action"] = sem_action[i]
            r["sem_sid"] = sem_sid[i]
            r["sem_kscore"] = sem_kscore[i]
            r["sem_kmahal"] = sem_kmahal[i]
            r["sem_ksim"] = sem_ksim[i]
            w.writerow(r)
    print(Counter(sem_action))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
