"""Frozen-feature BSP replay (no representation adaptation ablation).

Uses the corrected TSE projection as the trajectory representation (causal
per-track running mean), the legal supported-known TRAIN Gaussian statistics
as known states, and the same posterior-predictive assign-vs-spawn decision
as Architecture A, with a single rho.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase8a.model.bayesian_process import (
    SemanticStateSet,
    TrajectoryState,
    bsp_step,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--rho", type=float, default=-200.0)
    ap.add_argument("--sigma2", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:9")
    args = ap.parse_args()

    dev = torch.device(args.device)
    tse, _, _ = load_tse(dev)
    st = dict(np.load(ROOT / "outputs/iclr27_phase7b/assets/known_stats.npz"))
    known_ids = [int(x) for x in st["known_ids"]]
    states = SemanticStateSet(
        dim=128, sigma2=args.sigma2, rho=args.rho, max_slots=4096)
    states.init_known(st["known_ids"], st["mu"], st["counts"].astype(float))

    with open(ROOT / args.proposals) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    feats = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    z_all = project(dev, tse, feats)
    track_state = {}
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    sem_kscore = [""] * len(rows)
    chrono = sorted(
        range(len(rows)),
        key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
                       int(rows[i].get("proposal_local_id") or 0),
                       int(rows[i]["track_id"])))
    for i in chrono:
        key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
        tr = track_state.get(key)
        if tr is None:
            tr = TrajectoryState(dim=128)
            track_state[key] = tr
        action, sid, slot, scores = bsp_step(
            z_all[i], tr, states, known_ids, key,
            rho=args.rho, sigma2=args.sigma2)
        sem_action[i] = ["known", "existing", "new"][action]
        sem_sid[i] = str(sid) if sid is not None else ""
        if action in (0, 1) and scores is not None:
            p_assign = 1.0 / (1.0 + np.exp(args.rho - scores.max()))
            sem_kscore[i] = f"{float(p_assign):.6f}"

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["sem_kscore"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + extra)
        w.writeheader()
        for i, r in enumerate(rows):
            r = dict(r)
            r["sem_action"] = sem_action[i]
            r["sem_sid"] = sem_sid[i]
            r["sem_kscore"] = sem_kscore[i]
            w.writerow(r)
    print(Counter(sem_action))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
