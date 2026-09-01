"""Strict-causal KPOC replay for Q1 DEV / heldout with a calibration offset.

Identical memory/head mechanics to Phase 7B replay, plus `--known-offset`
applied to the KNOWN logit before the unified argmax. The offset is selected
on the legal meta-val frontier and frozen here.
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
    TrackState,
    base_feature_names,
    tose_step,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project
from src.iclr27_phase7b.evaluation.replay_tose import (
    load_stats,
    precompute,
    replay_track_ema,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--known-offset", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, anchors, known_ids = load_tse(dev)
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

    policy = TOSEHead(
        base_dim=len(base_feature_names(True, True))).to(dev)
    ck = torch.load(ROOT / args.head_ckpt, map_location=dev,
                    weights_only=False)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    visible = np.ones(len(known_ids), dtype=bool)
    mem = EMAMemory(dim=128)
    track_stats = {}
    chrono = sorted(
        rows,
        key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                       int(r.get("proposal_local_id") or 0),
                       int(r["track_id"])))
    row_index = {id(r): i for i, r in enumerate(rows)}
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    sem_kscore = [""] * len(rows)
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackState()
            track_stats[key] = ts
        res = tose_step(
            policy, z_all[i], mem, anchors, visible, known_ids, stats, ts,
            int(r["frame_id"]), key,
            asims_row=asims_all[i], loglik_row=loglik_all[i],
            known_offset=args.known_offset)
        sem_action[i] = ["known", "existing", "new"][res["decision"]]
        sem_sid[i] = str(res["sid"]) if res["sid"] is not None else ""
        sem_kscore[i] = f"{res['kscore']:.6f}"

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
