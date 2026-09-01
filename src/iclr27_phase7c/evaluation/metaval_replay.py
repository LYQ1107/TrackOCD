"""Real strict-causal meta-val replay for KPOC frontier validation.

Replays the legal class-level meta-val with a trained checkpoint and a
known-offset, then computes class-level Known / First / Reuse metrics with
the exact online memory dynamics (not the logit approximation).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
from src.iclr27_phase7b.evaluation.replay_tose import load_stats, precompute
from src.iclr27_phase7c.training.train_kpoc import replay_track_ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--mode", choices=["hard", "random"], default="hard")
    ap.add_argument("--known-offset", type=float, default=0.0)
    ap.add_argument("--val-fp-keep", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=2717)
    ap.add_argument("--base-dim", type=int, default=None)
    ap.add_argument("--legacy-features", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, anchors, known_ids = load_tse(dev)
    stats = load_stats()
    ep = {k: np.asarray(v) for k, v in np.load(
        ROOT / f"outputs/iclr27_phase7c/assets/metaval_{args.mode}.npz").items()}
    split = json.loads((ROOT /
        f"outputs/iclr27_phase7c/assets/class_split_{args.mode}.json")
        .read_text())
    z_all = project(dev, model, ep["feats"].astype(np.float32))
    h_all = replay_track_ema(ep, z_all)
    asims_all, loglik_all = precompute(h_all, anchors, stats)

    traj_feats = 3 if args.legacy_features else 5
    bdim = args.base_dim or len(base_feature_names(True, True, traj_feats))
    policy = TOSEHead(base_dim=bdim).to(dev)
    ck = torch.load(ROOT / args.head_ckpt, map_location=dev,
                    weights_only=False)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    visible = np.isin(
        known_ids,
        np.asarray(sorted(set(split["train_visible"])), dtype=np.int64))
    mem = EMAMemory(dim=128)
    track_stats = {}
    slot_class = {}
    class_seen = {}
    class_slot = {}
    n_known = n_first = n_attach = 0
    acc_known = acc_first = acc_attach = 0
    # chronological replay (episodes are packed per video in order)
    for i in range(len(z_all)):
        if int(ep["row_split"][i]) < 0 and (
                (i * 2654435761 + args.seed) % 1000 / 10.0
                >= args.val_fp_keep * 100):
            continue
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackState()
            track_stats[key] = ts
        rs = int(ep["row_split"][i])
        cat = int(ep["gt_category_id"][i])
        target = None
        if rs == 0:
            target = 0
        elif rs == 1:
            if cat not in class_seen:
                target = 2
                class_seen[cat] = key
            elif cat not in class_slot:
                target = 2  # class seen but no durable slot yet
            else:
                target = 1  # same physical track after birth
        res = tose_step(
            policy, z_all[i], mem, anchors, visible, known_ids, stats, ts,
            int(ep["frame_ids"][i]), key, slot_class=cat if rs == 1 else -1,
            target_slot=None,
            asims_row=asims_all[i], loglik_row=loglik_all[i],
            known_offset=args.known_offset, traj_feats=traj_feats)
        if res["decision"] == 2 and rs == 1 and res["slot_idx"] is not None:
            slot_class.setdefault(res["slot_idx"], cat)
            class_slot.setdefault(cat, res["slot_idx"])
        if rs == 0:
            n_known += 1
            acc_known += (res["decision"] == 0)
        elif rs == 1:
            if target == 2:
                n_first += 1
                acc_first += (res["decision"] == 2)
            elif target == 1:
                n_attach += 1
                acc_attach += (
                    res["decision"] == 1
                    and res["slot_idx"] is not None
                    and slot_class.get(res["slot_idx"]) == cat)
    out = {
        "known_offset": args.known_offset,
        "n_known": n_known,
        "n_first": n_first,
        "n_attach": n_attach,
        "known_acc": acc_known / max(n_known, 1),
        "first_acc": acc_first / max(n_first, 1),
        "reuse_acc": acc_attach / max(n_attach, 1),
        "novel_score": (acc_first / max(n_first, 1))
        * (acc_attach / max(n_attach, 1)),
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
