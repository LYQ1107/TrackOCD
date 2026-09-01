"""Strict-causal BSP replay for any proposals stream (DEV / heldout).

Loads a trained CausalTrajectoryAdapter + the legal supported-known TRAIN
centroids, replays every proposal row in chronological order with immediate
immutable decisions, and writes sem_action / sem_sid / sem_kscore columns so
the standard strict evaluators can be applied unchanged.
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
from src.iclr27_phase8a.model.adapter import (
    CausalTrajectoryAdapter,
    TorchSemanticStateSet,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def load_rows(path):
    with open(ROOT / path) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    return rows, fieldnames


def compute_centroids(adapter, ep, z_all, device):
    st = dict(np.load(ROOT / "outputs/iclr27_phase7b/assets/known_stats.npz"))
    known_ids = [int(x) for x in st["known_ids"]]
    cls_idx = {c: i for i, c in enumerate(known_ids)}
    track_state = {}
    sums = torch.zeros(len(known_ids), adapter.dim, device=device)
    cnt = torch.zeros(len(known_ids), device=device)
    with torch.no_grad():
        for i in range(len(z_all)):
            if int(ep["gt_role"][i]) != 1:
                continue
            c = int(ep["gt_category_id"][i])
            if c not in cls_idx:
                continue
            key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            z = torch.from_numpy(z_all[i]).to(device).unsqueeze(0)
            h, state = adapter(z, prev)
            track_state[key] = state.detach()
            sums[cls_idx[c]] += h[0]
            cnt[cls_idx[c]] += 1.0
    mu = torch.nn.functional.normalize(
        sums / torch.clamp(cnt, min=1.0)[:, None], dim=-1)
    return mu, cnt, known_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--adapter-ckpt", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--sigma2", type=float, default=None)
    ap.add_argument("--rho", type=float, default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(ROOT / args.adapter_ckpt, map_location=dev,
                    weights_only=False)
    adapter = CausalTrajectoryAdapter(
        dim=ck.get("args", {}).get("dim", 128),
        rho_init=float(ck.get("rho", 40.0)),
        sigma2=args.sigma2 if args.sigma2 is not None
        else ck.get("sigma2", 0.05)).to(dev)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    if args.rho is not None:
        with torch.no_grad():
            adapter.rho.fill_(args.rho)

    tse, _, _ = load_tse(dev)
    ep = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7c/assets/metaval_hard.npz").items()}
    z_anchor = project(dev, tse, ep["feats"].astype(np.float32))
    mu, cnt, known_ids = compute_centroids(adapter, ep, z_anchor, dev)

    rows, fieldnames = load_rows(args.proposals)
    feats = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    assert len(rows) == len(feats)
    z_all = project(dev, tse, feats)

    states = TorchSemanticStateSet(
        dim=adapter.dim, max_slots=4096, sigma2=adapter.sigma2).to(dev)
    states.init_known(mu.detach(), cnt.detach())
    track_state = {}
    track_count = {}
    chrono = sorted(
        range(len(rows)),
        key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
                       int(rows[i].get("proposal_local_id") or 0),
                       int(rows[i]["track_id"])))
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    sem_kscore = [""] * len(rows)
    sem_slot = [""] * len(rows)
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            z = torch.from_numpy(z_all[i]).to(dev).unsqueeze(0)
            h, state = adapter(z, prev)
            h = h[0]
            w = float(track_count.get(key, 0) + 1)
            track_count[key] = int(w)
            track_state[key] = state.detach()
            logits = states.logits(h, w, adapter.rho)
            scores = logits[:states.n]
            rho = logits[states.n]
            logsumexp = torch.logsumexp(scores, dim=0) if states.n else rho
            p_assign = 1.0 / (1.0 + torch.exp(rho - logsumexp))
            pred = int(torch.argmax(logits))
            if pred == states.n:
                slot = states.spawn(h, w)
                if slot is not None:
                    sem_action[i] = "new"
                    sem_sid[i] = str(100000 + slot)
                    sem_slot[i] = str(slot)
                else:
                    best = int(torch.argmax(scores))
                    prov = int(states.provenance[best])
                    states.assign(best, h, w)
                    sem_action[i] = "known" if prov == 0 else "existing"
                    sem_sid[i] = (str(known_ids[best]) if prov == 0
                                  else str(100000 + best))
                    sem_slot[i] = str(best)
            else:
                slot = pred
                prov = int(states.provenance[slot])
                states.assign(slot, h, w)
                sem_action[i] = "known" if prov == 0 else "existing"
                sem_sid[i] = (str(known_ids[slot]) if prov == 0
                              else str(100000 + slot))
                sem_slot[i] = str(slot)
            sem_kscore[i] = f"{float(p_assign):.6f}"

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extra = ["sem_kscore", "sem_slot"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + extra)
        w.writeheader()
        for i, r in enumerate(rows):
            r = dict(r)
            r["sem_action"] = sem_action[i]
            r["sem_sid"] = sem_sid[i]
            r["sem_kscore"] = sem_kscore[i]
            r["sem_slot"] = sem_slot[i]
            w.writerow(r)
    print(Counter(sem_action))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
