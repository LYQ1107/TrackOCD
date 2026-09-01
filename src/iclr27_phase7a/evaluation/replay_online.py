"""Online strict-causal semantic replay for Phase 7A.

Reads a frozen physical-stream proposals CSV + DINOv2 feats and recomputes
only sem_action / sem_sid:
  --mode racc : Reliability-Aware Causal Category Memory (trained head)
  --mode ema  : Phase 6C-style simple EMA baseline (frozen control)
The memory is global across the whole stream (chronological order), every
row gets an immediate immutable decision, and no future/GT information is
used.
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
from src.iclr27_phase6c.model.tse import TSE, KnownAnchors
from src.iclr27_phase7a.model.reliability_memory import (
    MemoryState,
    RACCHead,
    TrackStats,
    online_step,
)
from src.iclr27_phase7a.model.reliability_memory_v2 import v2_step


def load_tse(device, ckpt):
    state = torch.load(ROOT / ckpt, map_location=device, weights_only=False)
    model = TSE().to(device)
    model.load_pca(ROOT / "outputs/iclr27_phase6c/assets/pca.npz")
    model.load_state_dict(state["model"])
    model.eval()
    anchors = KnownAnchors(state["known_ids"]).to(device)
    anchors.load_state_dict(state["anchors"])
    with torch.no_grad():
        an = anchors.normalized().cpu().numpy().astype(np.float32)
    return model, an, np.asarray(state["known_ids"], dtype=np.int64)


def project(device, model, feats, batch=1024):
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch):
            x = torch.from_numpy(feats[i:i + batch].astype(np.float32)).to(device)
            out.append(model.project(x).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def replay_racc(rows, z_all, model, anchors, known_ids, policy, device,
                visible_ids=None, known_tau=0.65, use_rel=True,
                use_maturity=True, sem_only=False):
    if visible_ids is None:
        visible_ids = known_ids
    visible_mask = np.isin(known_ids, np.asarray(visible_ids))
    mem = MemoryState(dim=128)
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
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackStats()
            track_stats[key] = ts
        bbox = np.asarray(json.loads(r["bbox_xyxy"]), dtype=np.float64)
        res = online_step(
            policy, z_all[i], mem, anchors, visible_mask, known_ids, ts,
            float(r["score"]), int(r.get("prior_hits") or 0), bbox,
            int(r["frame_id"]), key, known_tau=known_tau, use_rel=use_rel,
            use_maturity=use_maturity, sem_only=sem_only)
        if res["decision"] == 0:
            sem_action[i] = "known"
        elif res["decision"] == 1:
            sem_action[i] = "existing"
        else:
            sem_action[i] = "new"
        sem_sid[i] = str(res["sid"]) if res["sid"] is not None else ""
    return sem_action, sem_sid


def replay_ema(rows, z_all, anchors, known_ids, tau=0.65, ema_alpha=0.30):
    state = {}
    novel = {}
    counts = {}
    next_id = 100000
    chrono = sorted(
        rows,
        key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                       int(r.get("proposal_local_id") or 0),
                       int(r["track_id"])),
    )
    row_index = {id(r): i for i, r in enumerate(rows)}
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        z = z_all[i]
        if key in state:
            state[key] = ((1 - ema_alpha) * state[key] + ema_alpha * z)
            state[key] /= np.linalg.norm(state[key]) + 1e-12
        else:
            state[key] = z.copy()
        h = state[key]
        ksims = anchors @ h
        ki = int(np.argmax(ksims))
        if ksims[ki] >= tau:
            sem_action[i] = "known"
            sem_sid[i] = str(known_ids[ki])
            continue
        if novel:
            sids = list(novel)
            sims = np.asarray([float(novel[s] @ h) for s in sids])
            j = int(np.argmax(sims))
            if sims[j] >= tau:
                sid = sids[j]
                novel[sid] = ((novel[sid] * counts[sid] + h)
                              / (counts[sid] + 1))
                novel[sid] /= np.linalg.norm(novel[sid]) + 1e-12
                counts[sid] += 1
                sem_action[i] = "existing"
                sem_sid[i] = str(sid)
                continue
        sem_action[i] = "new"
        sem_sid[i] = str(next_id)
        novel[next_id] = h.copy()
        counts[next_id] = 1
        next_id += 1
    return sem_action, sem_sid


def replay_racc2(rows, z_all, anchors, known_ids, cfg, visible_ids=None):
    if visible_ids is None:
        visible_ids = known_ids
    visible_mask = np.isin(known_ids, np.asarray(visible_ids))
    mem = MemoryState(dim=128)
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
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackStats()
            track_stats[key] = ts
        bbox = np.asarray(json.loads(r["bbox_xyxy"]), dtype=np.float64)
        res = v2_step(
            z_all[i], mem, anchors, visible_mask, known_ids, ts,
            float(r["score"]), int(r.get("prior_hits") or 0), bbox,
            int(r["frame_id"]), key, cfg)
        sem_action[i] = ["known", "existing", "new"][res["decision"]]
        sem_sid[i] = str(res["sid"]) if res["sid"] is not None else ""
    return sem_action, sem_sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--tse-ckpt",
                    default="outputs/iclr27_phase6c/training/tse_main/checkpoint.pth")
    ap.add_argument("--mode", choices=["racc", "ema", "racc2"], default="racc")
    ap.add_argument("--head-ckpt", default=None)
    ap.add_argument("--tau", type=float, default=0.65)
    ap.add_argument("--ema-alpha", type=float, default=0.30)
    ap.add_argument("--known-tau", type=float, default=0.65)
    ap.add_argument("--no-rel", action="store_true")
    ap.add_argument("--no-maturity", action="store_true")
    ap.add_argument("--sem-only", action="store_true")
    ap.add_argument("--v2-nu", type=float, default=2.0)
    ap.add_argument("--v2-pen", type=float, default=0.05)
    ap.add_argument("--v2-tau-attach", type=float, default=0.40)
    ap.add_argument("--v2-rel-birth", type=float, default=0.30)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--visible-ids", default=None)
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, anchors, known_ids = load_tse(dev, args.tse_ckpt)
    with open(ROOT / args.proposals) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    feats = np.load(ROOT / args.feats)["feats"].astype(np.float32)
    assert len(rows) == len(feats)
    z_all = project(dev, model, feats)
    visible = None
    if args.visible_ids:
        visible = [int(x) for x in args.visible_ids.split(",")]
    if args.mode == "racc":
        assert args.head_ckpt, "racc requires --head-ckpt"
        policy = RACCHead().to(dev)
        ck = torch.load(ROOT / args.head_ckpt, map_location=dev,
                        weights_only=False)
        policy.load_state_dict(ck["policy"])
        policy.eval()
        sem_action, sem_sid = replay_racc(
            rows, z_all, model, anchors, known_ids, policy, dev, visible,
            known_tau=args.known_tau, use_rel=not args.no_rel and
            not args.sem_only, use_maturity=not args.no_maturity,
            sem_only=args.sem_only)
    else:
        if args.mode == "racc2":
            cfg = {
                "tau_k": args.known_tau,
                "nu": args.v2_nu,
                "pen": args.v2_pen,
                "tau_attach": args.v2_tau_attach,
                "rel_birth": args.v2_rel_birth,
            }
            sem_action, sem_sid = replay_racc2(
                rows, z_all, anchors, known_ids, cfg, visible)
        else:
            assert args.mode == "ema"
            sem_action, sem_sid = replay_ema(
                rows, z_all, anchors, known_ids, tau=args.tau,
                ema_alpha=args.ema_alpha)
    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows):
            r = dict(r)
            r["sem_action"] = sem_action[i]
            r["sem_sid"] = sem_sid[i]
            w.writerow(r)
    print(Counter(sem_action))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
