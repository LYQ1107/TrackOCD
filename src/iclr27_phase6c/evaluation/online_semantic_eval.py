"""Phase 6C online semantic replay on a frozen physical-stream proposals CSV.

The physical stream (Phase 6B DSCT final Q1) is kept unchanged; only
sem_action/sem_sid are recomputed by the Trajectory Semantic Encoder (TSE)
with an online B2-style memory:

  - per physical track: causal EMA of TSE frame embeddings
  - KNOWN if max known-anchor cosine >= tau
  - EXISTING if max novel-memory cosine >= tau
  - NEW otherwise (slot birth, then reusable by future physical tracks)

No future frames, no retroactive relabel, no novel GT.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase6c.model.tse import TSE, KnownAnchors


def load_tse(ckpt, device):
    state = torch.load(ckpt, map_location=device)
    model = TSE().to(device)
    model.load_pca(ROOT / "outputs/iclr27_phase6c/assets/pca.npz")
    model.load_state_dict(state["model"])
    model.eval()
    anchors = KnownAnchors(state["known_ids"]).to(device)
    anchors.load_state_dict(state["anchors"])
    return model, anchors, state["known_ids"]


def calibrate_threshold(known_npz, model, anchors, device, seed=1027,
                        grid=None):
    """Official proxy rule: split the 48 supported-known classes into
    proxy-known / proxy-novel (seed 1027), build prototypes from
    proxy-known, replay B2 over proxy-novel, pick best Hungarian ACC.
    No val labels used."""
    from src.evaluation.metrics import hungarian_acc
    from src.ocd_v2.common import proxy_split
    feats = torch.from_numpy(known_npz["mean_feats"].astype(np.float32)).to(device)
    labels = known_npz["labels"]
    sample_ids = known_npz["sample_ids"]
    label_map = {sid: int(c) for sid, c in zip(sample_ids, labels)}
    pk, pn = proxy_split(label_map, seed=seed)
    pk_ids = sorted(sid for sid, c in label_map.items() if c in pk)
    pn_ids = sorted(sid for sid, c in label_map.items() if c in pn)
    with torch.no_grad():
        z = model.project(feats).cpu().numpy()
    protos = {}
    for c in pk:
        idx = [i for i, sid in enumerate(sample_ids) if label_map[sid] == c]
        m = z[idx].mean(axis=0)
        protos[c] = m / (np.linalg.norm(m) + 1e-12)
    order = {sid: i for i, sid in enumerate(pn_ids)}
    y = np.array([label_map[s] for s in pn_ids])
    grid = grid or [0.45 + 0.05 * i for i in range(11)]
    best = (0.45, -1.0, None)
    curves = []
    for thr in grid:
        novel = {}
        counts = {}
        nid = 100000
        preds = []
        for sid in pn_ids:
            x = z[list(sample_ids).index(sid)]
            best_k, best_s = None, -1.0
            for cid, p in protos.items():
                s = float(np.dot(x, p))
                if s > best_s:
                    best_s, best_k = s, cid
            if best_s >= thr:
                preds.append(best_k)
                continue
            best_n, best_ns = None, -1.0
            for cid, c in novel.items():
                s = float(np.dot(x, c))
                if s > best_ns:
                    best_ns, best_n = s, cid
            if best_ns >= thr:
                novel[best_n] = ((novel[best_n] * counts[best_n] + x)
                                 / (counts[best_n] + 1))
                novel[best_n] /= np.linalg.norm(novel[best_n]) + 1e-12
                counts[best_n] += 1
                preds.append(best_n)
            else:
                novel[nid] = x.copy()
                counts[nid] = 1
                preds.append(nid)
                nid += 1
        pv = np.asarray(preds)
        uniq = sorted(set(int(v) for v in pv))
        remap = {v: i for i, v in enumerate(uniq)}
        pv = np.array([remap[int(v)] for v in pv])
        acc = float(hungarian_acc(y, pv)[0])
        curves.append({"threshold": thr, "proxy_novel_acc": acc})
        if acc > best[1]:
            best = (thr, acc, curves[-1])
    return best[0], best[1], curves


def replay(rows, feats, model, anchors, device, tau, ema_alpha=0.30,
           use_ema=True, tau_known=None, tau_novel=None, novel_margin=0.0):
    tau_known = tau_known if tau_known is not None else tau
    tau_novel = tau_novel if tau_novel is not None else tau
    with torch.no_grad():
        z_all = model.project(
            torch.from_numpy(feats.astype(np.float32)).to(device)
        ).cpu().numpy()
    an = anchors.normalized().detach().cpu().numpy()
    known_ids = anchors.known_ids.cpu().numpy().tolist()
    chrono = sorted(
        rows,
        key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                       int(r.get("proposal_local_id") or 0), int(r["track_id"])),
    )
    row_index = {id(r): i for i, r in enumerate(rows)}
    state = {}
    novel = {}
    counts = {}
    next_id = 100000
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        z = z_all[i]
        if use_ema:
            if key not in state:
                state[key] = z.copy()
            else:
                state[key] = (1 - ema_alpha) * state[key] + ema_alpha * z
                state[key] /= np.linalg.norm(state[key]) + 1e-12
            h = state[key]
        else:
            h = z
        ksims = an @ h
        ki = int(np.argmax(ksims))
        if ksims[ki] >= tau_known:
            sem_action[i] = "known"
            sem_sid[i] = str(known_ids[ki])
            continue
        nsims = []
        for sid, c in novel.items():
            nsims.append((float(np.dot(c, h)), sid))
        if nsims:
            ns, sid = max(nsims)
            if ns >= tau_novel and ns >= float(ksims[ki]) + novel_margin:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--known-tau", type=float, default=None)
    ap.add_argument("--novel-tau", type=float, default=None)
    ap.add_argument("--novel-margin", type=float, default=0.0)
    ap.add_argument("--ema-alpha", type=float, default=0.30)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    model, anchors, _ = load_tse(ROOT / args.ckpt, dev)
    tau = args.tau
    cal = None
    if args.calibrate:
        known = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
        tau, acc, curves = calibrate_threshold(known, model, anchors, dev)
        cal = {"tau": tau, "proxy_novel_acc": acc, "curves": curves}
        print("calibrated tau", tau, "proxy acc", acc)
    assert tau is not None or (args.known_tau is not None
                               and args.novel_tau is not None), \
        "provide --tau or --calibrate"

    with open(args.proposals) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        df = list(reader)
    feats = np.load(ROOT / args.feats)["feats"]
    assert len(df) == len(feats)
    rows = [dict(r) for r in df]
    sem_action, sem_sid = replay(rows, feats, model, anchors, dev, tau,
                                 ema_alpha=args.ema_alpha, use_ema=args.use_ema,
                                 tau_known=args.known_tau,
                                 tau_novel=args.novel_tau,
                                 novel_margin=args.novel_margin)
    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(df):
            r = dict(r)
            r["sem_action"] = sem_action[i]
            r["sem_sid"] = sem_sid[i]
            writer.writerow(r)
    print(out_path)
    from collections import Counter
    print(Counter(sem_action))
    if cal:
        (out_path.parent / "calibration.json").write_text(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
