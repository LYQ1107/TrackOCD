"""V0-style non-parametric baseline on the Q1 strict stream.

Raw DINOv2 per-frame features + per-track causal EMA (or single frame) +
train-known prototypes + B2 online memory. Threshold 0.45 (frozen V0 rule)
or calibrated by the official proxy rule. No training.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import hungarian_acc
from src.ocd_v2.common import proxy_split


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def build_prototypes(mean_feats, labels):
    ids = np.unique(labels)
    out = []
    for c in ids:
        m = mean_feats[labels == c].mean(axis=0)
        out.append(l2norm(m))
    return ids, np.stack(out)


def calibrate(mean_feats, labels, sample_ids, seed=1027):
    label_map = {sid: int(c) for sid, c in zip(sample_ids, labels)}
    pk, pn = proxy_split(label_map, seed=seed)
    pk_ids = sorted(sid for sid, c in label_map.items() if c in pk)
    pn_ids = sorted(sid for sid, c in label_map.items() if c in pn)
    z = l2norm(mean_feats)
    protos = {}
    for c in pk:
        m = z[[i for i, s in enumerate(sample_ids) if label_map[s] == c]].mean(axis=0)
        protos[c] = l2norm(m)
    y = np.array([label_map[s] for s in pn_ids])
    order = {s: i for i, s in enumerate(sample_ids.tolist())}
    best = (0.45, -1.0)
    for thr in [0.45 + 0.05 * i for i in range(11)]:
        novel, counts, nid, preds = {}, {}, 100000, []
        for sid in pn_ids:
            x = z[order[sid]]
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
                novel[best_n] = (novel[best_n] * counts[best_n] + x) / (counts[best_n] + 1)
                novel[best_n] = l2norm(novel[best_n])
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
        acc = float(hungarian_acc(y, np.array([remap[int(v)] for v in pv]))[0])
        if acc > best[1]:
            best = (thr, acc)
    return best[0], best[1]


def replay(rows, feats, known_ids, protos, tau, ema_alpha=0.30, use_ema=True):
    z = l2norm(feats)
    chrono = sorted(
        rows,
        key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                       int(r.get("proposal_local_id") or 0), int(r["track_id"])),
    )
    row_index = {id(r): i for i, r in enumerate(rows)}
    state, novel, counts = {}, {}, {}
    next_id = 100000
    sem_action = [""] * len(rows)
    sem_sid = [""] * len(rows)
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        if use_ema:
            if key not in state:
                state[key] = z[i].copy()
            else:
                state[key] = (1 - ema_alpha) * state[key] + ema_alpha * z[i]
                state[key] = l2norm(state[key])
            h = state[key]
        else:
            h = z[i]
        ksims = protos @ h
        ki = int(np.argmax(ksims))
        if ksims[ki] >= tau:
            sem_action[i] = "known"
            sem_sid[i] = str(int(known_ids[ki]))
            continue
        if novel:
            best_sid, best_s = None, -1.0
            for sid, c in novel.items():
                s = float(np.dot(c, h))
                if s > best_s:
                    best_s, best_sid = s, sid
            if best_s >= tau:
                novel[best_sid] = ((novel[best_sid] * counts[best_sid] + h)
                                   / (counts[best_sid] + 1))
                novel[best_sid] = l2norm(novel[best_sid])
                counts[best_sid] += 1
                sem_action[i] = "existing"
                sem_sid[i] = str(best_sid)
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
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ema-alpha", type=float, default=0.30)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    args = ap.parse_args()

    k = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    known_ids, protos = build_prototypes(k["mean_feats"], k["labels"])
    tau = args.tau
    cal = None
    if args.calibrate:
        tau, acc = calibrate(k["mean_feats"], k["labels"], k["sample_ids"])
        cal = {"tau": tau, "proxy_novel_acc": acc}
        print("calibrated tau", tau, "proxy acc", acc)
    with open(args.proposals) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        df = list(reader)
    feats = np.load(ROOT / args.feats)["feats"]
    assert len(df) == len(feats)
    rows = [dict(r) for r in df]
    sem_action, sem_sid = replay(rows, feats, known_ids, protos, tau,
                                 args.ema_alpha, args.use_ema)
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
