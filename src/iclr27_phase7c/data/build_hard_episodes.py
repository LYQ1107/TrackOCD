"""Build Phase 7C legal class-level meta-train/meta-val episodes (v2).

Source: corrected Phase 4T TRAIN stream (9,047 supported-known rows + 30,000
sampled FP rows). Only supported-known labels are used.

Split (per class, deterministic seed):
  - train_visible (20): KNOWN targets in train; anchors visible in meta-val;
  - train_hidden (14): proxy-OOD NEW/EXISTING targets in train (semantic-
    hard episodes; anchors masked in train chunks);
  - meta_hidden (13): completely HELD OUT of training (rows excluded from
    the train episodes); anchors masked at meta-val. This is the clean
    held-out proxy for true OOV and avoids the 7B/7C-v1 failure where the
    model memorized meta-val hidden classes because they were trained as
    known.

Hardness is measured by the per-row track-EMA max-ksim to the visible
anchor set (legal, supported-known only). A random control split is also
produced for ablations.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_all_rows():
    tr = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/train_episodes.npz").items()}
    va = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/val_episodes.npz").items()}
    ep = {}
    for k in tr:
        ep[k] = np.concatenate([tr[k], va[k]], axis=0)
    return ep


def load_anchors():
    state = torch.load(
        ROOT / "outputs/iclr27_phase6c/training/tse_main/checkpoint.pth",
        map_location="cpu", weights_only=False)
    from src.iclr27_phase6c.model.tse import KnownAnchors
    an = KnownAnchors(state["known_ids"]).to("cpu")
    an.load_state_dict(state["anchors"])
    anchors = an.normalized().detach().numpy().astype(np.float32)
    return anchors, [int(x) for x in state["known_ids"]]


def class_stats(ep, h_all, anchors, known_ids):
    m = ep["gt_role"] == 1
    cats = ep["gt_category_id"][m].astype(np.int64)
    hs = h_all[m]
    sim = hs @ anchors.T
    kid = {c: i for i, c in enumerate(known_ids)}
    stats = {}
    for c in np.unique(cats):
        c = int(c)
        xs = sim[cats == c]
        stats[c] = {
            "n": int(len(xs)),
            "median_max": float(np.median(xs.max(1))),
            "mean_max": float(xs.max(1).mean()),
            "anchor_idx": kid.get(c, -1),
        }
    return stats


def make_split(ep, h_all, anchors, known_ids, seed, mode):
    stats = class_stats(ep, h_all, anchors, known_ids)
    classes = sorted(stats)
    rng = random.Random(seed)
    n_visible, n_hidden_train, n_hidden_val = 20, 14, 13
    best = None
    for trial in range(40):
        cand = set(rng.sample(classes, n_visible))
        vis_idx = [known_ids.index(c) for c in sorted(cand)]
        # per-class median max-ksim to visible anchors
        m = ep["gt_role"] == 1
        cats = ep["gt_category_id"][m].astype(np.int64)
        hs = h_all[m]
        sim = hs @ anchors[vis_idx].T
        scores = {}
        for c in classes:
            xs = sim[cats == c]
            scores[c] = float(np.median(xs.max(1)))
        # score: how many remaining classes are hard-ish (>=0.32) and have
        # moderate row counts (>=15) for a usable meta-val
        hardish = [c for c in classes if c not in cand
                   and scores[c] >= 0.32 and 15 <= stats[c]["n"] <= 900]
        score = (len(hardish), sum(stats[c]["n"] for c in hardish))
        if best is None or score > best[0]:
            best = (score, cand, dict(scores), stats)
    _, cand, scores, stats = best
    rest = [c for c in classes if c not in cand]
    rest.sort(key=lambda c: (-scores[c], stats[c]["n"]))
    usable = [c for c in rest if 15 <= stats[c]["n"] <= 900]
    if mode == "hard":
        # interleave the hardest pool so both train and meta-val hidden sets
        # contain hard classes (hard training episodes + hard meta-val)
        pool = (usable + [c for c in rest if c not in usable])[
            :n_hidden_val + n_hidden_train]
        hidden_val = pool[::2][:n_hidden_val]
        hidden_train = [c for c in pool if c not in hidden_val][
            :n_hidden_train]
    else:
        rng.shuffle(rest)
        hidden_val = rest[:n_hidden_val]
        hidden_train = rest[n_hidden_val:n_hidden_val + n_hidden_train]
    split = {
        "train_visible": sorted(cand),
        "hidden_train": sorted(hidden_train),
        "hidden_val": sorted(hidden_val),
        "all": classes,
        "hardness_scores": {str(c): round(scores[c], 3) for c in classes},
    }
    return split


def pack(ep, split, mode):
    vis = set(split["train_visible"])
    ht = set(split["hidden_train"])
    hv = set(split["hidden_val"])
    cats = ep["gt_category_id"]
    roles = ep["gt_role"]
    row_split = np.full(len(cats), -1, dtype=np.int8)
    for i, (r, c) in enumerate(zip(roles, cats)):
        if int(r) != 1:
            continue
        c = int(c)
        if mode == "train":
            if c in hv:
                row_split[i] = -2  # held out of training entirely
            elif c in vis:
                row_split[i] = 0
            elif c in ht:
                row_split[i] = 1
        else:  # meta-val
            if c in vis:
                row_split[i] = 0
            elif c in hv:
                row_split[i] = 1
            elif c in ht:
                row_split[i] = -1  # seen-proxy classes: unlabeled at meta-val
    out = dict(ep)
    out["row_split"] = row_split
    if mode == "train":
        keep = row_split != -2
        for k in out:
            out[k] = out[k][keep]
    return out


def main():
    from src.iclr27_phase7a.training.train_reliability_head import load_tse, project

    dev = torch.device("cuda:0")
    model, _, _ = load_tse(dev)
    ep = load_all_rows()
    z_all = project(dev, model, ep["feats"].astype(np.float32))
    anchors, known_ids = load_anchors()
    h_all = replay_track_ema_ep(ep, z_all)
    out = ROOT / "outputs/iclr27_phase7c/assets"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "h_all.npz", h=h_all.astype(np.float32))
    stats_all = {}
    for name, seed in (("hard", 2717), ("random", 2718)):
        split = make_split(ep, h_all, anchors, known_ids, seed, name)
        tr = pack(ep, split, "train")
        va = pack(ep, split, "val")
        np.savez_compressed(out / f"train_{name}.npz", **tr)
        np.savez_compressed(out / f"metaval_{name}.npz", **va)
        (out / f"class_split_{name}.json").write_text(
            json.dumps(split, indent=2))
        counts = defaultdict(int)
        for c in split["train_visible"]:
            counts["visible"] += 1
        for c in split["hidden_train"]:
            counts["hidden_train"] += 1
        for c in split["hidden_val"]:
            counts["hidden_val"] += 1
        stats_all[name] = {
            "train_rows": int((tr["row_split"] != -2).sum()),
            "train_known_rows": int((tr["row_split"] == 0).sum()),
            "train_hidden_rows": int((tr["row_split"] == 1).sum()),
            "metaval_known_rows": int((va["row_split"] == 0).sum()),
            "metaval_hidden_rows": int((va["row_split"] == 1).sum()),
            **counts,
        }
    (out / "episode_stats.json").write_text(json.dumps(stats_all, indent=2))
    print(json.dumps(stats_all, indent=2))


def replay_track_ema_ep(ep, z_all, alpha=0.30):
    ema = {}
    h_all = np.zeros_like(z_all)
    for i in range(len(z_all)):
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
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


if __name__ == "__main__":
    main()
