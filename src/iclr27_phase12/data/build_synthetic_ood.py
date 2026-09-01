"""Build legal support/query OOD episodes from public train-known tracks.

The hidden categories are selected from the existing Phase-7C class split. No
Q1 files, private labels, or Q1 proposal metadata are read. Categories used to
form support/query episodes are still public training labels, which is the
registered synthetic feasibility setting.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def atomic_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase12/synthetic")
    ap.add_argument("--seed", type=int, default=1212)
    ap.add_argument("--max-tracks-per-category", type=int, default=8)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tracks = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    labels = tracks["labels"].astype(np.int64)
    split = json.loads((ROOT / "outputs/iclr27_phase7c/assets/class_split_hard.json").read_text())
    counts = {int(c): int(np.sum(labels == int(c))) for c in split["all"]}
    rng = random.Random(args.seed)

    # Hidden-train categories provide supervised correspondence examples;
    # hidden-val categories are untouched during training and become OOD
    # support/query evaluation categories. Classes with one track cannot make
    # a correspondence pair and are excluded from the pair probe.
    train_cats = [int(c) for c in split["hidden_train"] if counts[int(c)] >= 2]
    eval_cats = [int(c) for c in split["hidden_val"] if counts[int(c)] >= 2]
    visible_cats = [int(c) for c in split["train_visible"] if counts[int(c)] >= 2]

    def choose(cats):
        result = []
        for c in cats:
            idx = np.flatnonzero(labels == c).tolist()
            rng.shuffle(idx)
            idx = idx[:args.max_tracks_per_category]
            n_support = max(1, len(idx) // 2)
            result.append((c, idx[:n_support], idx[n_support:]))
        return result

    train_groups = choose(train_cats)
    eval_groups = choose(eval_cats)
    visible_groups = choose(visible_cats)

    train_indices = [i for _, s, q in train_groups for i in (s + q)]
    eval_support = [i for _, s, _ in eval_groups for i in s]
    eval_query = [i for _, _, q in eval_groups for i in q]
    visible_indices = [i for _, s, q in visible_groups for i in (s + q)]
    if not eval_query:
        raise RuntimeError("no synthetic OOD query tracks")

    meta = {
        "seed": args.seed,
        "source": "outputs/iclr27_phase6c/assets/known_tracks.npz",
        "q1_labels_used": False,
        "private_gt_used": False,
        "physical_id_used_as_feature": False,
        "train_hidden_categories": [c for c, _, _ in train_groups],
        "eval_hidden_categories": [c for c, _, _ in eval_groups],
        "visible_categories": [c for c, _, _ in visible_groups],
        "train_tracks": len(train_indices),
        "eval_support_tracks": len(eval_support),
        "eval_query_tracks": len(eval_query),
        "train_groups": [{"category": c, "support": s, "query": q} for c, s, q in train_groups],
        "eval_groups": [{"category": c, "support": s, "query": q} for c, s, q in eval_groups],
        "visible_groups": [{"category": c, "tracks": s + q} for c, s, q in visible_groups],
    }
    atomic_npz(
        out / "episodes.npz",
        train_indices=np.asarray(train_indices, dtype=np.int64),
        eval_support=np.asarray(eval_support, dtype=np.int64),
        eval_query=np.asarray(eval_query, dtype=np.int64),
        visible_indices=np.asarray(visible_indices, dtype=np.int64),
        labels=labels,
    )
    (out / "episodes.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in (
        "train_hidden_categories", "eval_hidden_categories", "visible_categories",
        "train_tracks", "eval_support_tracks", "eval_query_tracks",
        "q1_labels_used", "private_gt_used")}, indent=2))


if __name__ == "__main__":
    main()
