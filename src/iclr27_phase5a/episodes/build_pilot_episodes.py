"""Phase 5A pilot episodes over the real Q1 TRAIN physical tracklet stream.

Each episode is a strict-causal deployment stream:
  - active pseudo-known categories (bank size varies: 4/6/12/24),
  - pseudo-novel categories (in-universe, excluded from the active bank),
  - real FP tracklets,
  - per frame: frozen d2_joint_v2 TSR state h_t (trajectory evidence) and
    the raw DINOv2 frame feature z_t (independent-frame evidence),
  - occurrence metadata (role, GT category, first-in-stream, physical key)
    is retained so cross-physical reuse can be measured.

No benchmark novel GT and no dev GT are used; only the 48 TRAIN supported
known categories are touched.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4t.episodes import RealStreamStore
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.downstream.model import (
    HierarchicalTSRCore,
    build_tsr_known_protos,
)
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)
from src.iclr27_phase4s.protocol import known_ids


def build_split(args, split: str, n_episodes: int, sizes: list[int],
                pool: list[int],
                store: RealStreamStore, rep: TSR, protos: torch.Tensor,
                known_list: list[int], device: str):
    cfg = WEpisodeConfig(max_len=args.max_len, fp_per_episode=args.fp_per_episode)
    rng = random.Random(args.seed)
    proto_index = {c: i for i, c in enumerate(known_list)}

    steps = []
    meta = []
    ep_known = []
    ep_novel = []
    with torch.no_grad():
        for e in range(n_episodes):
            nk = sizes[rng.randrange(len(sizes))]
            ep = make_episode(store, pool, cfg, rng, num_pseudo_known=nk)
            active_protos = torch.stack([protos[proto_index[c]] for c in ep["pseudo_known"]])
            ep_known.append(ep["pseudo_known"])
            ep_novel.append(ep["pseudo_novel"])
            for oi, occ in enumerate(ep["occurrences"]):
                z, q = store.tracklet_seq(occ["key"])
                n = min(len(z), args.max_len)
                if n < 1:
                    continue
                ft = torch.from_numpy(z[:n]).to(device)
                qt = torch.from_numpy(q[:n]).to(device)
                states = rep.embed_sequence(ft, qt).cpu().numpy().astype(np.float32)
                for t in range(n):
                    steps.append({
                        "h": states[t],
                        "f": z[t].astype(np.float32),
                    })
                    meta.append({
                        "ep": e, "occ": oi, "step": t,
                        "role": {"known": 0, "novel": 1, "fp": 2}[occ["role"]],
                        "cat": occ["category"],
                        "first": int(bool(occ.get("first"))),
                        "key0": int(occ["key"][0]), "key1": int(occ["key"][1]),
                        "n_occ": n,
                    })
            if (e + 1) % 50 == 0:
                print(f"[{split}] episode {e + 1}/{n_episodes}", flush=True)

    h = np.stack([s["h"] for s in steps]).astype(np.float32)
    f = np.stack([s["f"] for s in steps]).astype(np.float32)
    m = np.asarray(
        [[x["ep"], x["occ"], x["step"], x["role"], x["cat"], x["first"],
          x["key0"], x["key1"], x["n_occ"]] for x in meta],
        dtype=np.int64)
    return {
        "h": h, "f": f, "meta": m,
        "ep_known": np.asarray(ep_known, dtype=object),
        "ep_novel": np.asarray(ep_novel, dtype=object),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-metadev", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--known-set-sizes", default="4,6,12,24")
    ap.add_argument("--metadev-known-set-sizes", default="4,6")
    ap.add_argument("--fp-per-episode", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool_train = split["meta_train_categories"]
    pool_meta = split["meta_dev_categories"]
    known_list = sorted(known_ids())

    rep = TSR(arch="gru").to(args.device)
    ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                    map_location=args.device)
    tsr_sd = {k[len("rep."):]: v for k, v in ck["model"].items() if k.startswith("rep.")}
    rep.load_state_dict(tsr_sd)
    rep.eval()
    protos = build_tsr_known_protos(rep, args.device)
    store = load_store()

    train_sizes = [int(x) for x in args.known_set_sizes.split(",")]
    meta_sizes = [int(x) for x in args.metadev_known_set_sizes.split(",")]
    train = build_split(args, "train", args.n_train, train_sizes, pool_train,
                        store, rep, protos, known_list, args.device)
    meta = build_split(args, "metadev", args.n_metadev, meta_sizes, pool_meta,
                       store, rep, protos, known_list, args.device)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "train.npz",
        h=train["h"], f=train["f"], meta=train["meta"],
        ep_known=np.array([np.asarray(x, dtype=np.int64) for x in train["ep_known"]],
                          dtype=object),
        ep_novel=np.array([np.asarray(x, dtype=np.int64) for x in train["ep_novel"]],
                          dtype=object))
    np.savez_compressed(
        out / "metadev.npz",
        h=meta["h"], f=meta["f"], meta=meta["meta"],
        ep_known=np.array([np.asarray(x, dtype=np.int64) for x in meta["ep_known"]],
                          dtype=object),
        ep_novel=np.array([np.asarray(x, dtype=np.int64) for x in meta["ep_novel"]],
                          dtype=object))
    np.savez_compressed(
        out / "protos.npz",
        protos=protos.cpu().numpy().astype(np.float32),
        known_list=np.asarray(known_list, dtype=np.int64))
    # frame-space (DINOv2 768-d) category prototypes from the real store
    fproto_list = []
    for c in known_list:
        keys = store.by_cat.get(c, [])
        arrs = []
        for k in keys:
            z, _ = store.tracklet_seq(k)
            arrs.append(z)
        if not arrs:
            fproto_list.append(np.zeros(768, dtype=np.float32))
            continue
        m = np.concatenate(arrs, axis=0).mean(axis=0).astype(np.float32)
        fproto_list.append(m / (np.linalg.norm(m) + 1e-12))
    frame_protos = np.stack(fproto_list).astype(np.float32)
    np.savez_compressed(
        out / "frame_protos.npz",
        protos=frame_protos,
        known_list=np.asarray(known_list, dtype=np.int64))
    (out / "meta.json").write_text(json.dumps({
        "n_train_episodes": args.n_train,
        "n_metadev_episodes": args.n_metadev,
        "seed": args.seed,
        "max_len": args.max_len,
        "known_set_sizes": train_sizes,
        "metadev_known_set_sizes": meta_sizes,
        "fp_per_episode": args.fp_per_episode,
        "train_steps": int(train["h"].shape[0]),
        "metadev_steps": int(meta["h"].shape[0]),
        "train_pool": pool_train,
        "metadev_pool": pool_meta,
    }, indent=2))
    print(json.dumps({
        "train_steps": int(train["h"].shape[0]),
        "metadev_steps": int(meta["h"].shape[0]),
        "out": str(out),
    }))


if __name__ == "__main__":
    main()
