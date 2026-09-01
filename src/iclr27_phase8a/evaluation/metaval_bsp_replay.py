"""Strict-causal meta-val replay for the Bayesian semantic state process.

Answers the three Phase 8A gate questions on the legal class-held-out
meta-val:
  A. existing known-state assignment works;
  B. a hidden category's first physical track spawns a NEW state;
  C. a later different physical track of the same hidden category assigns to
     the same online-born state (cross-track reuse).

Only supported-known TRAIN statistics initialize known states (train_visible
classes). Hidden meta-val classes are never initialized. The replay is fully
chronological; every row is decided immediately from <= t evidence.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def load_meta(mode: str):
    ep = {k: np.asarray(v) for k, v in np.load(
        ROOT / f"outputs/iclr27_phase7c/assets/metaval_{mode}.npz").items()}
    split = json.loads((ROOT /
        f"outputs/iclr27_phase7c/assets/class_split_{mode}.json").read_text())
    return ep, split


def init_known_states(states, visible_ids):
    st = dict(np.load(ROOT / "outputs/iclr27_phase7b/assets/known_stats.npz"))
    mask = np.isin(st["known_ids"], np.asarray(sorted(visible_ids), dtype=np.int64))
    states.init_known(
        st["known_ids"][mask], st["mu"][mask], st["counts"][mask].astype(float))
    return st["known_ids"][mask].tolist()


def replay(ep, z_all, visible_ids, rho, sigma2, fp_keep, seed=2717):
    states = SemanticStateSet(dim=128, sigma2=sigma2, rho=rho)
    known_ids_used = init_known_states(states, visible_ids)
    slot_class = {}
    slot_birth_track = {}
    track_state = {}
    class_first_track = {}
    class_slot = {}
    n = {"known": 0, "first": 0, "same_attach": 0, "cross_attach": 0}
    ok = {"known": 0, "first": 0, "same_attach": 0, "cross_attach": 0}
    n_new = 0
    for i in range(len(z_all)):
        rs = int(ep["row_split"][i])
        if rs < 0 and (i * 2654435761 + seed) % 1000 / 10.0 >= fp_keep * 100:
            continue
        cat = int(ep["gt_category_id"][i])
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        tr = track_state.get(key)
        if tr is None:
            tr = TrajectoryState(dim=128)
            track_state[key] = tr
        action, sid, slot, scores = bsp_step(
            z_all[i], tr, states, known_ids_used, key,
            rho=rho, sigma2=sigma2)
        if action == 2 and slot is not None:
            n_new += 1
            if rs == 1:
                slot_class.setdefault(slot, cat)
                slot_birth_track.setdefault(slot, key)
                class_slot.setdefault(cat, slot)
        if rs == 0:
            n["known"] += 1
            ok["known"] += (action == 0 and sid == cat)
        elif rs == 1:
            if cat not in class_first_track:
                class_first_track[cat] = key
                n["first"] += 1
                ok["first"] += (action == 2)
            else:
                # first occurrence already seen: every later row is an
                # attach (same or different physical track).
                target_slot = class_slot.get(cat)
                correct = (
                    action == 1 and target_slot is not None
                    and slot == target_slot and slot_class.get(slot) == cat)
                if target_slot is not None and key == slot_birth_track.get(
                        target_slot):
                    n["same_attach"] += 1
                    ok["same_attach"] += correct
                else:
                    n["cross_attach"] += 1
                    ok["cross_attach"] += correct
    return {
        "rho": rho,
        "sigma2": sigma2,
        "n_known": n["known"],
        "known_acc": ok["known"] / max(n["known"], 1),
        "n_first": n["first"],
        "first_acc": ok["first"] / max(n["first"], 1),
        "n_same_attach": n["same_attach"],
        "same_attach_acc": ok["same_attach"] / max(n["same_attach"], 1),
        "n_cross_attach": n["cross_attach"],
        "cross_attach_acc": ok["cross_attach"] / max(n["cross_attach"], 1),
        "n_states": states.n,
        "n_new_born": n_new,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hard", "random"], default="hard")
    ap.add_argument("--fp-keep", type=float, default=0.25)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--sigma2", type=float, default=0.001)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device(args.device)
    ep, split = load_meta(args.mode)
    model, _, _ = load_tse(dev)
    z_all = project(dev, model, ep["feats"].astype(np.float32))
    visible = sorted(set(split["train_visible"]))
    rhos = [args.rho] if args.rho is not None else [-70, -60, -50, -40,
                                                    -30, -20, -10, 0, 10,
                                                    20, 30, 40, 50, 60, 70]
    results = []
    for rho in rhos:
        res = replay(ep, z_all, visible, rho, args.sigma2, args.fp_keep)
        results.append(res)
        print(json.dumps(res))
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
