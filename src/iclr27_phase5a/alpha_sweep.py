"""Quick EMA-alpha sensitivity for the trajectory online variant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase5a.pilot_gates import load_episodes, run_split, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/iclr27_phase5a/pilot/episodes")
    ap.add_argument("--out", default="outputs/iclr27_phase5a/pilot/gates")
    ap.add_argument("--taus", default="0.60,0.65,0.70,0.75,0.80,0.85")
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.5")
    ap.add_argument("--update-thresholds", default="none,0.80,0.85")
    args = ap.parse_args()

    data_dir = ROOT / args.data
    train = load_episodes(data_dir / "train.npz")
    meta = load_episodes(data_dir / "metadev.npz")
    p = np.load(data_dir / "protos.npz")
    protos = np.asarray(p["protos"], dtype=np.float32)
    known_list = [int(c) for c in p["known_list"]]
    taus = [float(x) for x in args.taus.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    thresholds = [None if x == "none" else float(x)
                  for x in args.update_thresholds.split(",")]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for alpha in alphas:
        for th in thresholds:
            key = f"alpha_{alpha}_th_{th}"
            summary[key] = {}
            for tau in taus:
                rec = run_split(meta, protos, known_list, tau, embed="h",
                                update_novel=True, ema_alpha=alpha,
                                update_threshold=th)
                s = summarize(rec)
                summary[key][str(tau)] = {"metadev": s}
                print(f"alpha={alpha} th={th} tau={tau} "
                      f"known={s['known_step_acc']:.3f} "
                      f"first={s['first_novel_birth_acc']:.3f} "
                      f"reuse={s['reuse_acc']:.3f} "
                      f"cross={s['cross_physical_reuse_acc']:.3f} "
                      f"switch={s['semantic_switch_rate']:.3f}")
            best_tau = max(taus, key=lambda t: (
                summary[key][str(t)]["metadev"]["known_step_acc"] +
                summary[key][str(t)]["metadev"]["first_novel_birth_acc"] +
                summary[key][str(t)]["metadev"]["reuse_acc"]) / 3)
            rec = run_split(train, protos, known_list, best_tau, embed="h",
                            update_novel=True, ema_alpha=alpha,
                            update_threshold=th)
            summary[key]["best_tau"] = best_tau
            summary[key]["train"] = summarize(rec)
            print(f"alpha={alpha} th={th} best_tau={best_tau} "
                  f"train_known={summary[key]['train']['known_step_acc']:.3f} "
                  f"train_reuse={summary[key]['train']['reuse_acc']:.3f}")
    (out_dir / "alpha_sweep.json").write_text(
        json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
