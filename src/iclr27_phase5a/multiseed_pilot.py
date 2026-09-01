"""Multi-seed stability of the Phase 5A pilot gates (fixed hyperparameters)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase5a.pilot_gates import load_episodes, run_split, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/iclr27_phase5a/pilot/episodes")
    ap.add_argument("--seeds", default="base,20260817,20260818")
    ap.add_argument("--out", default="outputs/iclr27_phase5a/multiseed")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--ema-alpha", type=float, default=0.1)
    ap.add_argument("--update-threshold", type=float, default=0.85)
    args = ap.parse_args()

    base = ROOT / args.base
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for seed in args.seeds.split(","):
        data_dir = base if seed == "base" else base / f"seed_{seed}"
        meta = load_episodes(data_dir / "metadev.npz")
        p = np.load(data_dir / "protos.npz")
        protos = np.asarray(p["protos"], dtype=np.float32)
        fp = np.load(data_dir / "frame_protos.npz")
        frame_protos = np.asarray(fp["protos"], dtype=np.float32)
        known_list = [int(c) for c in p["known_list"]]
        summary[seed] = {}
        for name, embed, upd in (("traj_online", "h", True),
                                 ("traj_static", "h", False),
                                 ("frame_online", "f", True)):
            proto_bank = protos if embed == "h" else frame_protos
            rec = run_split(meta, proto_bank, known_list, args.tau, embed=embed,
                            update_novel=upd, ema_alpha=args.ema_alpha,
                            update_threshold=args.update_threshold)
            s = summarize(rec)
            summary[seed][name] = s
            print(f"seed={seed} {name} known={s['known_step_acc']:.3f} "
                  f"first={s['first_novel_birth_acc']:.3f} "
                  f"reuse={s['reuse_acc']:.3f} cross={s['cross_physical_reuse_acc']:.3f}")
    (out_dir / "multiseed.json").write_text(
        json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
