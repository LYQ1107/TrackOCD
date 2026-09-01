"""Calibrate rule-based RACC-v2 on the legal proxy-val split (no Q1/heldout).

Grid search over (nu, pen, tau_attach, rel_birth) with tau_k=0.65 fixed
from the Phase 6C legal calibration. Metrics: proxy known acc, novel first
acc, novel reuse acc, and global new rate on the val episodes.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase7a.model.reliability_memory import MemoryState, TrackStats
from src.iclr27_phase7a.model.reliability_memory_v2 import v2_step
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project


def replay_val(z, ep, anchors, known_ids, split, cfg):
    visible = set(split["known"])
    novel = set(split["novel_val"])
    vis = np.isin(known_ids, np.asarray(sorted(visible)))
    mem = MemoryState(dim=128)
    ts = {}
    slot_class = {}
    class_seen = set()
    n_known = n_first = n_attach = 0
    acc_known = acc_first = acc_attach = 0
    n_new = n_all = 0
    for ri in range(len(z)):
        role = int(ep["gt_role"][ri])
        cat = int(ep["gt_category_id"][ri])
        key = (int(ep["video_ids"][ri]), int(ep["track_ids"][ri]))
        st = ts.get(key)
        if st is None:
            st = TrackStats()
            ts[key] = st
        res = v2_step(
            z[ri], mem, anchors, vis, known_ids, st,
            float(ep["score"][ri]), int(ep["prior_hits"][ri]),
            ep["bbox_xyxy"][ri], int(ep["frame_ids"][ri]), key, cfg)
        n_all += 1
        n_new += (res["decision"] == 2)
        if role == 1 and cat in visible:
            n_known += 1
            acc_known += (res["decision"] == 0
                          and res["sid"] == cat)
        elif role == 1 and cat in novel:
            if cat not in class_seen:
                class_seen.add(cat)
                n_first += 1
                acc_first += (res["decision"] == 2)
            else:
                n_attach += 1
                acc_attach += (res["decision"] == 1
                               and slot_class.get(res["slot_idx"]) == cat)
            if res["decision"] == 2 and res["slot_idx"] is not None:
                slot_class.setdefault(res["slot_idx"], cat)
    return {
        "known_acc": acc_known / max(n_known, 1),
        "first_acc": acc_first / max(n_first, 1),
        "reuse_acc": acc_attach / max(n_attach, 1),
        "new_rate": n_new / max(n_all, 1),
        "n_first": n_first,
        "n_attach": n_attach,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="outputs/iclr27_phase7a/eval/v2_calibration.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    split = json.loads(
        (ROOT / "outputs/iclr27_phase7a/assets/class_split.json").read_text())
    ep = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/val_episodes.npz").items()}
    dev = torch.device(args.device)
    model, anchors, known_ids = load_tse(dev)
    z = project(dev, model, ep["feats"])
    results = []
    grid = list(itertools.product(
        [1.0, 2.0, 4.0], [0.0, 0.05, 0.10],
        [0.30, 0.40, 0.50], [0.20, 0.30, 0.40]))
    for nu, pen, tau_attach, rel_birth in grid:
        cfg = {"tau_k": 0.65, "nu": nu, "pen": pen,
               "tau_attach": tau_attach, "rel_birth": rel_birth}
        m = replay_val(z, ep, anchors, known_ids, split, cfg)
        score = m["reuse_acc"] * m["first_acc"] - 0.3 * m["new_rate"]
        results.append({"cfg": cfg, "score": score, **m})
    results.sort(key=lambda r: r["score"], reverse=True)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    for r in results[:10]:
        print(r["cfg"], "score=%.4f" % r["score"],
              "known=%.3f first=%.3f reuse=%.3f new=%.4f" % (
                  r["known_acc"], r["first_acc"], r["reuse_acc"],
                  r["new_rate"]))
    print("best cfg", results[0]["cfg"])


if __name__ == "__main__":
    main()
