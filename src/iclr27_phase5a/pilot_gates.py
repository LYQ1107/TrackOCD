"""Phase 5A pilot gates on genuine-OOV meta-dev episodes.

Gate 1: immediate assign-or-create is learnable (reported accuracy at a
        legal validation threshold; also a full tau sweep).
Gate 2: online past-stream prototype update > frozen static clustering.
Gate 3: physical-trajectory TSR evidence > independent frame features.
Gate 4: cross-physical semantic reuse works.

All decisions are strict-causal: only occurrences at <= t on the current
physical track may influence the action at t, and past actions are never
rewritten.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase5a.assign_create.memory import CategoryMemory


def load_episodes(path: Path):
    d = np.load(path, allow_pickle=True)
    return d


def run_split(data, protos, known_list, tau, embed="h",
              update_novel=True, ema_alpha=0.5, update_threshold=None):
    """Replay every episode causally; returns per-step action records."""
    meta = data["meta"]
    x = data[embed]
    ep_known = data["ep_known"]
    proto_index = {int(c): i for i, c in enumerate(known_list)}
    records = []
    n_eps = max(int(meta[:, 0].max()) + 1, len(ep_known))
    mem = CategoryMemory(torch.from_numpy(protos), known_list,
                         ema_alpha=ema_alpha)
    for ep in range(n_eps):
        mem.reset()
        active = [int(c) for c in ep_known[ep]]
        mem.known_protos = torch.zeros(0, protos.shape[1])
        # rebuild active known anchors in the order of active ids
        mem.known_protos = torch.stack(
            [torch.from_numpy(protos[proto_index[c]]) for c in active])
        mem.known_ids = active
        rows = np.where(meta[:, 0] == ep)[0]
        occ_order = sorted(
            [(int(meta[i, 1]), int(meta[i, 2]), i) for i in rows])
        # stream is already stored in generation order; group by occ index
        occ_rows = defaultdict(list)
        for i in rows:
            occ_rows[int(meta[i, 1])].append(i)
        for oi in sorted(occ_rows):
            idxs = sorted(occ_rows[oi], key=lambda i: int(meta[i, 2]))
            key = (int(meta[idxs[0], 6]), int(meta[idxs[0], 7]))
            for i in idxs:
                h = torch.from_numpy(x[i])
                a, sid, sim = mem.step(h, tau, key, update_novel=update_novel,
                                       update_threshold=update_threshold)
                records.append({
                    "ep": ep, "occ": oi, "step": int(meta[i, 2]),
                    "role": int(meta[i, 3]), "cat": int(meta[i, 4]),
                    "first": int(meta[i, 5]), "key": key,
                    "n_occ": int(meta[i, 8]),
                    "action": a, "sid": sid, "sim": float(sim),
                })
    return records


def summarize(records):
    """Per-step + per-occurrence (first step) metrics."""
    known = [r for r in records if r["role"] == 0]
    novel = [r for r in records if r["role"] == 1]
    first_occ = [r for r in novel if r["first"] and r["step"] == 0]
    reuse = [r for r in novel if not r["first"]]
    reuse_first = [r for r in novel if not r["first"] and r["step"] == 0]

    # birth-slot map per episode (slot -> gt cat) from first occurrences
    slot_cat = {}
    for r in sorted(first_occ, key=lambda r: (r["ep"], r["occ"])):
        if r["action"] == "new":
            slot_cat[(r["ep"], r["sid"])] = r["cat"]

    def correct_slot(r):
        return slot_cat.get((r["ep"], r["sid"])) == r["cat"]

    known_correct = sum(1 for r in known if r["action"] == "known" and r["sid"] == r["cat"])
    first_new_ok = sum(1 for r in first_occ if r["action"] == "new")
    reuse_ok = sum(1 for r in reuse_first if r["action"] == "existing"
                   and correct_slot(r))
    # cross-physical reuse: reuse occurrence whose physical key differs from
    # the birth occurrence's key for that category
    birth_key = {}
    for r in sorted(first_occ, key=lambda r: (r["ep"], r["occ"])):
        birth_key.setdefault((r["ep"], r["cat"]), r["key"])
    cross = [r for r in reuse_first if birth_key.get((r["ep"], r["cat"])) != r["key"]]
    cross_ok = sum(1 for r in cross if r["action"] == "existing" and correct_slot(r))

    new_for_known = sum(1 for r in known if r["action"] == "new")
    new_for_reuse = sum(1 for r in reuse_first if r["action"] == "new")
    known_assigned_wrong = sum(1 for r in known if r["action"] == "known"
                               and r["sid"] != r["cat"])
    # semantic switch rate within physical tracks (adjacent steps, any action)
    switches = 0
    adj = 0
    by_track = defaultdict(list)
    for r in records:
        by_track[(r["ep"], r["key"])].append(r)
    for tr in by_track.values():
        tr.sort(key=lambda r: r["step"])
        for a, b in zip(tr, tr[1:]):
            adj += 1
            if (a["action"], a["sid"]) != (b["action"], b["sid"]):
                switches += 1

    return {
        "n_steps": len(records),
        "n_known_steps": len(known),
        "n_novel_steps": len(novel),
        "n_first_occ": len(first_occ),
        "n_reuse_occ": len(reuse_first),
        "n_cross_occ": len(cross),
        "known_step_acc": known_correct / max(len(known), 1),
        "first_novel_birth_acc": first_new_ok / max(len(first_occ), 1),
        "reuse_acc": reuse_ok / max(len(reuse_first), 1),
        "cross_physical_reuse_acc": cross_ok / max(len(cross), 1),
        "cross_physical_reuse_share": len(cross) / max(len(reuse_first), 1),
        "known_to_new_rate": new_for_known / max(len(known), 1),
        "reuse_to_new_rate": new_for_reuse / max(len(reuse_first), 1),
        "known_wrong_assign_rate": known_assigned_wrong / max(len(known), 1),
        "semantic_switch_rate": switches / max(adj, 1),
        "n_novel_births": sum(1 for r in first_occ if r["action"] == "new"),
        "n_novel_gt_cats": len({(r["ep"], r["cat"]) for r in first_occ}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/iclr27_phase5a/pilot/episodes")
    ap.add_argument("--out", default="outputs/iclr27_phase5a/pilot/gates")
    ap.add_argument("--tau-grid", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--ema-alpha", type=float, default=0.5)
    args = ap.parse_args()

    data_dir = ROOT / args.data
    train = load_episodes(data_dir / "train.npz")
    meta = load_episodes(data_dir / "metadev.npz")
    p = np.load(data_dir / "protos.npz")
    protos = np.asarray(p["protos"], dtype=np.float32)
    fp = np.load(data_dir / "frame_protos.npz")
    frame_protos = np.asarray(fp["protos"], dtype=np.float32)
    known_list = [int(c) for c in p["known_list"]]
    taus = [float(x) for x in args.tau_grid.split(",")]

    variants = [
        ("traj_online", "h", True),
        ("traj_static", "h", False),
        ("frame_online", "f", True),
        ("frame_static", "f", False),
    ]
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"taus": taus, "variants": {}}
    for name, embed, upd in variants:
        proto_bank = protos if embed == "h" else frame_protos
        rows = []
        for tau in taus:
            rec = run_split(meta, proto_bank, known_list, tau, embed=embed,
                            update_novel=upd, ema_alpha=args.ema_alpha)
            s = summarize(rec)
            rows.append({"tau": tau, **s})
        # choose tau maximizing balanced first+reuse+known on meta-dev
        best = max(rows, key=lambda r: (r["first_novel_birth_acc"] +
                                        r["reuse_acc"] + r["known_step_acc"]) / 3)
        summary["variants"][name] = {
            "embed": embed, "update_novel": upd,
            "selected_tau": best["tau"], "selected": best,
            "sweep": rows,
        }
        print(f"[{name}] best tau={best['tau']:.2f} "
              f"known={best['known_step_acc']:.3f} "
              f"first={best['first_novel_birth_acc']:.3f} "
              f"reuse={best['reuse_acc']:.3f} "
              f"cross={best['cross_physical_reuse_acc']:.3f} "
              f"switch={best['semantic_switch_rate']:.3f}")

    # train-side replay at the same mechanism (threshold re-selected on meta-dev)
    for name, embed, upd in variants:
        proto_bank = protos if embed == "h" else frame_protos
        tau = summary["variants"][name]["selected_tau"]
        rec = run_split(train, proto_bank, known_list, tau, embed=embed,
                        update_novel=upd, ema_alpha=args.ema_alpha)
        summary["variants"][name]["train_at_metadev_tau"] = summarize(rec)

    (out_dir / "gates.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote", out_dir / "gates.json")


if __name__ == "__main__":
    main()
