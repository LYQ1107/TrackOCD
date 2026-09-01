"""Routing mechanism audit on the full per-step Q1 dev dump.

Produces:
  - single-frame vs trajectory-prefix AUROC (known vs novel, aligned tracks)
  - prefix-age evidence table
  - complete-track diagnostic AUROC
  - routing error taxonomy (STABLE/UNSTABLE_FALSE_KNOWN, etc.)
  - representative per-track evidence curves (PNG)

Q1 dev GT is used only for this descriptive audit; no parameter selection for
the Phase4Z router is made from these numbers (the router is selected on
category-disjoint TRAIN meta-dev).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.iclr27_phase4u.data import ROOT


def load_dump(path: Path):
    records = json.loads((path / "records.json").read_text())
    tracks = json.loads((path / "tracks.json").read_text())
    states = np.load(path / "states.npz")["states"]
    assert len(states) == len(records)
    return records, tracks, states


def per_track(records):
    out = defaultdict(list)
    for rec in records:
        out[tuple(rec["key"])].append(rec)
    return out


def step_scalars(rec, states, i):
    kl = np.asarray(rec["kl"], dtype=np.float32)
    h = states[i]
    return {
        "top1_p": rec["top1_p"],
        "margin": rec["margin"],
        "entropy": rec["entropy"],
        "energy": rec["energy"],
        "max_kl": float(kl.max()),
        "l1_known_p": rec["l1_probs"][0],
        "l1_novel_p": rec["l1_probs"][1],
        "q0": rec["q"][0],
        "r": rec["r"],
        "top1_idx": int(kl.argmax()),
    }


def prefix_features(steps, states):
    """Trajectory-prefix aggregate features up to each step."""
    n = len(steps)
    feats = []
    seen = set()
    switch = 0
    prev = None
    for i, (rec, st) in enumerate(zip(steps, states)):
        sc = step_scalars(rec, st, i)
        seen.add(sc["top1_idx"])
        if prev is not None and sc["top1_idx"] != prev:
            switch += 1
        prev = sc["top1_idx"]
        arr = np.array([sc["top1_p"], sc["margin"], sc["entropy"],
                        sc["energy"], sc["max_kl"], sc["l1_known_p"],
                        sc["l1_novel_p"], sc["q0"], sc["r"]], dtype=np.float32)
        agg = {
            "mean_top1_p": float(arr[0] if i == 0 else np.mean([step_scalars(r, s, j)["top1_p"]
                                                                for j, (r, s) in
                                                                enumerate(zip(steps[:i + 1], states[:i + 1]))])),
            "mean_entropy": float(np.mean([step_scalars(r, s, j)["entropy"]
                                           for j, (r, s) in
                                           enumerate(zip(steps[:i + 1], states[:i + 1]))])),
            "mean_energy": float(np.mean([step_scalars(r, s, j)["energy"]
                                          for j, (r, s) in
                                          enumerate(zip(steps[:i + 1], states[:i + 1]))])),
            "mean_margin": float(np.mean([step_scalars(r, s, j)["margin"]
                                          for j, (r, s) in
                                          enumerate(zip(steps[:i + 1], states[:i + 1]))])),
            "max_top1_p": float(max(step_scalars(r, s, j)["top1_p"]
                                    for j, (r, s) in
                                    enumerate(zip(steps[:i + 1], states[:i + 1])))),
            "min_entropy": float(min(step_scalars(r, s, j)["entropy"]
                                     for j, (r, s) in
                                     enumerate(zip(steps[:i + 1], states[:i + 1])))),
            "switch_count": switch,
            "n_unique": len(seen),
            "consistency": float(max((steps[:i + 1][j]["top1_p"] > 0) for j in range(i + 1))
                                 if False else np.mean([step_scalars(r, s, j)["top1_idx"] == sc["top1_idx"]
                                                        for j, (r, s) in
                                                        enumerate(zip(steps[:i + 1], states[:i + 1]))])),
            "last_top1_p": sc["top1_p"],
            "last_entropy": sc["entropy"],
            "last_energy": sc["energy"],
            "last_l1_known_p": sc["l1_known_p"],
            "last_l1_novel_p": sc["l1_novel_p"],
            "last_q0": sc["q0"],
            "last_r": sc["r"],
            "state_cos_first": float(st @ states[0] / (np.linalg.norm(st) * np.linalg.norm(states[0]) + 1e-12)),
        }
        feats.append(agg)
    return feats


def auroc(y, scores):
    if len(np.unique(y)) < 2 or len(scores) < 2:
        return None
    return float(roc_auc_score(y, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="outputs/iclr27_phase4z/routing_audit/dev_dump_full")
    ap.add_argument("--out", default="outputs/iclr27_phase4z/routing_audit/audit")
    args = ap.parse_args()

    records, tracks, states = load_dump(ROOT / args.dump)
    by_key = per_track(records)
    aligned = {k: v for k, v in by_key.items() if v[0].get("protocol_role") is not None}
    print("aligned tracks:", len(aligned))

    # per-track step lists with states
    track_data = {}
    idx = 0
    for k, recs in by_key.items():
        n = len(recs)
        track_data[k] = {
            "recs": recs,
            "states": states[idx:idx + n],
            "role": recs[0].get("protocol_role"),
            "gt_cat": recs[0].get("gt_cat"),
            "sid": recs[0].get("sid"),
            "n_steps": n,
        }
        idx += n

    # ---- single-frame AUROC by prefix age ----
    ages = [1, 2, 3, 4, 5, 8, 12, 20, 30]
    sf_rows = []
    for a in ages:
        ys, xs = [], defaultdict(list)
        for k, td in track_data.items():
            if td["role"] is None or len(td["recs"]) < a:
                continue
            r = td["recs"][a - 1]
            sc = step_scalars(r, td["states"], a - 1)
            y = 1 if td["role"] in ("supported_known", "zero_shot_known") else 0
            ys.append(y)
            for name, v in sc.items():
                if name != "top1_idx":
                    xs[name].append(v)
        row = {"age": a, "n": len(ys)}
        for name, vals in xs.items():
            row[name] = auroc(ys, vals)
        sf_rows.append(row)

    # ---- trajectory-prefix AUROC ----
    pref_rows = []
    for a in ages:
        ys = []
        feats = defaultdict(list)
        for k, td in track_data.items():
            if td["role"] is None or len(td["recs"]) < a:
                continue
            pf = prefix_features(td["recs"][:a], td["states"][:a])[-1]
            y = 1 if td["role"] in ("supported_known", "zero_shot_known") else 0
            ys.append(y)
            for name, v in pf.items():
                feats[name].append(v)
        row = {"age": a, "n": len(ys)}
        for name, vals in feats.items():
            row[name] = auroc(ys, vals)
        pref_rows.append(row)

    # ---- complete-track diagnostic ----
    y_full, sf_full, pf_full = [], defaultdict(list), defaultdict(list)
    for k, td in track_data.items():
        if td["role"] is None:
            continue
        y = 1 if td["role"] in ("supported_known", "zero_shot_known") else 0
        y_full.append(y)
        last = step_scalars(td["recs"][-1], td["states"], len(td["recs"]) - 1)
        for name, v in last.items():
            if name != "top1_idx":
                sf_full[name].append(v)
        pf = prefix_features(td["recs"], td["states"])[-1]
        for name, v in pf.items():
            pf_full[name].append(v)
    complete = {"n": len(y_full),
                "single_frame": {k: auroc(y_full, v) for k, v in sf_full.items()},
                "trajectory": {k: auroc(y_full, v) for k, v in pf_full.items()}}

    # ---- routing error taxonomy (model's own first decision) ----
    tax = defaultdict(int)
    novel_steps = defaultdict(list)
    known_steps = defaultdict(list)
    for k, td in track_data.items():
        if td["role"] is None:
            continue
        ts = tracks["_".join(map(str, k))]
        first = ts.get("first")
        recs, sts = td["recs"], td["states"]
        if first is None:
            tax["UNRESOLVED_" + ("KNOWN" if td["role"].startswith(("supported", "zero")) else "NOVEL")] += 1
            continue
        ft = first["t"]
        fstep = next((i for i, r in enumerate(recs) if r["t"] == ft), len(recs) - 1)
        top1s = [step_scalars(r, sts, i)["top1_idx"] for i, (r, st) in enumerate(zip(recs, sts))]
        p_max = max(step_scalars(r, sts, i)["top1_p"] for i, (r, st) in enumerate(zip(recs, sts)))
        switches = sum(1 for i in range(1, len(top1s)) if top1s[i] != top1s[i - 1])
        if td["role"] in ("supported_known", "zero_shot_known"):
            if first["l1"] == 0:
                tax["KNOWN_OK"] += 1
                known_steps["ok"].append((k, recs, sts))
            elif first["l1"] == 1:
                tax["KNOWN_TO_NOVEL"] += 1
                known_steps["to_novel"].append((k, recs, sts))
            else:
                tax["UNRESOLVED_KNOWN"] += 1
        else:
            if first["l1"] == 1:
                tax["NOVEL_OK"] += 1
                novel_steps["ok"].append((k, recs, sts))
            elif first["l1"] == 0:
                if switches == 0:
                    tag = "STABLE_FALSE_KNOWN"
                else:
                    tag = "UNSTABLE_FALSE_KNOWN"
                if p_max >= 0.8:
                    tag += "_HIGH_CONF"
                elif p_max < 0.5:
                    tag += "_LOW_CONF"
                tax[tag] += 1
                novel_steps["absorbed"].append((k, recs, sts))
            else:
                tax["UNRESOLVED_NOVEL"] += 1

    # ---- representative curves ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def draw(ax, recs, sts, title, color):
        xs = [r["t"] + 1 for r in recs]
        ax.plot(xs, [r["top1_p"] for r in recs], color=color, label="top1_p")
        ax.plot(xs, [r["l1_probs"][1] for r in recs], color=color, ls="--", label="l1 novel p")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("prefix age")
        ax.set_ylim(0, 1.05)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (known_steps["ok"][:1], "correct known", axes[0][0], "tab:green"),
        (novel_steps["ok"][:1], "correct novel", axes[0][1], "tab:orange"),
        (novel_steps["absorbed"][:1], "absorbed novel", axes[1][0], "tab:red"),
        (known_steps["to_novel"][:1], "known -> novel", axes[1][1], "tab:purple"),
    ]
    for samples, title, ax, color in panels:
        if samples:
            k, recs, sts = samples[0]
            draw(ax, recs, sts, f"{title} ({k[0]},{k[1]})", color)
        else:
            ax.set_title(f"{title} (none)", fontsize=9)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "representative_curves.png", dpi=140)
    plt.close(fig)

    summary = {
        "n_aligned": len(aligned),
        "n_known": sum(1 for td in track_data.values() if td["role"] in ("supported_known", "zero_shot_known")),
        "n_novel": sum(1 for td in track_data.values() if td["role"] == "novel"),
        "single_frame_auroc": sf_rows,
        "prefix_auroc": pref_rows,
        "complete_track": complete,
        "error_taxonomy": dict(tax),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
