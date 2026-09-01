"""CHP meta-development evaluation.

Primary proxy: real held-out class proxy.  The 10 frozen meta-dev classes
are treated as truly unseen pseudo-novel classes; their prototypes are
EXCLUDED from the known prototype space (this fixes the Phase 4D long-stream
degeneracy where meta-dev tracks matched their own class prototype).

Secondary proxies (reported, never used alone for selection):
  - mixed long-stream proxy (Phase 4D cache, known space = 48 classes);
  - synthetic scale stress (synthetic portion of the long stream).

No official validation data is used for any threshold or tier boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit_msr.evaluate import attach_gt, summarize
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)
from src.orbit_mdc.evaluate_mdc import (
    evaluate_long_mdc,
    load_mdc_model,
    run_mdc_stream,
)
from src.orbit_chp.episodes import hardness_of_classes, build_protos
from src.orbit_chp.split import load_chp_split


def build_real_heldout_stream(seed=2026, n_known_repeat=2):
    """Deterministic real held-out proxy stream (train-side classes only)."""
    split = load_chp_split()
    meta_train = split["episode_pool"]
    meta_dev = split["heldout"]
    all_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)
    known_rows = []
    for c in meta_train:
        for sid in by_class[c]:
            known_rows.append({"sample_id": sid, "class": c,
                               "role": "known"})
    novel_rows = []
    for c in meta_dev:
        for sid in by_class[c]:
            novel_rows.append({"sample_id": sid, "class": c,
                               "role": "novel"})
    rng = random.Random(seed)
    all_rows = []
    for rep in range(n_known_repeat):
        rep_rows = list(known_rows)
        rng.shuffle(rep_rows)
        for r in rep_rows:
            all_rows.append({"sample_id": f"{r['sample_id']}#{rep}",
                             "class": r["class"], "role": "known",
                             "orig": r["sample_id"]})
    all_rows += list(novel_rows)
    rng.shuffle(all_rows)
    rows = []
    gt_rows = []
    feats = {}
    seen = set()
    for i, r in enumerate(all_rows):
        sid = r["sample_id"]
        if r["role"] == "known":
            orig = r["orig"]
            feats[sid] = all_feats[orig][:8]
            gt_rows.append({"sample_id": sid,
                            "ground_truth_category_id": r["class"],
                            "protocol_role": "supported_known"})
        else:
            feats[sid] = all_feats[sid][:8]
            gt_rows.append({"sample_id": sid,
                            "ground_truth_category_id": r["class"],
                            "protocol_role": "novel"})
        first = r["class"] not in seen
        seen.add(r["class"])
        rows.append({"sample_id": sid, "stream_order": i, "role": r["role"],
                     "class": r["class"], "first_occurrence": first})
    return rows, gt_rows, feats


def pool_hardness_distribution():
    """Train-side leave-one-out hardness of the 38-class episode pool."""
    split = load_chp_split()
    pool = split["episode_pool"]
    all_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in all_feats:
            by_class[int(c)].append(sid)
    # Raw DINO mean per track is a legal frozen-evidence hardness;
    # adapted features would require the model, which is unavailable here.
    z_cache = {sid: np.mean(f, axis=0).astype(np.float32)
               for sid, f in all_feats.items()}
    h = {}
    for c in pool:
        others = [o for o in pool if o != c]
        h[c] = hardness_of_classes(z_cache, by_class, [c], others)[c]
    return np.array(sorted(h.values()))


def tier_of(track_best_known, boundaries):
    if track_best_known < boundaries[0]:
        return "easy"
    if track_best_known < boundaries[1]:
        return "medium"
    return "hard"


def evaluate_checkpoint(model, ck, device, gate_thr=0.5, compat_thr=0.45,
                        compat_margin=0.05, out_prefix="chp"):
    split = load_chp_split()
    meta_train = split["episode_pool"]
    labels = load_train_labels()
    all_feats = load_frame_features("train_known_mean")
    labels38 = {sid: c for sid, c in labels.items()
                if int(c) in meta_train and sid in all_feats}
    train_feats38 = {sid: all_feats[sid][:8] for sid in labels38}
    rows, gt_rows, feats = build_real_heldout_stream()
    logs = run_mdc_stream(model, ck, rows, feats, labels38, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          compat_margin=compat_margin,
                          proto_feats=train_feats38)
    attach_gt(logs, gt_rows)
    rows_out = []
    r = summarize("real_heldout", logs, gt_rows, "overall")
    if r:
        rows_out.append(r)
    r = summarize("real_heldout", logs, gt_rows, "real_only",
                  select=lambda l: l["true_role"] == "novel")
    if r:
        rows_out.append(r)
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        r = summarize("real_heldout", logs, gt_rows, bucket,
                      select=lambda l, b=bucket:
                      active_bucket(l["active_novel_prototypes"]) == b)
        if r:
            rows_out.append(r)
    boundaries = np.percentile(pool_hardness_distribution(), [33, 66])
    tier_rows = []
    for t in ["easy", "medium", "hard"]:
        ls = [l for l in logs if l["true_role"] == "novel"
              and tier_of(l["best_known_similarity"], boundaries) == t]
        if not ls:
            continue
        rr = summarize("real_heldout", ls, gt_rows, f"tier_{t}")
        if rr:
            rr["n"] = len(ls)
            tier_rows.append(rr)
    out_dir = ROOT / "outputs" / "orbit_chp" / "meta_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{out_prefix}_real_heldout.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    with open(out_dir / f"{out_prefix}_hardness_tier_results.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tier_rows[0].keys()))
        w.writeheader()
        w.writerows(tier_rows)
    with open(out_dir / f"{out_prefix}_tier_boundaries.json", "w") as f:
        json.dump({"boundaries": boundaries.tolist()}, f)
    return rows_out, tier_rows, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--gpu", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"
    names = args.names or [Path(p).parent.name for p in args.checkpoints]
    combined = []
    for name, path in zip(names, args.checkpoints):
        model, ck = load_mdc_model(path, device)
        rows, tiers, logs = evaluate_checkpoint(
            model, ck, device, args.gate_threshold, args.compat_threshold,
            args.compat_margin, out_prefix=name)
        for r in rows:
            r["checkpoint"] = name
        combined.extend(rows)
        # secondary: mixed long-stream proxy
        ls_out, ls_logs, r_real, r_syn = evaluate_long_mdc(
            model, ck, device, gate_thr=args.gate_threshold,
            compat_thr=args.compat_threshold,
            compat_margin=args.compat_margin)
        for r in ls_out:
            r["checkpoint"] = name
        combined.extend(ls_out)
        print(name, "real:", rows[0] if rows else None,
              "long:", ls_out[0] if ls_out else None, flush=True)
    out_dir = ROOT / "outputs" / "orbit_chp" / "meta_dev"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        w.writeheader()
        w.writerows(combined)
    print("saved", out_dir / "model_comparison.csv")


if __name__ == "__main__":
    main()
