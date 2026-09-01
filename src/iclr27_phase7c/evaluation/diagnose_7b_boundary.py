"""Diagnose Phase 7B boundary errors on Q1 DEV (evaluation-only).

For the two failure regions:
  A. novel absorbed as KNOWN (cat 190/224/817-like),
  B. true known routed to EXISTING_NOVEL,
we replay a Phase 7B checkpoint with a diagnostic wrapper that records
per-row evidence (ksim, klogp, kmahal, kcover), the chosen slot, and the
slot's birth provenance (aligned novel / unaligned-FP / unknown). GT is used
only for labeling the regions, never for decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase7b.model.explainability import (
    EMAMemory,
    TOSEHead,
    TrackState,
    tose_step,
)
from src.iclr27_phase7a.training.train_reliability_head import load_tse, project
from src.iclr27_phase7b.evaluation.replay_tose import (
    load_stats,
    precompute,
    replay_track_ema,
)
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import group_tracks, load_proposals
from src.iclr27_phase7a.evaluation.strict_eval_any import load_gt_videos

DEV_VIDEOS = [88, 90, 122, 291, 334, 888, 931, 1159, 1232, 1276, 1572,
              1865, 2254, 2347, 2564, 2675, 2690, 2759, 2802, 2888]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)

    model, anchors, known_ids = load_tse(dev)
    stats = load_stats()
    with open(ROOT / "outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(r) for r in reader]
    feats = np.load(ROOT / "outputs/iclr27_phase6b/q1/final_dsct/feats.npz")[
        "feats"].astype(np.float32)
    z_all = project(dev, model, feats)
    h_all = replay_track_ema(rows, z_all)
    asims_all, loglik_all = precompute(h_all, anchors, stats)

    policy = TOSEHead(base_dim=14).to(dev)
    ck = torch.load(ROOT / args.head_ckpt, map_location=dev,
                    weights_only=False)
    policy.load_state_dict(ck["policy"])
    policy.eval()

    visible = np.ones(len(known_ids), dtype=bool)
    mem = EMAMemory(dim=128)
    track_stats = {}
    slot_birth = {}  # slot_idx -> (key, frame, ksim, klogp, validity)
    rows_by_idx = {}
    chrono = sorted(
        rows,
        key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                       int(r.get("proposal_local_id") or 0),
                       int(r["track_id"])))
    row_index = {id(r): i for i, r in enumerate(rows)}
    per_row = {}
    for r in chrono:
        i = row_index[id(r)]
        key = (int(r["video_id"]), int(r["track_id"]))
        ts = track_stats.get(key)
        if ts is None:
            ts = TrackState()
            track_stats[key] = ts
        res = tose_step(
            policy, z_all[i], mem, anchors, visible, known_ids, stats, ts,
            int(r["frame_id"]), key,
            asims_row=asims_all[i], loglik_row=loglik_all[i])
        if res["decision"] == 2 and res["slot_idx"] is not None:
            slot_birth[res["slot_idx"]] = {
                "key": list(key), "frame": int(r["frame_id"]),
                "ksim": float(res["ksim"]), "klogp": float(res["klogp"]),
                "kmahal": float(res["kmahal"]),
                "score": float(r["score"]),
                "role": r.get("gt_role"),
                "gt_cat": int(r["gt_category_id"])
                if r.get("gt_category_id") not in ("", None) else -1,
            }
        per_row[i] = {
            "key": list(key), "frame": int(r["frame_id"]),
            "action": ["known", "existing", "new"][res["decision"]],
            "sid": res["sid"], "slot_idx": res["slot_idx"],
            "ksim": float(res["ksim"]), "klogp": float(res["klogp"]),
            "kmahal": float(res["kmahal"]), "kscore": res["kscore"],
            "score": float(r["score"]), "role": r.get("gt_role"),
        }

    # align GT
    all_rows = load_proposals(ROOT / "outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv")
    tracks = group_tracks(all_rows)
    stream, labels = load_gt_videos(DEV_VIDEOS)
    mapping = align_pred_to_gt(tracks, gt_track_boxes(stream))

    cases_a = []  # novel -> known
    cases_b = []  # known -> existing
    for i, r in enumerate(rows):
        key = (int(r["video_id"]), int(r["track_id"]))
        if key not in mapping:
            continue
        sid = mapping[key]
        lab = labels[sid]
        role = lab["protocol_role"]
        if role not in ("supported_known", "zero_shot_known", "novel"):
            continue
        rec = per_row[i]
        if role == "novel" and rec["action"] == "known":
            cases_a.append({
                "cat": int(lab["ground_truth_category_id"]),
                "ksim": rec["ksim"], "klogp": rec["klogp"],
                "kmahal": rec["kmahal"], "kscore": rec["kscore"],
                "known_sid": rec["sid"], "frame": rec["frame"],
                "track": rec["key"][1], "score": rec["score"],
            })
        elif role in ("supported_known", "zero_shot_known") \
                and rec["action"] == "existing":
            si = rec["slot_idx"]
            birth = slot_birth.get(si, {})
            cases_b.append({
                "cat": int(lab["ground_truth_category_id"]),
                "ksim": rec["ksim"], "klogp": rec["klogp"],
                "kmahal": rec["kmahal"], "kscore": rec["kscore"],
                "slot": rec["sid"], "slot_birth_key": birth.get("key"),
                "slot_birth_frame": birth.get("frame"),
                "slot_birth_role": birth.get("role"),
                "slot_birth_ksim": birth.get("ksim"),
                "slot_birth_score": birth.get("score"),
                "frame": rec["frame"], "track": rec["key"][1],
                "score": rec["score"],
            })
    out = {
        "n_novel_absorbed": len(cases_a),
        "n_known_to_existing": len(cases_b),
        "novel_absorbed": cases_a,
        "known_to_existing": cases_b,
        "slot_births": [
            {"idx": k, **v} for k, v in sorted(slot_birth.items())],
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print("novel_absorbed", len(cases_a), "known_to_existing", len(cases_b))
    # summarize
    by_cat_a = defaultdict(list)
    for c in cases_a:
        by_cat_a[c["cat"]].append(c)
    for cat, cs in sorted(by_cat_a.items()):
        print(f"ABS cat {cat}: n={len(cs)} "
              f"ksim={np.mean([x['ksim'] for x in cs]):.3f} "
              f"klogp={np.mean([x['klogp'] for x in cs]):.3f} "
              f"known_sids={sorted(set(x['known_sid'] for x in cs))}")
    birth_roles = defaultdict(int)
    for c in cases_b:
        birth_roles[str(c["slot_birth_role"])] += 1
    print("known->existing slot birth roles:", dict(birth_roles))
    print("known->existing mean ksim %.3f klogp %.3f" % (
        np.mean([x["ksim"] for x in cases_b]),
        np.mean([x["klogp"] for x in cases_b])))


if __name__ == "__main__":
    main()
