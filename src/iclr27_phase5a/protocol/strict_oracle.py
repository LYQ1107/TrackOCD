"""Strict-causal per-occurrence oracle ceiling on Q1 dev.

Protocol: every proposal occurrence gets an immediate semantic action
{KNOWN(c), EXISTING_NOVEL(k), NEW_NOVEL} using only past occurrences.

Oracle actions (perfect category state):
  known  -> KNOWN(gt category)
  novel  -> EXISTING_NOVEL(k) if that gt category was already observed,
            else NEW_NOVEL (born now).

Frozen-known variant: known occurrences are classified by the frozen
Stage C / d2_joint_v2 known head (argmax over 48), novel occurrences keep the
perfect causal oracle. This is the strict-protocol analogue of O1c.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    known_ids,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.downstream.dev_eval import qphys_from_rows
from src.iclr27_phase4u.downstream.model import (
    HierarchicalTSRCore,
    build_tsr_known_protos,
)
from src.iclr27_phase4u.trajectory.model import TSR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    known_list = sorted(known_ids())
    rep = TSR(arch="gru").to(args.device)
    ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                    map_location=args.device)
    tsr_sd = {k[len("rep."):]: v for k, v in ck["model"].items() if k.startswith("rep.")}
    rep.load_state_dict(tsr_sd)
    rep.eval()
    protos = build_tsr_known_protos(rep, args.device)
    model = HierarchicalTSRCore(rep, protos, use_defer=False,
                                use_qphys=True).to(args.device)
    model.load_t3_init(str(ROOT / "outputs/iclr27_phase4t/t3/checkpoint.pth"),
                       args.device)
    ck2 = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                     map_location=args.device)
    ck2["model"].pop("known_raw", None)
    model.load_state_dict(ck2["model"], strict=False)
    model.eval()

    rows = load_proposals(Path(Q1_DEV))
    arr = np.load(ROOT / "outputs/iclr27_phase4s/q1_features/feats.npz")["feats"]
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    # precompute per-track TSR states in key order.
    # Each state is the causal TSR embedding after seeing exactly the rows
    # up to and including that occurrence (embed_sequence emits one state
    # per input frame). We index states by occurrence within the track.
    states_by_key = {}
    with torch.no_grad():
        for key in sorted(tracks):
            tr = [r for r in tracks[key]
                  if feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                       int(r["image_id"]))) is not None]
            if not tr:
                continue
            ft = np.stack([feats_by_key[(int(r["video_id"]), int(r["track_id"]),
                                         int(r["image_id"]))] for r in tr])
            qt = np.stack([qmap[id(r)] for r in tr]).astype(np.float32)
            states = rep.embed_sequence(
                torch.from_numpy(ft).to(args.device),
                torch.from_numpy(qt).to(args.device))
            states_by_key[key] = [s.unsqueeze(0) for s in states]

    # chronological occurrence stream
    chrono = sorted(rows, key=lambda r: (int(r["video_id"]), int(r["frame_id"]),
                                         int(r.get("proposal_local_id") or 0),
                                         int(r["track_id"])))
    ptr = defaultdict(int)
    novel_id = {}
    next_novel = 0
    records = []
    n_aligned = n_known = n_novel = n_first = n_reuse = 0
    frozen_known_correct = 0
    frozen_known_total = 0
    with torch.no_grad():
        for r in chrono:
            key = (int(r["video_id"]), int(r["track_id"]))
            sid = mapping.get(key)
            if sid is None:
                continue
            lab = labels[sid]
            role = lab["protocol_role"]
            gt_cat = int(lab["ground_truth_category_id"])
            hs = states_by_key.get(key)
            if hs is None:
                continue
            i = ptr[key]
            ptr[key] += 1
            if i >= len(hs):
                continue
            h = hs[i]
            kl = model.known_logits(h, list(range(len(known_list))))[0]
            p = torch.softmax(kl, dim=-1)
            pred_known = known_list[int(kl.argmax())]
            n_aligned += 1
            if role in ("supported_known", "zero_shot_known"):
                n_known += 1
                frozen_known_total += 1
                frozen_known_correct += int(pred_known == gt_cat)
                action = ("known", gt_cat)
                oracle_correct = True
                frozen_correct = int(pred_known == gt_cat)
            elif role == "novel":
                n_novel += 1
                if gt_cat in novel_id:
                    n_reuse += 1
                    action = ("existing", novel_id[gt_cat])
                    is_first = False
                else:
                    novel_id[gt_cat] = next_novel
                    next_novel += 1
                    n_first += 1
                    action = ("new", novel_id[gt_cat])
                    is_first = True
                oracle_correct = True
                frozen_correct = True
            else:
                continue
            records.append({
                "key": list(key), "frame_id": int(r["frame_id"]),
                "sid": sid, "role": role, "gt_cat": gt_cat,
                "action": list(action), "oracle_correct": oracle_correct,
                "frozen_known_correct": bool(frozen_correct),
                "is_first_novel": is_first if role == "novel" else False,
            })

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_aligned_occurrences": n_aligned,
        "n_known_occurrences": n_known,
        "n_novel_occurrences": n_novel,
        "n_first_novel_occurrences": n_first,
        "n_novel_reuse_occurrences": n_reuse,
        "n_novel_categories": len(novel_id),
        "oracle_action_accuracy": 1.0,
        "frozen_known_strict_accuracy": (
            (frozen_known_correct + n_novel) / n_aligned if n_aligned else None),
        "frozen_known_occurrence_accuracy": (
            frozen_known_correct / frozen_known_total if frozen_known_total else None),
        "novel_oracle_accuracy": 1.0,
        "reuse_opportunity_share": n_reuse / n_novel if n_novel else None,
    }
    (out / "oracle.json").write_text(json.dumps(summary, indent=2))
    (out / "occurrences.json").write_text(json.dumps(records, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
