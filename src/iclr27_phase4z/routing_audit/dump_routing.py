"""Dump per-track per-step routing evidence on Q1 dev (Stage C path).

Mirrors the O1c code path: HierarchicalTSRCore + d2_joint_v2 checkpoint,
full 48-way known universe at dev. Saves step features, TSR states,
model route, oracle route, GT role for diagnostic analysis only.
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
from src.iclr27_phase4s.model import NovelMemory
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
    tsr_protos = build_tsr_known_protos(rep, args.device)
    model = HierarchicalTSRCore(rep, tsr_protos, use_defer=False,
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
    w = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, w)
    known_idx = list(range(len(known_list)))
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    memory = NovelMemory(args.device)
    records = []
    states_all = []
    with torch.no_grad():
        for key in sorted(tracks):
            h, m = model.belief_init(1, args.device)
            track_rows = [r for r in tracks[key]
                          if feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                               int(r["image_id"]))) is not None]
            if track_rows:
                ft = np.stack([feats_by_key[(int(r["video_id"]), int(r["track_id"]),
                                             int(r["image_id"]))] for r in track_rows])
                qt = np.stack([qmap[id(r)] for r in track_rows]).astype(np.float32)
                model.begin_occurrence(torch.from_numpy(ft).to(args.device),
                                       torch.from_numpy(qt).to(args.device))
            first = None
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                z = model.encode(torch.from_numpy(f).unsqueeze(0).to(args.device))
                row_q = torch.tensor([qmap[id(r)]], device=args.device)
                row_r = torch.tensor([[float(r_scalar[id(r)])]], device=args.device)
                h, m, g = model.belief_step(z, row_q, h, m, t)
                age = torch.tensor([[float(t + 1)]], device=args.device)
                out = model.decision(h, known_idx, memory, row_q, age)
                a1 = int(out["l1_lsm"][0].argmax())
                kl = out["known"][0]
                p = torch.softmax(kl, dim=-1)
                top2 = torch.topk(p, k=2).values
                ent = -(p * torch.log(p + 1e-9)).sum(-1)
                ene = torch.logsumexp(kl, dim=-1)
                rec = {
                    "key": list(key), "t": t,
                    "l1": a1,
                    "top1_p": float(top2[0]), "margin": float(top2[0] - top2[1]),
                    "entropy": float(ent), "energy": float(ene),
                    "q": [float(x) for x in qmap[id(r)]],
                    "r": float(r_scalar[id(r)]),
                    "gt_role": r["gt_role"],
                }
                if key in mapping:
                    sid = mapping[key]
                    rec["sid"] = sid
                    rec["gt_cat"] = labels[sid]["ground_truth_category_id"]
                    rec["protocol_role"] = labels[sid]["protocol_role"]
                records.append(rec)
                states_all.append(h[0].cpu().numpy().astype(np.float32))
                if a1 == 2:
                    continue
                if a1 == 0:
                    pred = known_list[int(kl.argmax())]
                    first = ("known", pred)
                else:
                    a2 = int(out["l2_lsm"][0].argmax())
                    if a2 == 0 and memory.size() > 0:
                        slot = int(out["l2"]["novel"][0].argmax())
                        first = ("existing", slot)
                        memory.update(slot, h, float(r_scalar[id(r)]))
                    elif a2 == 1:
                        first = ("new", memory.size())
                        memory.create(h, float(r_scalar[id(r)]), {"track": key})
                    else:
                        first = ("defer", None)
                if first is not None:
                    break
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "states.npz",
                        states=np.stack(states_all).astype(np.float32))
    (out_dir / "records.json").write_text(json.dumps(records))
    (out_dir / "meta.json").write_text(json.dumps({
        "n_records": len(records), "n_states": len(states_all),
        "n_tracks": len(tracks), "n_aligned": len(mapping),
        "checkpoint": "d2_joint_v2", "universe": "full48",
    }, indent=2))
    print("dumped", len(records), "records,", len(states_all), "states")


if __name__ == "__main__":
    main()
