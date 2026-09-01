"""Oracle ceilings for the old prototype-router paradigm (Q1 dev).

O1: oracle KNOWN/NOVEL route, model identity/memory.
O2: model route, oracle novel identity assignment.
O3: oracle route + oracle identity (physical-coverage ceiling).
Dev GT used only for this ceiling decomposition, never for selection.
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
from src.iclr27_phase4u.downstream.dev_eval import qphys_from_rows
from src.iclr27_phase4u.downstream.model import (
    HierarchicalTSRCore,
    build_tsr_known_protos,
)
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase4u.data import ROOT
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["O1", "O2", "O3"], required=True)
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
    assert len(arr) == len(rows)
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
    novel_cats = sorted({labels[sid]["ground_truth_category_id"]
                         for sid in mapping.values()
                         if labels[sid]["protocol_role"] == "novel"})
    vid_map = {c: i for i, c in enumerate(novel_cats)}

    memory = NovelMemory(args.device)
    results = {}
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
                if a1 == 2:
                    continue
                if a1 == 0:
                    kl = out["known"][0]
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
            results[key] = first

    preds = []
    order = 0
    for key, sid in mapping.items():
        order += 1
        lab = labels[sid]
        role = lab["protocol_role"]
        role_known = role in ("known", "supported_known")
        role_novel = role == "novel"
        out = results[key]
        if args.mode in ("O1", "O3") and (role_known or role_novel):
            if role_known:
                if args.mode == "O3":
                    cid = lab["ground_truth_category_id"]
                else:
                    cid = out[1] if out is not None and out[0] == "known" else None
                if cid is not None:
                    preds.append({"sample_id": sid, "prediction_type": "known",
                                  "semantic_category_id": cid, "stream_order": order})
                    continue
            else:
                if args.mode == "O3":
                    vid = vid_map[lab["ground_truth_category_id"]]
                    preds.append({"sample_id": sid, "prediction_type": "novel",
                                  "virtual_category_id": vid, "stream_order": order})
                    continue
                if out is not None and out[0] in ("existing", "new"):
                    preds.append({"sample_id": sid, "prediction_type": "novel",
                                  "virtual_category_id": out[1], "stream_order": order})
                    continue
                # corrected O1: oracle routing forces a novel prediction even
                # when the model deferred; identity is a unique placeholder
                # (counts as wrong identity unless model slot matched).
                if args.mode == "O1":
                    preds.append({"sample_id": sid, "prediction_type": "novel",
                                  "virtual_category_id": 5000 + order,
                                  "stream_order": order})
                    continue
        # fallback: model's own prediction (O2 / unresolved paths)
        if out is None or out[0] == "defer":
            preds.append({"sample_id": sid, "prediction_type": "unresolved",
                          "stream_order": order})
        elif out[0] == "known":
            preds.append({"sample_id": sid, "prediction_type": "known",
                          "semantic_category_id": out[1], "stream_order": order})
        else:
            if args.mode == "O2" and role_novel:
                preds.append({"sample_id": sid, "prediction_type": "novel",
                              "virtual_category_id": vid_map[lab["ground_truth_category_id"]],
                              "stream_order": order})
            else:
                preds.append({"sample_id": sid, "prediction_type": "novel",
                              "virtual_category_id": out[1], "stream_order": order})
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    metrics = ev.evaluate(preds, metadata={"memory_size": memory.size()})
    summary = {
        "mode": args.mode,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "memory_slots": memory.size(),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
