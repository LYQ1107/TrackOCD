"""Q1 dev end-to-end evaluation: Phase 4Z router + frozen O1c downstream.

The router replaces only the level-1 decision. Known commits use the frozen
known classification at the commit step; novel commits use the frozen
level-2 / NovelMemory path; no commit by track end -> unresolved. Tracks are
processed in sorted key order (same as O1c) and evaluated by the corrected
TrackOCDEvaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.dev_eval import compute_r_phys, r_phys_calibration
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    Q2_DEV,
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
from src.iclr27_phase4z.evidence.step_evidence import step_evidence
from src.iclr27_phase4z.training.train_routing import (
    GRURouter,
    MLPRouter,
    feature_sets,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def load_router(path: Path, device: str):
    ck = torch.load(path / "router.pth", map_location=device)
    mode = ck["mode"]
    if mode == "gru":
        model = GRURouter(283, ck["args"].get("hidden", 96)).to(device)
    else:
        in_dim = {"static": 283, "meanpool": 283, "singleframe": 283,
                  "aggregated": 27 * 4 + 27 + 256 + 256 + 1}[mode]
        model = MLPRouter(in_dim, ck["args"].get("hidden", 128)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    norm = {k: ck[k] for k in ("ev_mean", "ev_std", "h_mean", "h_std")}
    return model, mode, ck, norm


def router_logits(model, mode, h, ev, state):
    """Returns (logits (1,3), next_state)."""
    if mode == "gru":
        x = torch.from_numpy(np.concatenate([h, ev], axis=0)).unsqueeze(0).to(
            next(model.parameters()).device)
        state = model.gru(x, state)
        logits = model.head(state)
        return logits, state
    return None, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frontend", choices=["q1", "q2"], default="q1")
    args = ap.parse_args()

    router, mode, ck, norm = load_router(ROOT / args.router, args.device)
    print("router mode", mode, "tau", args.tau, flush=True)

    known_list = sorted(known_ids())
    full_idx = list(range(len(known_list)))
    rep = TSR(arch="gru").to(args.device)
    ck0 = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                     map_location=args.device)
    tsr_sd = {k[len("rep."):]: v for k, v in ck0["model"].items() if k.startswith("rep.")}
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

    prop_path = Q1_DEV if args.frontend == "q1" else Q2_DEV
    feat_path = ("outputs/iclr27_phase4s/q1_features/feats.npz"
                 if args.frontend == "q1"
                 else "outputs/iclr27_phase4s/q2_features/feats.npz")
    rows = load_proposals(Path(prop_path))
    arr = np.load(ROOT / feat_path)["feats"]
    qmap = qphys_from_rows(rows)
    feats_by_key = {}
    for i, r in enumerate(rows):
        feats_by_key[(int(r["video_id"]), int(r["track_id"]), int(r["image_id"]))] = arr[i]
    tracks = group_tracks(rows)
    w = r_phys_calibration(rows)
    r_scalar = compute_r_phys(rows, w)
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    memory = NovelMemory(args.device)
    results = {}
    commits = []
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
            out = None
            commit = None
            gstate = (torch.zeros(1, ck["args"].get("hidden", 96),
                                  device=args.device) if mode == "gru" else None)
            evs = []
            hs = []
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                zt = model.encode(torch.from_numpy(f).unsqueeze(0).to(args.device))
                row_q = torch.tensor([qmap[id(r)]], device=args.device)
                row_r = torch.tensor([[float(r_scalar[id(r)])]], device=args.device)
                h, m, g = model.belief_step(zt, row_q, h, m, t)
                age = torch.tensor([[float(t + 1)]], device=args.device)
                ev, l1p, kl_f = step_evidence(
                    model, h, row_q, age, qmap[id(r)], float(r_scalar[id(r)]),
                    full_idx, full_idx)
                ev = np.concatenate([ev[:18], ev[21:]]).astype(np.float32)
                ev = (ev - norm["ev_mean"]) / norm["ev_std"]
                hnp = h[0].cpu().numpy().astype(np.float32)
                hnp = (hnp - norm["h_mean"]) / norm["h_std"]
                hs.append(hnp)
                evs.append(ev)
                if mode == "gru":
                    logits, gstate = router_logits(router, mode, hnp, ev, gstate)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                else:
                    if mode == "singleframe" and t > 0:
                        break
                    if mode in ("static", "meanpool", "aggregated"):
                        feats = feature_sets([{
                            "ev": np.stack(evs), "h": np.stack(hs),
                            "role": 0, "len": len(evs),
                        }], mode)
                        fv = feats[-1][0]
                    else:
                        fv = np.concatenate([hnp, ev])
                    x = torch.from_numpy(fv).unsqueeze(0).to(args.device)
                    probs = torch.softmax(router(x), dim=-1)[0].cpu().numpy()
                a = int(probs.argmax())
                if a != 2 and probs[a] >= args.tau:
                    if a == 0:
                        pred = known_list[int(kl_f[0].argmax())]
                        commit = ("known", pred, t)
                        break
                    d = model.decision(h, full_idx, memory, row_q, age)
                    a2 = int(d["l2_lsm"][0].argmax())
                    if a2 == 0 and memory.size() > 0:
                        slot = int(d["l2"]["novel"][0].argmax())
                        memory.update(slot, h, float(r_scalar[id(r)]))
                        commit = ("existing", slot, t)
                    elif a2 == 1:
                        slot = memory.size()
                        memory.create(h, float(r_scalar[id(r)]), {"track": key})
                        commit = ("new", slot, t)
                    else:
                        commit = ("defer", None, t)
                    break
            results[key] = commit
            if commit:
                commits.append((key, commit[0], commit[1], commit[2] + 1))

    preds = []
    order = 0
    for key, sid in mapping.items():
        order += 1
        lab = labels[sid]
        out = results.get(key)
        if out is None or out[0] == "defer":
            preds.append({"sample_id": sid, "prediction_type": "unresolved",
                          "stream_order": order})
        elif out[0] == "known":
            preds.append({"sample_id": sid, "prediction_type": "known",
                          "semantic_category_id": out[1], "stream_order": order})
        else:
            preds.append({"sample_id": sid, "prediction_type": "novel",
                          "virtual_category_id": out[1], "stream_order": order})
    evr = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    metrics = evr.evaluate(preds, metadata={"memory_size": memory.size()})
    summary = {
        "router": str(args.router), "mode": mode, "tau": args.tau,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "memory_slots": memory.size(),
        "n_commits": len(commits),
    }
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
