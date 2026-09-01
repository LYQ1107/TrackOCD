"""Q1 dev evaluation for X3 (vMF sequential posterior, frozen)."""
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
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.evaluation.dev_eval import qphys_from_rows
from src.iclr27_phase4x.components.vmf_memory import (
    CompatSemanticMemory,
    VMFSemanticMemory,
)
from src.iclr27_phase4x.evaluation.pilot_x3 import load_tsr
from src.iclr27_phase4x.likelihood.train_compatibility import CompatibilityNet
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kappa", type=float, default=16.0)
    ap.add_argument("--log-prior-new", type=float, default=-1.5)
    ap.add_argument("--log-prior-noise", type=float, default=-3.0)
    ap.add_argument("--noise-alpha", type=float, default=2.0)
    ap.add_argument("--commit-threshold", type=float, default=0.5)
    ap.add_argument("--margin-ratio", type=float, default=1.5)
    ap.add_argument("--min-age", type=int, default=2)
    ap.add_argument("--compat", default=None,
                    help="path to learned compatibility checkpoint (X4)")
    args = ap.parse_args()

    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    active_idx = list(range(len(cat_ids)))
    compat = None
    if args.compat:
        ck = torch.load(ROOT / args.compat, map_location=args.device)
        compat = CompatibilityNet().to(args.device)
        compat.load_state_dict(ck["model"])
        compat.eval()
        mem = CompatSemanticMemory(anchors, compat,
                                   log_prior_new=args.log_prior_new,
                                   log_prior_noise=args.log_prior_noise,
                                   noise_alpha=args.noise_alpha,
                                   device=args.device)
    else:
        mem = VMFSemanticMemory(anchors, kappa=args.kappa,
                                log_prior_new=args.log_prior_new,
                                log_prior_noise=args.log_prior_noise,
                                noise_alpha=args.noise_alpha, device=args.device)

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
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    results = {}
    state_stats = defaultdict(int)
    with torch.no_grad():
        for key in sorted(tracks):
            state = tsr.init_state(1, args.device)
            commit = None
            for t, r in enumerate(tracks[key]):
                f = feats_by_key.get((int(r["video_id"]), int(r["track_id"]),
                                      int(r["image_id"])))
                if f is None:
                    continue
                ft = torch.from_numpy(f).unsqueeze(0).to(args.device)
                qt = torch.tensor([qmap[id(r)]], device=args.device)
                rs = float(r_scalar[id(r)])
                s, state = tsr.step(ft, qt, state)
                if t < args.min_age:
                    continue
                post, _, _ = mem.posterior(s, float(qmap[id(r)][0]), active_idx)
                p = post[0]
                top2 = torch.topk(p, k=2).values
                if float(p.max()) < args.commit_threshold:
                    continue
                if float(top2[0] / max(top2[1], 1e-9)) < args.margin_ratio:
                    continue
                a = int(p.argmax())
                if a < len(active_idx):
                    commit = ("known", cat_ids[a], t)
                    state_stats["known_commit"] += 1
                elif a < len(active_idx) + mem.size():
                    k = a - len(active_idx)
                    commit = ("existing", k, t)
                    mem.update(k, s, rs)
                    state_stats["existing_commit"] += 1
                elif a == len(active_idx) + mem.size():
                    k = mem.create(s, rs)
                    commit = ("new", k, t)
                    state_stats["new_commit"] += 1
                else:
                    commit = ("noise", None, t)
                    state_stats["noise_commit"] += 1
                break
            results[key] = commit

    preds = []
    order = 0
    for key, sid in mapping.items():
        order += 1
        out = results[key]
        if out is None or out[0] == "noise":
            preds.append({"sample_id": sid, "prediction_type": "unresolved",
                          "stream_order": order})
        elif out[0] == "known":
            preds.append({"sample_id": sid, "prediction_type": "known",
                          "semantic_category_id": out[1], "stream_order": order})
        else:
            preds.append({"sample_id": sid, "prediction_type": "novel",
                          "virtual_category_id": out[1], "stream_order": order})
    ev = TrackOCDEvaluator([labels[sid] for sid in sorted(labels)])
    metrics = ev.evaluate(preds, metadata={"memory_size": mem.size()})
    absorption = {"novel_gt": 0, "absorbed_as_known": 0, "routed_novel": 0}
    for key, sid in mapping.items():
        if labels[sid]["protocol_role"] != "novel":
            continue
        absorption["novel_gt"] += 1
        out = results[key]
        if out is not None and out[0] == "known":
            absorption["absorbed_as_known"] += 1
        elif out is not None and out[0] in ("new", "existing"):
            absorption["routed_novel"] += 1
    summary = {
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                    for k, v in metrics.items() if k != "hungarian_assignment"},
        "n_aligned": len(mapping),
        "memory_slots": mem.size(),
        "absorption": absorption,
        "state_stats": dict(state_stats),
        "hparams": {k: getattr(args, k) for k in
                    ("kappa", "log_prior_new", "log_prior_noise", "noise_alpha",
                     "commit_threshold", "margin_ratio", "min_age")},
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
