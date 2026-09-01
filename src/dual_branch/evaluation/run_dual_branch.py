#!/usr/bin/env python3
"""Run D0 (A1 repro), D1 (shared transformer), D2 (hard dual branch),
D3-T / D3-D (oracle-route diagnostics) on Pure and OV-assisted protocols."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.discovery_encoder import DiscoveryEncoder
from src.dual_branch.models.semantic_router import SemanticRouter
from src.dual_branch.models.outputs import emit
from src.dual_branch.data.track_stream_dataset import load_stream_rows
from src.ocd_v2.common import (
    load_train_known, load_mean_features, build_prototypes,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt, subset_ids
from src.trackocd_v1.trajectory_encoder import (
    TrajectoryEncoder, load_frame_dict, encode_all, calibrate_ncm_thr,
)

OUT = PROJECT_ROOT / "outputs" / "dual_branch" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "dual_branch"
STREAMS = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
SUBSETS = ("full", "repeated", "balanced")


def load_semantic():
    ckpt = torch_load(RUNS_PREFIX / "trackocd_v1" / "traj_enc_transformer" / "model.pth")
    model = TrajectoryEncoder(len(ckpt["classes"]), variant=ckpt["variant"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().cuda()
    return model, ckpt


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu")


RUNS_PREFIX = PROJECT_ROOT / "runs"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    dino_mean_val = load_mean_features("dinov2", "gt_tracks_mean")
    dino_mean_tr, labels = load_train_known("dinov2")
    protos_mean = build_prototypes(dino_mean_tr, labels, set(labels.values()))

    model, ckpt = load_semantic()
    tr_d = load_frame_dict("dinov2", "train_known_mean")
    tr_c = load_frame_dict("clip", "train_known_mean")
    tr_ids = sorted(s for s in labels if s in tr_d and s in tr_c)
    tr_enc = encode_all(model, {"dino": tr_d, "clip": tr_c}, tr_ids)
    thr_d1 = calibrate_ncm_thr(tr_enc, labels)
    print("D1 threshold", thr_d1, flush=True)
    protos_enc = build_prototypes(tr_enc, labels, set(labels.values()))
    router = SemanticRouter(model, protos_enc, thr_d1)

    val_d = load_frame_dict("dinov2", "gt_tracks_mean")
    val_c = load_frame_dict("clip", "gt_tracks_mean")
    val_ids = sorted(s for s in val_d if s in val_c)
    val_enc = encode_all(model, {"dino": val_d, "clip": val_c}, val_ids)
    disco = DiscoveryEncoder(mean_features=dino_mean_val)

    rows_out = []
    route_masks = {}
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in SUBSETS:
            for stream in STREAMS:
                srows = load_stream_rows(stream)
                sub = subset_ids(proto, subset)
                for method in ("D0", "D1", "D2", "D3T", "D3D"):
                    preds, mask = run_method(
                        method, srows, stream, proto, gt, sub,
                        dino_mean_val, val_enc, protos_mean, protos_enc,
                        router, disco, thr_d1, 0.45,
                    )
                    route_masks[(method, proto, subset, stream)] = mask
                    ev = TrackOCDEvaluator(gt)
                    res = ev.evaluate(preds, subset_ids=sub)
                    row = {
                        "method": method, "protocol": proto, "subset": subset,
                        "seed": stream,
                        "prediction_log": preds,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    }
                    rows_out.append(row)
                    (RUNS / f"{method}_{proto}_{subset}_{stream}.json").write_text(
                        json.dumps(row, indent=1, default=str))
                    print(method, proto, subset, stream,
                          "all", round(res["all_track_acc"], 4),
                          "known", round(res["overall_known_acc"], 4),
                          "route", round(res["route_aware_novel_acc"], 4),
                          "cond", round(res["conditional_novel_acc"], 4),
                          "nmi", round(res["novel_only_nmi"], 4),
                          "ari", round(res["novel_only_ari"], 4),
                          flush=True)
    with open(OUT / "d0_d1_d2_d3_all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    # save route masks (decisions per sample) for D1/D2 identity check
    with open(RUNS / "route_masks.json", "w") as f:
        json.dump({f"{k[0]}_{k[1]}_{k[2]}_{k[3]}": v for k, v in route_masks.items()}, f)
    print("done")


def run_method(method, srows, stream, proto, gt, sub,
               dino_mean, val_enc, protos_mean, protos_enc,
               router, disco, thr_d1, thr_d0):
    preds = []
    mask = {}
    if method == "D0":
        mem = B2Memory(protos_mean, threshold=thr_d0)
        for i, r in enumerate(srows):
            vid, kind = mem.predict_one(dino_mean[r["sample_id"]], r["sample_id"], i)
            preds.append(emit(r["sample_id"], i, kind, vid if kind == "known" else None,
                              vid if kind == "novel" else None))
            mask[r["sample_id"]] = kind
    elif method == "D1":
        mem = B2Memory(protos_enc, threshold=thr_d1)
        for i, r in enumerate(srows):
            vid, kind = mem.predict_one(val_enc[r["sample_id"]], r["sample_id"], i)
            preds.append(emit(r["sample_id"], i, kind, vid if kind == "known" else None,
                              vid if kind == "novel" else None))
            mask[r["sample_id"]] = kind
    elif method == "D2":
        mem = B2Memory(protos_mean, threshold=thr_d0, novel_only=True)
        for i, r in enumerate(srows):
            is_known, kid, score = router.decide(val_enc[r["sample_id"]])
            kind = "known" if is_known else "novel"
            if is_known:
                preds.append(emit(r["sample_id"], i, "known", known_id=kid))
            else:
                vid, _ = mem.predict_one(dino_mean[r["sample_id"]], r["sample_id"], i)
                preds.append(emit(r["sample_id"], i, "novel", virtual_id=vid))
            mask[r["sample_id"]] = kind
    elif method in ("D3T", "D3D"):
        gt_by_sid = {g["sample_id"]: g for g in gt}
        mem = B2Memory(
            {},
            threshold=thr_d1 if method == "D3T" else thr_d0,
            novel_only=True,
        )
        for i, r in enumerate(srows):
            g = gt_by_sid.get(r["sample_id"])
            if g is not None and g["protocol_role"] in ("supported_known", "zero_shot_known"):
                preds.append(emit(r["sample_id"], i, "known", known_id=g["ground_truth_category_id"]))
                mask[r["sample_id"]] = "known"
            else:
                emb = val_enc[r["sample_id"]] if method == "D3T" else dino_mean[r["sample_id"]]
                vid, _ = mem.predict_one(emb, r["sample_id"], i)
                preds.append(emit(r["sample_id"], i, "novel", virtual_id=vid))
                mask[r["sample_id"]] = "novel"
    return preds, mask


if __name__ == "__main__":
    main()
