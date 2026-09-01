"""Evaluate ORBIT and train-side proxy (meta-dev) evaluation."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.orbit.action_router import KNOWN, EXISTING_NOVEL, NEW_NOVEL, ACTIONS
from src.orbit.bi_memory import BiMemory, stats_to_tensor
from src.orbit.model import ORBITModel
from src.orbit.protocol import (
    load_frame_features, load_train_labels, load_stream, load_gt, subset_ids,
    meta_classes,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_model(path: Path, stats_dim=11, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    model = ORBITModel(dim=768, bottleneck=ck.get("bottleneck", 128),
                       use_adapter=ck.get("variant") in ("D1", "D2"),
                       use_reliability=ck.get("variant") == "D2",
                       stats_dim=stats_dim)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def embed_track(model, frames: np.ndarray, device):
    if len(frames) == 0:
        return np.zeros(768, dtype=np.float32), 1.0
    x = torch.as_tensor(frames[:8], dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        out = model.aggregate(x, mask)
    z = out["z"][0].cpu().numpy().astype(np.float32)
    reliability = float(out["cos"][0].mean()) if out["cos"].numel() else 1.0
    return z, reliability


def build_known(model, features, labels, class_ids, device):
    protos = {}
    radii = {}
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if c in class_ids and sid in features:
            by_class[c].append(sid)
    for c, ids in by_class.items():
        zs = []
        for sid in ids:
            z, _ = embed_track(model, features[sid], device)
            zs.append(z)
        Z = np.stack(zs)
        p = Z.mean(axis=0)
        p = p / (np.linalg.norm(p) + 1e-12)
        protos[int(c)] = p
        cos = Z @ p
        radii[int(c)] = float(np.percentile(1.0 - cos, 50).clip(min=0.02))
    return protos, radii


def run_stream(model, rows, features, known_protos, radii, device,
               mode="joint", sequential_thr=0.45, novel_update_rate=0.2):
    mem = BiMemory(known_protos, radii, novel_update_rate=novel_update_rate)
    preds = []
    for i, r in enumerate(rows):
        z, rel = embed_track(model, features[r["sample_id"]], device)
        kid, ks = mem.known_id(z)
        nid, ns = mem.existing_novel(z)
        stats = mem.stats(z, rel, len(features[r["sample_id"]]),
                          known_id=kid, novel_id=nid)
        st = stats_to_tensor(stats, device=device)
        if mode == "sequential":
            if ks >= sequential_thr and kid is not None:
                action = KNOWN
            elif ns >= sequential_thr and nid is not None:
                action = EXISTING_NOVEL
            else:
                action = NEW_NOVEL
        else:
            with torch.no_grad():
                logits = model.action_net(st)
            action = int(logits.argmax(dim=1).item())
        if action == KNOWN and kid is not None:
            preds.append({
                "sample_id": r["sample_id"], "stream_order": i,
                "prediction_type": "known", "semantic_category_id": int(kid),
            })
        elif action == EXISTING_NOVEL and nid is not None:
            mem.update_novel(nid, z)
            preds.append({
                "sample_id": r["sample_id"], "stream_order": i,
                "prediction_type": "novel", "virtual_category_id": int(nid),
            })
        else:
            vid = mem.create_novel(z)
            preds.append({
                "sample_id": r["sample_id"], "stream_order": i,
                "prediction_type": "novel", "virtual_category_id": int(vid),
            })
    return preds, mem


def evaluate_proxy(model, device, seed=1027, max_known=600):
    """Train-side meta-dev proxy: meta-train = known, meta-dev = pseudo-novel."""
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    meta_tr = meta_classes("meta_train_classes")
    meta_dev = meta_classes("meta_dev_classes")
    known_protos, radii = build_known(model, train_feats, train_labels, meta_tr, device)
    dev_ids = [sid for sid, c in train_labels.items() if c in meta_dev and sid in train_feats]
    known_ids = [sid for sid, c in train_labels.items() if c in meta_tr and sid in train_feats]
    rng = np.random.RandomState(seed)
    rng.shuffle(known_ids)
    known_ids = known_ids[:max_known]
    rows = []
    for sid in dev_ids:
        rows.append({"sample_id": sid, "stream_order": len(rows),
                     "role": "novel", "label": train_labels[sid]})
    for sid in known_ids:
        rows.append({"sample_id": sid, "stream_order": len(rows),
                     "role": "known", "label": train_labels[sid]})
    rows.sort(key=lambda r: r["stream_order"])
    preds, _ = run_stream(model, rows, train_feats, known_protos, radii, device)
    gt_rows = [
        {"sample_id": r["sample_id"], "ground_truth_category_id": r["label"],
         "protocol_role": "supported_known" if r["role"] == "known" else "novel"}
        for r in rows
    ]
    ev = TrackOCDEvaluator(gt_rows)
    res = ev.evaluate(preds)
    return res


def evaluate_official(model, proto, subset, stream, device, mode="joint"):
    gt = load_gt(proto)
    rows = load_stream(proto, stream)
    feats = load_frame_features("gt_tracks_mean")
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    all_known = set(train_labels.values())
    known_protos, radii = build_known(model, train_feats, train_labels, all_known, device)
    preds, _ = run_stream(model, rows, feats, known_protos, radii, device)
    sub = subset_ids(proto, subset)
    ev = TrackOCDEvaluator(gt)
    res = ev.evaluate(preds, subset_ids=sub)
    return res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["joint", "sequential"], default="joint")
    ap.add_argument("--proto", choices=["pure", "ov_assisted"], default="pure")
    ap.add_argument("--subset", choices=["full", "repeated", "balanced"], default="full")
    ap.add_argument("--stream", default="main_seed1027")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, ck = load_model(Path(args.checkpoint), device=args.device)
    res, preds = evaluate_official(model, args.proto, args.subset, args.stream,
                                   args.device, mode=args.mode)
    print(json.dumps(res, indent=1, default=str))
    out = ROOT / "runs" / "orbit" / f"eval_{Path(args.checkpoint).parent.name}_{args.proto}_{args.subset}_{args.stream}.json"
    out.write_text(json.dumps({**res, "prediction_log": preds}, indent=1, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
