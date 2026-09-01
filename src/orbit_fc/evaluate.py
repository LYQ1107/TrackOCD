"""Evaluate ORBIT-FC on meta-dev proxy and official validation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.orbit.evaluate import build_known, embed_track
from src.orbit.protocol import (
    load_gt,
    load_stream,
    subset_ids,
    load_frame_features,
    load_mean_features,
    load_train_labels,
    meta_classes,
)
from src.orbit_fc.causal_memory import CausalNovelMemory
from src.orbit_fc.model import ORBITFCModel
from src.orbit_fc.protocol import (
    ROOT,
    frozen_known_protos,
    known_stats,
    novel_stats,
    stats_to_tensor,
    build_meta_proxy_rows,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def load_fc_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    model = ORBITFCModel(dim=768, bottleneck=ck.get("bottleneck", 128),
                         gate_dim=ck.get("gate_dim", 13),
                         reuse_dim=ck.get("reuse_dim", 11),
                         hidden=64, use_adapter=True)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def run_stream_fc(model, rows, feats, known_protos, radii, frozen_protos, device,
                  gate_threshold=0.5, reuse_threshold=0.5, novel_update_rate=0.2,
                  update_radius=False, include_anchor=True, mean_feats=None,
                  return_logs=False):
    mem = CausalNovelMemory(known_protos, radii, novel_update_rate=novel_update_rate)
    P_known = np.stack([known_protos[c] for c in sorted(known_protos)]).astype(np.float32)
    known_ids = sorted(known_protos)
    P_frozen = np.stack([frozen_protos[c] for c in sorted(frozen_protos)]).astype(np.float32)
    preds = []
    logs = []
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        z, rel = embed_track(model, feats[sid], device)
        z0 = None
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        anchor = -1.0
        P_novel = np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)]).astype(np.float32) if mem.novel else np.empty((0, 768), dtype=np.float32)
        nid = None
        best_n = second_n = -1.0
        margin_n = 0.0
        dist_n = 1.0
        if P_novel.shape[0]:
            ns = P_novel @ z
            best_n = float(ns.max())
            order = np.argsort(ns)[::-1]
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
            nid = int(sorted(mem.novel)[int(order[0])])
            r_n = mem.novel_radii.get(nid, 0.3)
            dist_n = (1.0 - best_n) / max(r_n, 1e-6)
        if P_frozen.shape[0]:
            # original DINO anchor score from the frozen mean cache
            z0 = mean_feats.get(sid) if mean_feats is not None else None
            if z0 is not None:
                anchor = float(np.max(z0 @ P_frozen.T))
        gs = known_stats(z, P_known, radii, known_ids=known_ids, anchor_best=anchor,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel, track_len=len(feats[sid]),
                         n_novel=len(mem.novel), include_anchor=include_anchor)
        with torch.no_grad():
            gate_logit = float(model.gate_forward(stats_to_tensor(gs, device))[0])
        if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_threshold and kid is not None:
            preds.append({"sample_id": sid, "stream_order": i, "prediction_type": "known",
                          "semantic_category_id": int(kid)})
            action = "KNOWN"
            out_id = kid
            support = 0
            age = -1
        else:
            rs = novel_stats(z, P_novel, mem.novel_counts, mem.novel_radii,
                             novel_ids=sorted(mem.novel) if mem.novel else None,
                             best_k=best_k, margin_k=_margin_k(ks),
                             rel=rel, track_len=len(feats[sid]),
                             n_novel=len(mem.novel),
                             age_norm=mem.age(nid, i) if nid is not None else 0.0)
            with torch.no_grad():
                reuse_logit = float(model.reuse_forward(stats_to_tensor(rs, device))[0])
            prob_reuse = float(torch.sigmoid(torch.as_tensor(reuse_logit)))
            if prob_reuse >= reuse_threshold and nid is not None:
                support = mem.support(nid)
                age = mem.age(nid, i)
                cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
                mem.update_novel(nid, z, cos_to_center=cos_to_center,
                                 update_radius=update_radius)
                preds.append({"sample_id": sid, "stream_order": i, "prediction_type": "novel",
                              "virtual_category_id": int(nid)})
                action = "EXISTING_NOVEL"
                out_id = nid
            else:
                vid = mem.create_novel(z, created_at=i)
                preds.append({"sample_id": sid, "stream_order": i, "prediction_type": "novel",
                              "virtual_category_id": int(vid)})
                action = "NEW_NOVEL"
                out_id = vid
                support = 0
                age = 0
        if return_logs:
            logs.append({
                "sample_id": sid, "arrival_index": i, "predicted_action": action,
                "predicted_virtual_novel_id": out_id if action != "KNOWN" else None,
                "predicted_known_id": kid if action == "KNOWN" else None,
                "best_known_similarity": best_k, "best_novel_similarity": best_n,
                "novel_margin": margin_n, "prototype_support": support,
                "prototype_age": age, "gate_logit": gate_logit,
                "reuse_logit": reuse_logit if action != "KNOWN" else float("nan"),
                "track_length": len(feats[sid]),
            })
    return preds, mem, logs


def _margin_k(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def evaluate_proxy(model, ck, device, gate_thr=0.5, reuse_thr=0.5, update_radius=False):
    rows, gt_by_sid, feats, labels = build_meta_proxy_rows()
    meta_tr = meta_classes("meta_train_classes")
    feats = {sid: f[:8] for sid, f in feats.items()}
    known_protos, radii = build_known(model, feats, labels, meta_tr, device)
    frozen = frozen_known_protos(meta_tr)
    mean_feats = load_mean_features("train_known_mean")
    preds, mem, logs = run_stream_fc(model, rows, feats, known_protos, radii, frozen,
                                     device, gate_thr, reuse_thr,
                                     novel_update_rate=ck.get("novel_update_rate", 0.2),
                                     update_radius=update_radius,
                                     include_anchor=ck.get("use_anchor", True),
                                     mean_feats=mean_feats, return_logs=True)
    ev = TrackOCDEvaluator(list(gt_by_sid.values()))
    res = ev.evaluate(preds)
    for l in logs:
        g = gt_by_sid.get(l["sample_id"], {})
        l["true_role"] = g.get("protocol_role", "?")
        l["true_class"] = g.get("ground_truth_category_id", "?")
    for l in logs:
        seen = set()
        first = True
        for ll in logs:
            if ll["true_class"] == l["true_class"] and ll["arrival_index"] < l["arrival_index"]:
                first = False
                break
        l["first_occurrence"] = first
    return res, preds, logs


def evaluate_official(model, ck, proto, subset, stream, device,
                      gate_thr=0.5, reuse_thr=0.5, update_radius=False):
    gt = load_gt(proto)
    rows = load_stream(proto, stream)
    feats = {sid: f[:8] for sid, f in load_frame_features("gt_tracks_mean").items()}
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    all_known = set(train_labels.values())
    known_protos, radii = build_known(model, train_feats, train_labels, all_known, device)
    frozen = frozen_known_protos(all_known)
    mean_feats = load_mean_features("gt_tracks_mean")
    preds, mem, logs = run_stream_fc(model, rows, feats, known_protos, radii, frozen,
                                     device, gate_thr, reuse_thr,
                                     novel_update_rate=ck.get("novel_update_rate", 0.2),
                                     update_radius=update_radius,
                                     include_anchor=ck.get("use_anchor", True),
                                     mean_feats=mean_feats, return_logs=True)
    ev = TrackOCDEvaluator(gt)
    res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
    return res, preds, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--reuse_threshold", type=float, default=0.5)
    ap.add_argument("--update_radius", action="store_true")
    ap.add_argument("--proto", default="pure")
    ap.add_argument("--subset", default="full")
    ap.add_argument("--stream", default="main_seed1027")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, ck = load_fc_model(Path(args.checkpoint), device=args.device)
    res, preds, _ = evaluate_official(model, ck, args.proto, args.subset, args.stream,
                                      args.device, args.gate_threshold,
                                      args.reuse_threshold, args.update_radius)
    print(json.dumps({k: res[k] for k in
                      ["all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                       "conditional_novel_acc", "novel_routing_recall", "novel_only_nmi",
                       "novel_only_ari", "predicted_novel_count", "novel_count_abs_error"]},
                     indent=1))


if __name__ == "__main__":
    main()
