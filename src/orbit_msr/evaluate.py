"""Evaluate ORBIT-MSR on the long-stream proxy and official validation."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import build_known, embed_track
from src.orbit.bi_memory import stats_to_tensor
from src.orbit.protocol import (
    load_gt,
    load_stream,
    subset_ids,
    load_frame_features,
    load_mean_features,
    load_train_labels,
)
from src.orbit_fc.causal_memory import CausalNovelMemory
from src.orbit_fc.model import ORBITFCModel
from src.orbit_msr.protocol import frozen_known_protos, known_stats, novel_stats
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.iclr27_phase4c.audit_common import emit_preds, assignment_from_preds
from src.iclr27_phase4d.long_stream import active_bucket, stage_of, load_stream_cache


def load_msr_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    model = ORBITFCModel(dim=768, bottleneck=ck.get("bottleneck", 128),
                         gate_dim=ck.get("gate_dim", 11),
                         reuse_dim=ck.get("reuse_dim", 11),
                         hidden=64, use_adapter=True,
                         compat_dim=ck.get("compat_dim", 0))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def embed_many(model, feats, sids, device, batch=512):
    """Batched track embedding with per-track reliability (valid frames)."""
    zs = {}
    rels = {}
    for start in range(0, len(sids), batch):
        chunk = sids[start:start + batch]
        max_t = max(len(feats[s]) for s in chunk)
        x = np.zeros((len(chunk), max_t, 768), dtype=np.float32)
        m = np.zeros((len(chunk), max_t), dtype=bool)
        for i, s in enumerate(chunk):
            f = feats[s]
            x[i, :len(f)] = f
            m[i, :len(f)] = True
        xt = torch.as_tensor(x, device=device)
        mt = torch.as_tensor(m, device=device)
        with torch.no_grad():
            out = model.aggregate(xt, mt)
        z_np = out["z"].cpu().numpy()
        cos = out["cos"]
        rel_np = ((cos * mt.float()).sum(1) / mt.float().sum(1).clamp(min=1)).cpu().numpy()
        for i, s in enumerate(chunk):
            zs[s] = z_np[i]
            rels[s] = float(rel_np[i])
    return zs, rels


def run_stream_msr(model, ck, rows, feats, labels, device, gate_thr=0.5,
                   reuse_thr=0.5, syn_mean=None, proto_feats=None):
    known_classes = sorted(set(labels.values()))
    proto_feats = proto_feats if proto_feats is not None else feats
    protos, radii = build_known(model, proto_feats, labels, set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows], device)
    mem = CausalNovelMemory(protos, radii,
                            novel_update_rate=ck.get("novel_update_rate", 0.2))
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    mean_feats = load_mean_features("train_known_mean")
    n = len(rows)
    logs = []
    for i, r in enumerate(rows):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
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
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_logit = float(model.gate_forward(stats_to_tensor(gs, device))[0])
        if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_thr and kid is not None:
            logs.append(_log(r, i, "KNOWN", kid, None, len(mem.novel), 0,
                             best_k, best_n, gate_logit, float("nan")))
            continue
        rs = novel_stats(z, P_novel, mem.novel_counts, mem.novel_radii,
                         novel_ids=sorted(mem.novel) if mem.novel else None,
                         best_k=best_k, margin_k=_margin(ks),
                         rel=rel, track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel),
                         age_norm=mem.age(nid, i) if nid is not None else 0.0,
                         mem_scale_norm=ck.get("mem_scale_norm", False))
        with torch.no_grad():
            reuse_logit = float(model.reuse_forward(stats_to_tensor(rs, device))[0])
        prob_reuse = float(torch.sigmoid(torch.as_tensor(reuse_logit)))
        if prob_reuse >= reuse_thr and nid is not None:
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False))
            logs.append(_log(r, i, "EXISTING_NOVEL", None, nid, len(mem.novel),
                             support, best_k, best_n, gate_logit, reuse_logit))
        else:
            vid = mem.create_novel(z, created_at=i)
            logs.append(_log(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                             0, best_k, best_n, gate_logit, reuse_logit))
    return logs


def _log(r, i, action, kid, vid, active, support, bk, bn, gl, rl):
    return {
        "sample_id": r["sample_id"], "arrival_index": i,
        "role": r["role"], "class": r["class"],
        "first_occurrence": r["first_occurrence"],
        "predicted_action": action, "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active, "prototype_support": support,
        "best_known_similarity": bk, "best_novel_similarity": bn,
        "gate_logit": gl, "reuse_logit": rl,
        "stage": stage_of(i, len(r) if isinstance(r, list) else 5255),
    }


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def attach_gt(logs, gt_rows):
    gt_by_sid = {g["sample_id"]: g for g in gt_rows}
    for l in logs:
        g = gt_by_sid.get(l["sample_id"], {})
        l["true_role"] = g.get("protocol_role", "?")
        l["true_class"] = g.get("ground_truth_category_id", "?")
    return logs


def mechanism_rates(logs, assignment):
    known = [l for l in logs if l["true_role"] in ("supported_known", "zero_shot_known")]
    novel = [l for l in logs if l["true_role"] == "novel"]
    routed = [l for l in novel if l["predicted_action"] != "KNOWN"]
    first = [l for l in novel if l["first_occurrence"]]
    repeated = [l for l in novel if not l["first_occurrence"]]
    wrong_existing = 0
    for l in routed:
        if l["predicted_action"] == "EXISTING_NOVEL":
            vid = int(l["predicted_virtual_novel_id"])
            if assignment.get(vid) != int(l["true_class"]):
                wrong_existing += 1
    return {
        "known_to_novel": sum(1 for l in known if l["predicted_action"] != "KNOWN") / max(len(known), 1),
        "novel_to_known": sum(1 for l in novel if l["predicted_action"] == "KNOWN") / max(len(novel), 1),
        "repeated_false_birth": sum(1 for l in repeated if l["predicted_action"] == "NEW_NOVEL") / max(len(repeated), 1),
        "wrong_existing": wrong_existing / max(len(routed), 1),
        "first_merge": sum(1 for l in first if l["predicted_action"] == "EXISTING_NOVEL") / max(len(first), 1),
        "routing_recall": len(routed) / max(len(novel), 1),
    }


def evaluate_split(logs, gt_rows, select=None):
    if select is not None:
        logs = [l for l in logs if select(l)]
    if not logs:
        return None
    preds = emit_preds(logs)
    sids = {l["sample_id"] for l in logs}
    gt = [g for g in gt_rows if g["sample_id"] in sids]
    res, ev = assignment_from_preds(preds, gt)
    return res, ev


def summarize(name, logs, gt_rows, scope="overall", select=None):
    out = evaluate_split(logs, gt_rows, select=select)
    if out is None:
        return None
    res, ev = out
    mr = mechanism_rates(logs if select is None else [l for l in logs if select(l)],
                         res["hungarian_assignment"])
    return {
        "name": name, "scope": scope,
        "all_acc": res["all_track_acc"], "known_acc": res["overall_known_acc"],
        "rn_acc": res["route_aware_novel_acc"],
        "cond_novel_acc": res["conditional_novel_acc"],
        "routing_recall": res["novel_routing_recall"],
        "nmi": res["novel_only_nmi"], "ari": res["novel_only_ari"],
        "count_error": res["novel_count_abs_error"],
        "predicted_novel_count": res["predicted_novel_count"],
        "known_to_novel": mr["known_to_novel"], "novel_to_known": mr["novel_to_known"],
        "repeated_false_birth": mr["repeated_false_birth"],
        "wrong_existing": mr["wrong_existing"], "first_merge": mr["first_merge"],
    }


def evaluate_long_stream(model, ck, device, gate_thr=0.5, reuse_thr=0.5):
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    logs = run_stream_msr(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, reuse_thr=reuse_thr,
                          syn_mean=syn_mean)
    attach_gt(logs, gt_rows)
    rows_out = []
    r = summarize(ck.get("variant", "?"), logs, gt_rows, "overall")
    rows_out.append(r)
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        r = summarize(ck.get("variant", "?"), logs, gt_rows, bucket,
                      select=lambda l, b=bucket:
                      active_bucket(l["active_novel_prototypes"]) == b)
        if r:
            rows_out.append(r)
    for stage in ["early", "middle", "late"]:
        r = summarize(ck.get("variant", "?"), logs, gt_rows, stage,
                      select=lambda l, s=stage: l["stage"] == s)
        if r:
            rows_out.append(r)
    return rows_out, logs


def evaluate_official(model, ck, device, gate_thr=0.5, reuse_thr=0.5):
    gt = load_gt("pure")
    rows = load_stream("pure", "main_seed1027")
    gt_by_sid = {g["sample_id"]: g for g in gt}
    seen = set()
    for r in rows:
        g = gt_by_sid[r["sample_id"]]
        r["role"] = ("known" if g["protocol_role"] in
                     ("supported_known", "zero_shot_known") else "novel")
        r["class"] = g["ground_truth_category_id"]
        r["first_occurrence"] = r["class"] not in seen
        seen.add(r["class"])
    feats = {sid: f[:8] for sid, f in load_frame_features("gt_tracks_mean").items()}
    labels = load_train_labels()
    train_feats = {sid: f[:8] for sid, f in load_frame_features("train_known_mean").items()}
    logs = run_stream_msr(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, reuse_thr=reuse_thr,
                          proto_feats=train_feats)
    attach_gt(logs, gt)
    return logs, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--reuse_threshold", type=float, default=0.5)
    ap.add_argument("--official", action="store_true")
    args = ap.parse_args()
    model, ck = load_msr_model(Path(args.checkpoint))
    if args.official:
        logs, gt = evaluate_official(model, ck, "cuda", args.gate_threshold,
                                     args.reuse_threshold)
        res, ev = evaluate_split(logs, gt)
        print(json.dumps({k: res[k] for k in
                          ["all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                           "conditional_novel_acc", "novel_routing_recall", "novel_only_nmi",
                           "novel_only_ari", "predicted_novel_count", "novel_count_abs_error"]},
                         indent=1))
    else:
        rows_out, logs = evaluate_long_stream(model, ck, "cuda",
                                              args.gate_threshold,
                                              args.reuse_threshold)
        for r in rows_out:
            print(r, flush=True)


if __name__ == "__main__":
    main()
