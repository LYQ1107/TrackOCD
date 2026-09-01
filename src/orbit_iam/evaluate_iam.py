"""Evaluate ORBIT-IAM on long-stream proxy and official Pure Full.

Decision policy: known gate as C1; for novel tracks, compatibility q is
computed for every active prototype with the small MLP, and EXISTING_NOVEL
is taken iff q_best >= compat_thr and (memory < 2 or q_best - q_second >=
compat_margin).  No oracle K, no future access, no historical rewrite.
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

from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_stream,
    load_train_labels,
)
from src.orbit_msr.evaluate import (
    attach_gt,
    embed_many,
    load_msr_model,
    mechanism_rates,
    summarize,
)
from src.orbit_msr.protocol import known_stats, novel_stats
from src.iclr27_phase4c.audit_common import emit_preds, assignment_from_preds
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)
from src.orbit_iam.compat import compat_matrix_for_track
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_iam.model import ORBITIAMModel


def load_iam_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    compat_dim = ck.get("compat_dim", 6)
    sd = ck["state_dict"]
    reuse_dim = int(sd["reuse.net.0.weight"].shape[1])
    gate_dim = int(sd["gate.net.0.weight"].shape[1])
    state_dim = int(ck.get("state_dim", 0))
    model = ORBITIAMModel(dim=768, bottleneck=ck.get("bottleneck", 128),
                          gate_dim=gate_dim, reuse_dim=reuse_dim,
                          hidden=64, use_adapter=True,
                          compat_dim=compat_dim, state_dim=state_dim)
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ck


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def run_iam_stream(model, ck, rows, feats, labels, device, gate_thr=0.5,
                   compat_thr=0.5, compat_margin=0.02, syn_mean=None,
                   proto_feats=None, zs_rels=None, quarantine_support=0,
                   gate_mode="base", bias_head=None):
    from src.orbit.evaluate import build_known
    known_classes = sorted(set(labels.values()))
    proto_feats = proto_feats if proto_feats is not None else feats
    protos, radii = build_known(model, proto_feats, labels,
                                set(known_classes), device)
    if zs_rels is None:
        zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                              device)
    else:
        zs, rels = zs_rels
    mem = IamMemory(protos, radii,
                    novel_update_rate=ck.get("novel_update_rate", 0.2))
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    feat_names = [f.strip() for f in ck.get("compat_feats", "sim,margin,radius,support,mem,rel").split(",") if f.strip()]
    logs = []
    n = len(rows)
    for i, r in enumerate(rows):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                   .astype(np.float32)) if mem.novel else np.empty((0, 768), dtype=np.float32)
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
        if gate_mode == "state":
            st = mem.state_summary()
            state_vec = [st["log_mem"], st["mean_support"],
                         st["low_support_ratio"], st["mean_dispersion"]]
            with torch.no_grad():
                gate_logit = float(model.gate_forward(
                    torch.as_tensor([gs + state_vec], dtype=torch.float32,
                                    device=device))[0])
        else:
            with torch.no_grad():
                gate_logit = float(model.gate_forward(
                    torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
            if gate_mode == "residual":
                st = mem.state_summary()
                state_vec = [st["log_mem"], st["mean_support"],
                             st["low_support_ratio"], st["mean_dispersion"]]
                with torch.no_grad():
                    base_logit = model.gate_forward(
                        torch.as_tensor([gs], dtype=torch.float32, device=device))
                    st_t = torch.as_tensor([state_vec], dtype=torch.float32,
                                           device=device)
                    if bias_head is not None:
                        b = bias_head(st_t)
                        gate_logit = float((base_logit - b)[0])
                    else:
                        gate_logit = float(model.gate_logit_with_bias(
                            base_logit, st_t)[0])
        if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_thr and kid is not None:
            logs.append(_log(r, i, "KNOWN", kid, None, len(mem.novel), 0,
                             best_k, best_n, gate_logit, float("nan"),
                             float("nan"), n))
            continue
        # compatibility over all active prototypes
        q_best = -1.0
        q_second = -1.0
        states = {vid: mem.state(vid) for vid in sorted(mem.novel)}
        if P_novel.shape[0]:
            X = compat_matrix_for_track(z, {vid: mem.novel[vid]["proto"]
                                            for vid in sorted(mem.novel)},
                                        states, len(mem.novel), rel, margin_n,
                                        feat_names)
            with torch.no_grad():
                q = torch.sigmoid(model.compat_forward(
                    torch.as_tensor(X, dtype=torch.float32, device=device))).cpu().numpy()
            if quarantine_support > 0:
                q = q * np.array([min(1.0, states[vid]["support"] / quarantine_support)
                                  for vid in sorted(mem.novel)],
                                 dtype=np.float32)
            if np.isnan(q).any():
                bad = np.isnan(X).any(axis=1)
                print("NAN_Q sample", r["sample_id"], "mem", len(mem.novel),
                      "Xnan", int(bad.sum()), flush=True)
                idx = int(np.argmax(bad)) if bad.any() else 0
                print("Xrow", X[idx], "q", q[idx], flush=True)
                raise RuntimeError("NaN in compat input")
            if q.shape[0]:
                qorder = np.argsort(q)[::-1]
                q_best = float(q[qorder[0]])
                q_second = float(q[qorder[1]]) if q.shape[0] >= 2 else -1.0
                nid = int(sorted(mem.novel)[int(qorder[0])])
        reuse_ok = (q_best >= compat_thr
                    and (len(mem.novel) < 2 or q_best - q_second >= compat_margin))
        if reuse_ok and nid is not None:
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            logs.append(_log(r, i, "EXISTING_NOVEL", None, nid, len(mem.novel),
                             support, best_k, best_n, gate_logit, q_best,
                             q_best - q_second, n))
        else:
            vid = mem.create_novel(z, created_at=i)
            logs.append(_log(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                             0, best_k, best_n, gate_logit, q_best,
                             q_best - q_second, n))
    return logs


def _log(r, i, action, kid, vid, active, support, bk, bn, gl, qb, qm, n):
    return {
        "sample_id": r["sample_id"], "arrival_index": i,
        "role": r["role"], "class": r["class"],
        "first_occurrence": r["first_occurrence"],
        "predicted_action": action, "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active, "prototype_support": support,
        "best_known_similarity": bk, "best_novel_similarity": bn,
        "gate_logit": gl, "compat_best": qb, "compat_margin": qm,
        "stage": stage_of(i, n),
    }


def summarize_select(name, logs, gt_rows, scope, select=None):
    return summarize(name, logs, gt_rows, scope, select)


def evaluate_long(model, ck, device, gate_thr=0.5, compat_thr=0.5,
                  compat_margin=0.02, quarantine_support=0, gate_mode="base",
                  bias_head=None):
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    logs = run_iam_stream(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          compat_margin=compat_margin, syn_mean=syn_mean,
                          quarantine_support=quarantine_support,
                          gate_mode=gate_mode, bias_head=bias_head)
    attach_gt(logs, gt_rows)
    out = []
    r = summarize(ck.get("variant", "IAM"), logs, gt_rows, "overall")
    if r:
        out.append(r)
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        r = summarize(ck.get("variant", "IAM"), logs, gt_rows, bucket,
                      select=lambda l, b=bucket:
                      active_bucket(l["active_novel_prototypes"]) == b)
        if r:
            out.append(r)
    for stage in ["early", "middle", "late"]:
        r = summarize(ck.get("variant", "IAM"), logs, gt_rows, stage,
                      select=lambda l, s=stage: l["stage"] == s)
        if r:
            out.append(r)
    # real-only and synthetic-only meta rows
    r_real = summarize(ck.get("variant", "IAM"), logs, gt_rows, "real_only",
                       select=lambda l: l["role"] == "novel"
                       and int(l["class"]) < 1000000)
    r_syn = summarize(ck.get("variant", "IAM"), logs, gt_rows, "synthetic_only",
                      select=lambda l: l["role"] == "novel"
                      and int(l["class"]) >= 1000000)
    return out, logs, r_real, r_syn


def evaluate_official_iam(model, ck, device, gate_thr=0.5, compat_thr=0.5,
                          compat_margin=0.02, quarantine_support=0,
                          gate_mode="base", bias_head=None):
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
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    logs = run_iam_stream(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          compat_margin=compat_margin,
                          proto_feats=train_feats,
                          quarantine_support=quarantine_support,
                          gate_mode=gate_mode, bias_head=bias_head)
    attach_gt(logs, gt)
    return logs, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.5)
    ap.add_argument("--compat_margin", type=float, default=0.02)
    ap.add_argument("--quarantine_support", type=int, default=0)
    ap.add_argument("--gate_mode", choices=["base", "state", "residual"],
                    default="base")
    ap.add_argument("--bias_checkpoint", default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--out_prefix", default="candidate")
    args = ap.parse_args()
    model, ck = load_iam_model(ROOT / args.checkpoint)
    bias_head = None
    if args.bias_checkpoint:
        bck = torch.load(ROOT / args.bias_checkpoint, map_location="cpu")
        bias_head = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        bias_head.load_state_dict(bck["bias"])
        bias_head.eval().to("cuda")
    if args.official:
        logs, gt = evaluate_official_iam(model, ck, "cuda",
                                         args.gate_threshold,
                                         args.compat_threshold,
                                         args.compat_margin,
                                         args.quarantine_support,
                                         args.gate_mode, bias_head)
        res, ev = assignment_from_preds(emit_preds(logs), gt)
        mr = mechanism_rates(logs, res["hungarian_assignment"])
        row = {
            "candidate": args.out_prefix,
            "all_acc": res["all_track_acc"],
            "known_acc": res["overall_known_acc"],
            "rn_acc": res["route_aware_novel_acc"],
            "cond_novel_acc": res["conditional_novel_acc"],
            "routing_recall": res["novel_routing_recall"],
            "nmi": res["novel_only_nmi"], "ari": res["novel_only_ari"],
            "count_error": res["novel_count_abs_error"],
            "predicted_novel_count": res["predicted_novel_count"],
            "known_to_novel": mr["known_to_novel"],
            "novel_to_known": mr["novel_to_known"],
            "repeated_false_birth": mr["repeated_false_birth"],
            "wrong_existing": mr["wrong_existing"],
            "first_merge": mr["first_merge"],
        }
        print(json.dumps(row, indent=1), flush=True)
        return row, logs
    rows_out, logs, r_real, r_syn = evaluate_long(model, ck, "cuda",
                                                  args.gate_threshold,
                                                  args.compat_threshold,
                                                  args.compat_margin,
                                                  args.quarantine_support,
                                                  args.gate_mode, bias_head)
    for r in rows_out:
        print(r, flush=True)
    print("real_only:", r_real, flush=True)
    print("synthetic_only:", r_syn, flush=True)
    return rows_out, logs, r_real, r_syn


if __name__ == "__main__":
    main()
