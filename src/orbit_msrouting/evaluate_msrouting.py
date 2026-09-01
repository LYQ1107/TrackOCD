"""Evaluate ORBIT-MSRouting (Phase 4G) on long-stream proxy and official.

G0/G1/G2 share the Phase 4F decision pipeline (gate -> compatibility ->
reuse/birth), with the known gate optionally conditioned on legal
memory-state features.  No oracle K, no future access, no historical
rewrite.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
    mechanism_rates,
    summarize,
)
from src.orbit_msr.protocol import known_stats
from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)
from src.orbit_iam.compat import build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_msrouting.model import load_msrouting_model
from src.orbit_msrouting.state_features import MemoryStateTracker


def load_msrouting_checkpoint(path, device="cuda", gate_mode=None,
                              state_feats=None):
    model, ck = load_msrouting_model(path, device, gate_mode=gate_mode,
                                     state_feats=state_feats)
    return model, ck


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def run_msrouting_stream(model, ck, rows, feats, labels, device,
                         gate_thr=0.5, compat_thr=0.45, compat_margin=0.05,
                         syn_mean=None, proto_feats=None, zs_rels=None,
                         state_window=32, zero_state=False):
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
    feat_names = [f.strip() for f in ck.get(
        "compat_feats", "sim,margin,radius,support,conf,mem,rel").split(",")
        if f.strip()]
    state_names = [f.strip() for f in ck.get(
        "state_feats", "log_mem,low_support_ratio,mean_support,"
                       "recent_birth_rate,high_disp_ratio").split(",")
        if f.strip()]
    gate_mode = ck.get("gate_mode", "G0")
    tracker = MemoryStateTracker(window=int(state_window))
    logs = []
    n = len(rows)
    for i, r in enumerate(rows):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        P_novel = (np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)])
                   .astype(np.float32)) if mem.novel else np.empty(
            (0, 768), dtype=np.float32)
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
        state_vec = tracker.compute(mem, state_names)
        ev = torch.as_tensor([gs], dtype=torch.float32, device=device)
        if gate_mode in ("G1", "G2"):
            if zero_state:
                st = torch.zeros(1, len(state_names), dtype=torch.float32,
                                 device=device)
            else:
                st = torch.as_tensor([state_vec], dtype=torch.float32,
                                     device=device)
        else:
            st = None
        with torch.no_grad():
            gate_logit = float(model.gate_logit(ev, st)[0])
        gate_prob = float(torch.sigmoid(torch.as_tensor(gate_logit)))
        state_sum = tracker.summary(mem)
        if gate_prob >= gate_thr and kid is not None:
            tracker.note_action("KNOWN")
            logs.append(_log(r, i, "KNOWN", kid, None, len(mem.novel), 0,
                             best_k, best_n, gate_logit, -1.0, -1.0, -1.0,
                             n, state_sum))
            continue
        q_best = -1.0
        q_second = -1.0
        birth_logit = -1.0
        states = {vid: mem.state(vid) for vid in sorted(mem.novel)}
        if P_novel.shape[0]:
            X_rows = []
            for vid in sorted(mem.novel):
                stv = states[vid]
                X_rows.append(build_compat_features(
                    z, mem.novel[vid]["proto"], stv["radius"], stv["support"],
                    stv["conf"], len(mem.novel), rel, margin_n, feat_names))
            X = torch.as_tensor(np.asarray(X_rows, dtype=np.float32),
                                device=device)
            with torch.no_grad():
                q = torch.sigmoid(model.compat_forward(X)).cpu().numpy()
            if q.shape[0]:
                qorder = np.argsort(q)[::-1]
                q_best = float(q[qorder[0]])
                q_second = (float(q[qorder[1]])
                            if q.shape[0] >= 2 else -1.0)
                nid = int(sorted(mem.novel)[int(qorder[0])])
        reuse_ok = (q_best >= compat_thr
                    and (len(mem.novel) < 2
                         or q_best - q_second >= compat_margin))
        if reuse_ok and nid is not None:
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            tracker.note_action("EXISTING_NOVEL", nid)
            logs.append(_log(r, i, "EXISTING_NOVEL", None, nid, len(mem.novel),
                             support, best_k, best_n, gate_logit, q_best,
                             q_best - q_second, birth_logit, n, state_sum))
        else:
            vid = mem.create_novel(z, created_at=i)
            tracker.note_action("NEW_NOVEL", vid)
            logs.append(_log(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                             0, best_k, best_n, gate_logit, q_best,
                             q_best - q_second, birth_logit, n, state_sum))
    return logs


def _log(r, i, action, kid, vid, active, support, bk, bn, gl, qb, qm, bl,
         n, state_sum):
    row = {
        "sample_id": r["sample_id"], "arrival_index": i,
        "role": r["role"], "class": r["class"],
        "first_occurrence": r["first_occurrence"],
        "predicted_action": action, "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active, "prototype_support": support,
        "best_known_similarity": bk, "best_novel_similarity": bn,
        "gate_logit": gl, "compat_best": qb, "compat_margin": qm,
        "birth_logit": bl, "stage": stage_of(i, n),
        "memory_bucket": active_bucket(active),
    }
    row.update(state_sum)
    return row


def bucket_rows(logs, gt_rows, name):
    """Per-memory-bucket N2K/K2N/RN/ARI/count from a single stream replay."""
    out = []
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        sel = [l for l in logs
               if active_bucket(l["active_novel_prototypes"]) == bucket]
        if not sel:
            continue
        preds = emit_preds(sel)
        sids = {l["sample_id"] for l in sel}
        gt = [g for g in gt_rows if g["sample_id"] in sids]
        res, _ = assignment_from_preds(preds, gt)
        mr = mechanism_rates(sel, res["hungarian_assignment"])
        out.append({
            "candidate": name, "bucket": bucket, "n": len(sel),
            "n2k": mr["novel_to_known"], "k2n": mr["known_to_novel"],
            "rn": res["route_aware_novel_acc"],
            "cond": res["conditional_novel_acc"],
            "ari": res["novel_only_ari"],
            "count_error": res["novel_count_abs_error"],
            "nmi": res["novel_only_nmi"],
        })
    return out


def evaluate_long_msrouting(model, ck, device, gate_thr=0.5,
                            compat_thr=0.45, compat_margin=0.05):
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    logs = run_msrouting_stream(model, ck, rows, feats, labels, device,
                                gate_thr=gate_thr, compat_thr=compat_thr,
                                compat_margin=compat_margin,
                                syn_mean=syn_mean)
    attach_gt(logs, gt_rows)
    out = []
    r = summarize(ck.get("variant", "G"), logs, gt_rows, "overall")
    if r:
        out.append(r)
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        r = summarize(ck.get("variant", "G"), logs, gt_rows, bucket,
                      select=lambda l, b=bucket:
                      active_bucket(l["active_novel_prototypes"]) == b)
        if r:
            out.append(r)
    r_real = summarize(ck.get("variant", "G"), logs, gt_rows, "real_only",
                       select=lambda l: l["role"] == "novel"
                       and int(l["class"]) < 1000000)
    r_syn = summarize(ck.get("variant", "G"), logs, gt_rows, "synthetic_only",
                      select=lambda l: l["role"] == "novel"
                      and int(l["class"]) >= 1000000)
    return out, logs, r_real, r_syn


def evaluate_official_msrouting(model, ck, device, gate_thr=0.5,
                                compat_thr=0.45, compat_margin=0.05,
                                zero_state=False):
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
    feats = {sid: f[:8] for sid, f in
             load_frame_features("gt_tracks_mean").items()}
    labels = load_train_labels()
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    logs = run_msrouting_stream(model, ck, rows, feats, labels, device,
                                gate_thr=gate_thr, compat_thr=compat_thr,
                                compat_margin=compat_margin,
                                proto_feats=train_feats, zero_state=zero_state)
    attach_gt(logs, gt)
    return logs, gt


def result_row(logs, gt_rows, name):
    res, _ = assignment_from_preds(emit_preds(logs), gt_rows)
    mr = mechanism_rates(logs, res["hungarian_assignment"])
    return {
        "candidate": name,
        "all_acc": res["all_track_acc"],
        "known_acc": res["overall_known_acc"],
        "rn_acc": res["route_aware_novel_acc"],
        "cond_novel_acc": res["conditional_novel_acc"],
        "routing_recall": res["novel_routing_recall"],
        "nmi": res["novel_only_nmi"],
        "ari": res["novel_only_ari"],
        "count_error": res["novel_count_abs_error"],
        "predicted_novel_count": res["predicted_novel_count"],
        "known_to_novel": mr["known_to_novel"],
        "novel_to_known": mr["novel_to_known"],
        "repeated_false_birth": mr["repeated_false_birth"],
        "wrong_existing": mr["wrong_existing"],
        "first_merge": mr["first_merge"],
    }


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--out_prefix", default="candidate")
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--trajectory_csv", default=None)
    ap.add_argument("--bucket_csv", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    model, ck = load_msrouting_checkpoint(args.checkpoint, args.device)
    if args.official:
        logs, gt = evaluate_official_msrouting(
            model, ck, args.device, args.gate_threshold,
            args.compat_threshold, args.compat_margin)
        row = result_row(logs, gt, args.out_prefix)
        print(json.dumps(row, indent=1))
        if args.out_csv:
            write_csv(args.out_csv, [row])
        if args.trajectory_csv:
            write_csv(args.trajectory_csv, logs)
        if args.bucket_csv:
            write_csv(args.bucket_csv, bucket_rows(logs, gt, args.out_prefix))
        return
    out, logs, r_real, r_syn = evaluate_long_msrouting(
        model, ck, args.device, args.gate_threshold,
        args.compat_threshold, args.compat_margin)
    for r in out:
        print(json.dumps(r, default=str), flush=True)
    print("REAL_ONLY", json.dumps(r_real, default=str), flush=True)
    print("SYN_ONLY", json.dumps(r_syn, default=str), flush=True)
    if args.out_csv:
        write_csv(args.out_csv, out)
    if args.trajectory_csv:
        rows, gt_rows, _, _ = load_stream_cache()
        write_csv(args.trajectory_csv, logs)
    if args.bucket_csv:
        _, gt_rows, _, _ = load_stream_cache()
        write_csv(args.bucket_csv, bucket_rows(logs, gt_rows, args.out_prefix))


if __name__ == "__main__":
    main()
