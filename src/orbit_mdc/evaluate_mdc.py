"""Evaluate ORBIT-MDC on long-stream proxy and official Pure Full.

Decision policy: known gate as C1; for non-known tracks, compatibility q
over all active prototypes (optionally quarantine-influence scaled); reuse
decision by learned birth head (if checkpoint has one) or by
q_best >= compat_thr and q margin.  No oracle K, no future access, no
historical rewrite.
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

from src.orbit.protocol import load_frame_features, load_gt, load_stream, load_train_labels
from src.orbit_msr.evaluate import attach_gt, embed_many, mechanism_rates, summarize
from src.orbit_msr.protocol import known_stats
from src.iclr27_phase4c.audit_common import emit_preds, assignment_from_preds
from src.iclr27_phase4d.long_stream import active_bucket, load_stream_cache, stage_of
from src.orbit_iam.compat import build_compat_features
from src.orbit_iam.iam_memory import IamMemory
from src.orbit_mdc.model import ORBITMDCModel, build_birth_features
from src.orbit_mdc.train_mdc import quarantine_influence


def load_mdc_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu")
    sd = ck["state_dict"]
    compat_dim = ck.get("compat_dim", 6)
    # Infer head input widths from the checkpoint itself so any model
    # (MDC C1, M2, CHP) loads correctly; metadata is not always stored.
    gate_dim = (int(sd["gate.net.0.weight"].shape[1])
                if "gate.net.0.weight" in sd else ck.get("gate_dim", 11))
    reuse_dim = (int(sd["reuse.net.0.weight"].shape[1])
                 if "reuse.net.0.weight" in sd else ck.get("reuse_dim", 11))
    birth_dim = (int(sd["birth.net.0.weight"].shape[1])
                 if "birth.net.0.weight" in sd else ck.get("birth_dim", 0))
    model = ORBITMDCModel(dim=768, bottleneck=ck.get("bottleneck", 128),
                          gate_dim=gate_dim,
                          reuse_dim=reuse_dim,
                          hidden=64, use_adapter=True,
                          compat_dim=compat_dim, birth_dim=birth_dim)
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def run_mdc_stream(model, ck, rows, feats, labels, device, gate_thr=0.5,
                   compat_thr=0.45, compat_margin=0.05, birth_thr=0.5,
                   syn_mean=None, proto_feats=None, zs_rels=None,
                   quarantine=None, detailed_memory=False):
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
        "compat_feats", "sim,margin,radius,support,mem,rel").split(",") if f.strip()]
    birth_feats = [f.strip() for f in ck.get(
        "birth_feats", "q_best,q_second,q_margin,support,dispersion,rel,mem").split(",") if f.strip()]
    q_mode = quarantine if quarantine is not None else ck.get("quarantine", "none")
    use_birth = ck.get("birth_dim", 0) > 0
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
        with torch.no_grad():
            gate_logit = float(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
        if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_thr and kid is not None:
            logs.append(_log(r, i, "KNOWN", kid, None, len(mem.novel), 0,
                             best_k, best_n, gate_logit, -1.0, -1.0, -1.0,
                             n, mem, detailed_memory))
            continue
        q_best = -1.0
        q_second = -1.0
        birth_logit = -1.0
        states = {vid: mem.state(vid) for vid in sorted(mem.novel)}
        if P_novel.shape[0]:
            X_rows = []
            for vid in sorted(mem.novel):
                st = states[vid]
                X_rows.append(build_compat_features(
                    z, mem.novel[vid]["proto"], st["radius"], st["support"],
                    st["conf"], len(mem.novel), rel, margin_n, feat_names))
            X = torch.as_tensor(np.asarray(X_rows, dtype=np.float32),
                                device=device)
            with torch.no_grad():
                q = torch.sigmoid(model.compat_forward(X)).cpu().numpy()
            if q_mode != "none":
                infs = np.asarray(
                    [quarantine_influence(states[v], q_mode)
                     for v in sorted(mem.novel)], dtype=np.float32)
                q = q * infs
            if q.shape[0]:
                qorder = np.argsort(q)[::-1]
                q_best = float(q[qorder[0]])
                q_second = float(q[qorder[1]]) if q.shape[0] >= 2 else -1.0
                nid = int(sorted(mem.novel)[int(qorder[0])])
        if use_birth and len(mem.novel) > 0:
            best_st = states[nid] if nid is not None else None
            bf = build_birth_features(
                q_best, q_second,
                math.log1p(best_st["support"]) / math.log1p(300.0)
                if best_st else 0.0,
                best_st["dispersion"] if best_st else 0.5, rel,
                math.log1p(len(mem.novel)) / math.log1p(300.0), birth_feats)
            with torch.no_grad():
                birth_logit = float(model.birth_forward(
                    torch.as_tensor([bf], dtype=torch.float32,
                                    device=device))[0])
            reuse_ok = (float(torch.sigmoid(
                torch.as_tensor(birth_logit))) >= birth_thr)
        else:
            reuse_ok = (q_best >= compat_thr
                        and (len(mem.novel) < 2
                             or q_best - q_second >= compat_margin))
        if reuse_ok and nid is not None:
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            logs.append(_log(r, i, "EXISTING_NOVEL", None, nid, len(mem.novel),
                             support, best_k, best_n, gate_logit, q_best,
                             q_best - q_second, birth_logit, n, mem,
                             detailed_memory))
        else:
            vid = mem.create_novel(z, created_at=i)
            logs.append(_log(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                             0, best_k, best_n, gate_logit, q_best,
                             q_best - q_second, birth_logit, n, mem,
                             detailed_memory))
    return logs


def _log(r, i, action, kid, vid, active, support, bk, bn, gl, qb, qm, bl,
         n, mem, detailed):
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
    }
    if detailed:
        row["memory_mean_support"] = float(np.mean(
            [mem.support(v) for v in sorted(mem.novel)])) if mem.novel else 0.0
        row["memory_mean_dispersion"] = float(np.mean(
            [mem.state(v)["dispersion"] for v in sorted(mem.novel)])) if mem.novel else 0.0
        row["memory_low_conf_count"] = sum(
            1 for v in sorted(mem.novel)
            if mem.support(v) <= 2 or mem.state(v)["dispersion"] > 0.4
            or mem.state(v)["conf"] < 0.2) if mem.novel else 0
    return row


def evaluate_long_mdc(model, ck, device, gate_thr=0.5, compat_thr=0.45,
                      compat_margin=0.05, birth_thr=0.5, quarantine=None):
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    logs = run_mdc_stream(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          compat_margin=compat_margin, birth_thr=birth_thr,
                          syn_mean=syn_mean, quarantine=quarantine)
    attach_gt(logs, gt_rows)
    out = []
    r = summarize(ck.get("variant", "MDC"), logs, gt_rows, "overall")
    if r:
        out.append(r)
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        r = summarize(ck.get("variant", "MDC"), logs, gt_rows, bucket,
                      select=lambda l, b=bucket:
                      active_bucket(l["active_novel_prototypes"]) == b)
        if r:
            out.append(r)
    for stage in ["early", "middle", "late"]:
        r = summarize(ck.get("variant", "MDC"), logs, gt_rows, stage,
                      select=lambda l, s=stage: l["stage"] == s)
        if r:
            out.append(r)
    r_real = summarize(ck.get("variant", "MDC"), logs, gt_rows, "real_only",
                       select=lambda l: l["role"] == "novel"
                       and int(l["class"]) < 1000000)
    r_syn = summarize(ck.get("variant", "MDC"), logs, gt_rows, "synthetic_only",
                      select=lambda l: l["role"] == "novel"
                      and int(l["class"]) >= 1000000)
    return out, logs, r_real, r_syn


def evaluate_official_mdc(model, ck, device, gate_thr=0.5, compat_thr=0.45,
                          compat_margin=0.05, birth_thr=0.5, quarantine=None):
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
    logs = run_mdc_stream(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          compat_margin=compat_margin, birth_thr=birth_thr,
                          proto_feats=train_feats, quarantine=quarantine)
    attach_gt(logs, gt)
    return logs, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--birth_threshold", type=float, default=0.5)
    ap.add_argument("--quarantine", default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--out_prefix", default="candidate")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    model, ck = load_mdc_model(ROOT / args.checkpoint)
    if args.official:
        logs, gt = evaluate_official_mdc(
            model, ck, "cuda", args.gate_threshold, args.compat_threshold,
            args.compat_margin, args.birth_threshold, args.quarantine)
        res, ev = assignment_from_preds(emit_preds(logs), gt)
        mr = mechanism_rates(logs, res["hungarian_assignment"])
        row = {
            "candidate": args.out_prefix,
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
        print(json.dumps(row, indent=1))
        if args.out_csv:
            out = Path(args.out_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            print("wrote", out)
        return
    out, logs, r_real, r_syn = evaluate_long_mdc(
        model, ck, "cuda", args.gate_threshold, args.compat_threshold,
        args.compat_margin, args.birth_threshold, args.quarantine)
    for r in out:
        print(r, flush=True)
    print("REAL_ONLY", r_real, flush=True)
    print("SYN_ONLY", r_syn, flush=True)


if __name__ == "__main__":
    main()
