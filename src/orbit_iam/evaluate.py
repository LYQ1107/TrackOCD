"""Evaluate ORBIT-IAM on long-stream proxies and official validation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import build_known
from src.orbit.protocol import (
    load_gt,
    load_stream,
    load_frame_features,
    load_train_labels,
)
from src.orbit_msr.evaluate import (
    embed_many,
    load_msr_model,
    summarize,
    evaluate_split,
)
from src.orbit_msr.protocol import known_stats, novel_stats
from src.orbit_iam.memory import IamMemory
from src.orbit_iam.train import pair_features
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)


def load_iam_model(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from src.orbit_fc.model import ORBITFCModel
    model = ORBITFCModel(dim=768, bottleneck=ck.get("bottleneck", 128),
                         gate_dim=ck.get("gate_dim", 11),
                         reuse_dim=ck.get("reuse_dim", 11),
                         hidden=64, use_adapter=True,
                         compat_dim=ck.get("compat_dim", 0))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck


def run_stream_iam(model, ck, rows, feats, labels, device, gate_thr=0.5,
                   compat_thr=0.5, q_margin=0.02, conf_thr=0.0,
                   syn_mean=None, proto_feats=None):
    known_classes = sorted(set(labels.values()))
    proto_feats = proto_feats if proto_feats is not None else feats
    protos, radii = build_known(model, proto_feats, labels,
                                set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows], device)
    mem = IamMemory(protos, radii,
                    novel_update_rate=ck.get("novel_update_rate", 0.2))
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    n = len(rows)
    logs = []
    use_conf = bool(ck.get("conf_feature", False))
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
        q_best = q_second = float("nan")
        compat_prob = float("nan")
        states = {}
        if P_novel.shape[0]:
            novel_ids = sorted(mem.novel)
            ns = P_novel @ z
            best_n = float(ns.max())
            order = np.argsort(ns)[::-1]
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
            nid = int(novel_ids[int(order[0])])
            r_n = mem.novel_radii.get(nid, 0.3)
            dist_n = (1.0 - best_n) / max(r_n, 1e-6)
            states = {j: mem.state(j, i) for j in novel_ids}
            feats_ij = pair_features(
                z, P_novel,
                [states[j] for j in novel_ids],
                len(novel_ids), rel, len(feats[r["sample_id"]]), use_conf)
            with torch.no_grad():
                q = model.compat_forward(
                    torch.as_tensor(feats_ij, dtype=torch.float32,
                                    device=device))
            q_order = torch.argsort(q, descending=True).cpu().numpy()
            q_best = float(q[q_order[0]])
            q_second = float(q[q_order[1]]) if q.shape[0] >= 2 else q_best
            compat_prob = float(torch.sigmoid(q[q_order[0]]))
            best_q_vid = int(novel_ids[int(q_order[0])])
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_logit = float(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
        if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_thr \
                and kid is not None:
            logs.append(_log(r, i, "KNOWN", kid, None, len(mem.novel), 0,
                             best_k, best_n, gate_logit, float("nan"),
                             nid, q_best, q_second, compat_prob))
            continue
        rs = novel_stats(z, P_novel, mem.novel_counts, mem.novel_radii,
                         novel_ids=sorted(mem.novel) if mem.novel else None,
                         best_k=best_k, margin_k=_margin(ks),
                         rel=rel, track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel),
                         age_norm=mem.age(nid, i) if nid is not None else 0.0,
                         mem_scale_norm=ck.get("mem_scale_norm", False))
        with torch.no_grad():
            reuse_logit = float(model.reuse_forward(
                torch.as_tensor([rs], dtype=torch.float32, device=device))[0])
        conf_best = (states[best_q_vid]["confidence"]
                     if P_novel.shape[0] else 0.0)
        margin_ok = True
        if P_novel.shape[0] >= 2 and q_best - q_second < q_margin:
            margin_ok = False
        if P_novel.shape[0] and compat_prob >= compat_thr and margin_ok \
                and conf_best >= conf_thr:
            nid = best_q_vid
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            logs.append(_log(r, i, "EXISTING_NOVEL", None, nid, len(mem.novel),
                             support, best_k, best_n, gate_logit, reuse_logit,
                             nid, q_best, q_second, compat_prob))
        else:
            vid = mem.create_novel(z, created_at=i)
            logs.append(_log(r, i, "NEW_NOVEL", None, vid, len(mem.novel), 0,
                             best_k, best_n, gate_logit, reuse_logit,
                             nid, q_best, q_second, compat_prob))
    return logs


def _log(r, i, action, kid, vid, active, support, bk, bn, gl, rl,
         best_nid, q_best, q_second, compat_prob):
    return {
        "sample_id": r["sample_id"], "arrival_index": i,
        "role": r["role"], "class": r["class"],
        "first_occurrence": r["first_occurrence"],
        "predicted_action": action, "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active, "prototype_support": support,
        "best_known_similarity": bk, "best_novel_similarity": bn,
        "gate_logit": gl, "reuse_logit": rl,
        "best_novel_proto_id": best_nid,
        "compat_q_best": q_best, "compat_q_second": q_second,
        "compat_prob": compat_prob,
        "stage": stage_of(i, 5255),
        "bucket": active_bucket(active),
    }


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def prepare_official_rows():
    gt = load_gt("pure")
    rows = load_stream("pure", "main_seed1027")
    gt_by_sid = {g["sample_id"]: g for g in gt}
    seen = set()
    out = []
    for r in rows:
        g = gt_by_sid[r["sample_id"]]
        role = ("known" if g["protocol_role"] in
                ("supported_known", "zero_shot_known") else "novel")
        cls = g["ground_truth_category_id"]
        first = cls not in seen
        seen.add(cls)
        out.append({"sample_id": r["sample_id"], "role": role, "class": cls,
                    "first_occurrence": first})
    return out, gt


def evaluate_long_stream(model, ck, device, gate_thr=0.5, compat_thr=0.5,
                         q_margin=0.02, conf_thr=0.0):
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    labels = load_train_labels()
    logs = run_stream_iam(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          q_margin=q_margin, conf_thr=conf_thr,
                          syn_mean=syn_mean)
    out = [summarize(ck.get("variant", "IAM"), logs, gt_rows, "overall")]
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
    return out, logs


def evaluate_real_only(model, ck, device, gate_thr=0.5, compat_thr=0.5,
                       q_margin=0.02, conf_thr=0.0):
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    sel = [r for r in rows
           if r["role"] == "known" or int(r["class"]) < 1000000]
    labels = load_train_labels()
    logs = run_stream_iam(model, ck, sel, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          q_margin=q_margin, conf_thr=conf_thr,
                          syn_mean=syn_mean)
    gt_by_sid = {g["sample_id"]: g for g in gt_rows}
    gt_sel = [gt_by_sid[r["sample_id"]] for r in sel
              if r["sample_id"] in gt_by_sid]
    out = [summarize(ck.get("variant", "IAM"), logs, gt_sel, "overall")]
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        r = summarize(ck.get("variant", "IAM"), logs, gt_sel, bucket,
                      select=lambda l, b=bucket:
                      active_bucket(l["active_novel_prototypes"]) == b)
        if r:
            out.append(r)
    return out, logs


def evaluate_official(model, ck, device, gate_thr=0.5, compat_thr=0.5,
                      q_margin=0.02, conf_thr=0.0):
    gt = load_gt("pure")
    rows, _ = prepare_official_rows()
    feats = {sid: f[:8] for sid, f in
             load_frame_features("gt_tracks_mean").items()}
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    labels = load_train_labels()
    logs = run_stream_iam(model, ck, rows, feats, labels, device,
                          gate_thr=gate_thr, compat_thr=compat_thr,
                          q_margin=q_margin, conf_thr=conf_thr,
                          proto_feats=train_feats)
    return logs, gt


def _write_rows(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.5)
    ap.add_argument("--q_margin", type=float, default=0.02)
    ap.add_argument("--conf_threshold", type=float, default=0.0)
    ap.add_argument("--mode", default="long",
                    choices=["long", "official", "real_only", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    model, ck = load_iam_model(Path(args.checkpoint), args.device)
    rows_out = []
    if args.mode in ("long", "all"):
        out, logs = evaluate_long_stream(
            model, ck, args.device, args.gate_threshold,
            args.compat_threshold, args.q_margin, args.conf_threshold)
        rows_out += out
        if args.out_csv:
            with open(args.out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
                w.writeheader()
                w.writerows(out)
            Path(args.out_csv + ".logs.json").write_text(
                json.dumps(logs, indent=1, default=str))
    if args.mode in ("real_only", "all"):
        out, logs = evaluate_real_only(
            model, ck, args.device, args.gate_threshold,
            args.compat_threshold, args.q_margin, args.conf_threshold)
        rows_out += out
        if args.out_csv:
            Path(args.out_csv + ".realonly.json").write_text(
                json.dumps(logs, indent=1, default=str))
    if args.mode in ("official", "all"):
        logs, gt = evaluate_official(
            model, ck, args.device, args.gate_threshold,
            args.compat_threshold, args.q_margin, args.conf_threshold)
        res, ev = evaluate_split(logs, gt)
        rows_out.append({k: res[k] for k in
                         ["all_track_acc", "overall_known_acc",
                          "route_aware_novel_acc", "conditional_novel_acc",
                          "novel_routing_recall", "novel_only_nmi",
                          "novel_only_ari", "predicted_novel_count",
                          "novel_count_abs_error"]})
        print(json.dumps(rows_out[-1], indent=1), flush=True)
    for r in rows_out:
        print(r, flush=True)


if __name__ == "__main__":
    main()
