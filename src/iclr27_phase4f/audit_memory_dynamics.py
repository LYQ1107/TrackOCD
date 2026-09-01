"""Phase 4F memory-dynamics audit.

Replays C1 and Phase 4E Candidate A (IAM) on the official and long-stream
proxies with an instrumented causal memory, recording per-step memory state
(size, support distribution, dispersion, confidence, hub count,
known-origin contamination) alongside the decision statistics.  All GT
derived fields are offline audit only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"

from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_stream,
    load_train_labels,
)
from src.orbit_msr.evaluate import (
    embed_many,
    load_msr_model,
    mechanism_rates,
)
from src.orbit_msr.protocol import known_stats, novel_stats
from src.orbit_iam.evaluate_iam import (
    compat_matrix_for_track,
    load_iam_model,
)
from src.orbit_iam.iam_memory import IamMemory
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)


class InstrumentedMemory(IamMemory):
    """IamMemory plus offline per-prototype bookkeeping for audit."""

    def __init__(self, protos, radii, novel_update_rate=0.2):
        super().__init__(protos, radii, novel_update_rate)
        self.origin = {}          # vid -> first assigned GT role
        self.history = defaultdict(list)  # vid -> [(arrival, true_class)]

    def create_novel(self, z, created_at=0, origin_role="novel"):
        vid = super().create_novel(z, created_at)
        self.origin[vid] = origin_role
        return vid

    def note(self, vid, arrival, role, true_class):
        if vid not in self.origin:
            self.origin[vid] = role
        self.history[vid].append((arrival, true_class))

    def offline_stats(self, arrival):
        """Aggregate offline memory-state statistics at a decision time."""
        supports = []
        dispersions = []
        confs = []
        hubs = 0
        for vid in self.novel:
            supports.append(self.support(vid))
            st = self.state(vid)
            dispersions.append(st["dispersion"])
            confs.append(st["conf"])
            hist = self.history.get(vid, [])
            classes = {int(c) for _, c in hist}
            if len(classes) >= 2:
                hubs += 1
        n = len(self.novel)
        known_origin = sum(1 for v in self.novel
                           if self.origin.get(v) == "known")
        return {
            "memory_size": n,
            "mean_support": float(np.mean(supports)) if supports else 0.0,
            "p50_support": float(np.median(supports)) if supports else 0.0,
            "p90_support": float(np.percentile(supports, 90)) if supports else 0.0,
            "mean_dispersion": float(np.mean(dispersions)) if dispersions else 0.0,
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "low_conf_count": sum(1 for s in supports if s <= 2),
            "hub_count_offline": hubs,
            "known_origin_count_offline": known_origin,
            "novel_origin_count_offline": n - known_origin,
        }


def prepare_rows(stream):
    labels = load_train_labels()
    if stream == "long":
        rows, gt_rows, feats, syn_mean = load_stream_cache()
        proto_feats = None
        return rows, gt_rows, feats, syn_mean, proto_feats, labels
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
    train_feats = {sid: f[:8] for sid, f in
                   load_frame_features("train_known_mean").items()}
    return rows, gt, feats, None, train_feats, labels


def replay(method, model, ck, rows, feats, labels, device, stream,
           gate_thr=0.5, reuse_thr=0.45, compat_thr=0.45, compat_margin=0.05,
           proto_feats=None, syn_mean=None):
    from src.orbit.evaluate import build_known
    known_classes = sorted(set(labels.values()))
    proto_feats = proto_feats if proto_feats is not None else feats
    protos, radii = build_known(model, proto_feats, labels,
                                set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                          device)
    mem = InstrumentedMemory(protos, radii,
                             novel_update_rate=ck.get("novel_update_rate", 0.2))
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    feat_names = [f.strip() for f in ck.get(
        "compat_feats", "sim,margin,radius,support,mem,rel").split(",") if f.strip()]
    logs = []
    n = len(rows)
    for i, r in enumerate(rows):
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        margin_k = (float(ks[np.argsort(ks)[::-1][0]] -
                          ks[np.argsort(ks)[::-1][1]])
                    if ks.shape[0] >= 2 else 0.0)
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
        with torch.no_grad():
            gate_logit = float(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
        gate_prob = float(torch.sigmoid(torch.as_tensor(gate_logit)))
        state = mem.offline_stats(i)
        row = {
            "arrival_index": i, "sample_id": r["sample_id"],
            "true_role": r["role"], "true_class": int(r["class"]),
            "first_occurrence": int(bool(r["first_occurrence"])),
            "predicted_known_id": kid,
            "predicted_virtual_novel_id": None,
            "gate_prob": gate_prob, "known_best_sim": best_k,
            "known_margin": margin_k, "best_novel_sim": best_n,
            "novel_margin": margin_n,
            "stage": stage_of(i, n), "memory_bucket": active_bucket(len(mem.novel)),
            **state,
        }
        if gate_prob >= gate_thr and kid is not None:
            row["predicted_action"] = "KNOWN"
            row["compat_best"] = -1.0
            row["compat_margin_score"] = -1.0
            logs.append(row)
            continue
        if method == "c1":
            rs = novel_stats(z, P_novel, mem.novel_counts, mem.novel_radii,
                             novel_ids=sorted(mem.novel) if mem.novel else None,
                             best_k=best_k, margin_k=margin_k, rel=rel,
                             track_len=len(feats[r["sample_id"]]),
                             n_novel=len(mem.novel),
                             age_norm=mem.age(nid, i) if nid is not None else 0.0,
                             mem_scale_norm=ck.get("mem_scale_norm", False))
            with torch.no_grad():
                reuse_logit = float(model.reuse_forward(
                    torch.as_tensor([rs], dtype=torch.float32, device=device))[0])
            reuse_prob = float(torch.sigmoid(torch.as_tensor(reuse_logit)))
            reuse = reuse_prob >= reuse_thr and nid is not None
            row["compat_best"] = reuse_prob
            row["compat_margin_score"] = margin_n
        else:
            states = {vid: mem.state(vid) for vid in sorted(mem.novel)}
            q_best = -1.0
            q_second = -1.0
            if P_novel.shape[0]:
                X = compat_matrix_for_track(
                    z, {vid: mem.novel[vid]["proto"] for vid in sorted(mem.novel)},
                    states, len(mem.novel), rel, margin_n, feat_names)
                with torch.no_grad():
                    q = torch.sigmoid(model.compat_forward(
                        torch.as_tensor(X, dtype=torch.float32, device=device))
                    ).cpu().numpy()
                if q.shape[0]:
                    qorder = np.argsort(q)[::-1]
                    q_best = float(q[qorder[0]])
                    q_second = float(q[qorder[1]]) if q.shape[0] >= 2 else -1.0
                    nid = int(sorted(mem.novel)[int(qorder[0])])
            reuse = (q_best >= compat_thr
                     and (len(mem.novel) < 2 or q_best - q_second >= compat_margin))
            row["compat_best"] = q_best
            row["compat_margin_score"] = q_best - q_second
        if reuse and nid is not None:
            row["predicted_action"] = "EXISTING_NOVEL"
            row["predicted_virtual_novel_id"] = nid
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            mem.note(nid, i, r["role"], int(r["class"]))
        else:
            vid = mem.create_novel(z, created_at=i, origin_role=r["role"])
            row["predicted_action"] = "NEW_NOVEL"
            row["predicted_virtual_novel_id"] = vid
            mem.note(vid, i, r["role"], int(r["class"]))
        logs.append(row)
    return logs


def aggregate(logs, out_dir, stream):
    method = logs[0]["_method"]
    fn = list(logs[0].keys())
    with open(f"{out_dir}/memory_trajectory_{method}_{stream}.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(logs)

    # gate shift by memory bucket
    rows = []
    for bucket in ["0-32", "33-128", "129-256", "257+"]:
        ls = [l for l in logs if l["memory_bucket"] == bucket]
        if not ls:
            continue
        novel = [l for l in ls if l["true_role"] == "novel"]
        probs = [l["gate_prob"] for l in ls]
        n2k = [l for l in ls if l["true_role"] == "novel"
               and l["predicted_action"] == "KNOWN"]
        fb = [l for l in ls if l["true_role"] == "novel"
              and not l["first_occurrence"]
              and l["predicted_action"] == "NEW_NOVEL"]
        we = [l for l in ls if l["true_role"] == "novel"
              and l["predicted_action"] == "EXISTING_NOVEL"]
        rows.append({
            "memory_bucket": bucket, "tracks": len(ls),
            "gate_prob_mean": float(np.mean(probs)),
            "gate_prob_p50": float(np.median(probs)),
            "gate_prob_p90": float(np.percentile(probs, 90)),
            "novel_tracks": len(novel),
            "novel_to_known_rate": len(n2k) / max(len(novel), 1),
            "false_birth_rate": len(fb) / max(len(novel), 1),
            "existing_novel_count": len(we),
        })
    for r in rows:
        r["method"] = method
        r["stream"] = stream
    with open(f"{out_dir}/gate_shift_by_memory_state_{method}_{stream}.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # feedback loop analysis: 200-step windows
    wrows = []
    for start in range(0, len(logs), 200):
        win = logs[start:start + 200]
        n2k = sum(1 for l in win if l["true_role"] == "novel"
                  and l["predicted_action"] == "KNOWN")
        fb = sum(1 for l in win if l["true_role"] == "novel"
                 and not l["first_occurrence"]
                 and l["predicted_action"] == "NEW_NOVEL")
        we = sum(1 for l in win if l["true_role"] == "novel"
                 and l["predicted_action"] == "EXISTING_NOVEL")
        m = win[-1]
        wrows.append({
            "window_start": start,
            "memory_size_end": m["memory_size"],
            "mean_support_end": m["mean_support"],
            "known_origin_count_end": m["known_origin_count_offline"],
            "hub_count_end": m["hub_count_offline"],
            "novel_to_known": n2k,
            "false_birth": fb,
            "wrong_existing": we,
            "novel_tracks": sum(1 for l in win if l["true_role"] == "novel"),
        })
    for r in wrows:
        r["method"] = method
        r["stream"] = stream
    with open(f"{out_dir}/feedback_loop_analysis_{method}_{stream}.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wrows[0].keys()))
        w.writeheader()
        w.writerows(wrows)
    return rows, wrows


def merge_canonical(out_dir):
    """Merge per-method/per-stream audit tables into canonical files."""
    for stem in ["gate_shift_by_memory_state", "feedback_loop_analysis"]:
        all_rows = []
        for method in ["c1", "iam"]:
            for stream in ["official", "long"]:
                p = f"{out_dir}/{stem}_{method}_{stream}.csv"
                try:
                    rows = list(csv.DictReader(open(p)))
                except FileNotFoundError:
                    continue
                all_rows.extend(rows)
        if not all_rows:
            continue
        fn = []
        for r in all_rows:
            for k in r:
                if k not in fn:
                    fn.append(k)
        with open(f"{out_dir}/{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["official", "long"], default="official")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--methods", default="c1,iam")
    args = ap.parse_args()
    rows, gt_rows, feats, syn_mean, proto_feats, labels = prepare_rows(args.stream)
    out_dir = f"{ROOT}/outputs/iclr27_phase4f/audit"
    import pathlib
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",")]
    for method in methods:
        if method == "c1":
            model, ck = load_msr_model(f"{ROOT}/runs/orbit_msr/msr_nr2/model.pth",
                                       args.device)
            logs = replay("c1", model, ck, rows, feats, labels, args.device,
                          args.stream, gate_thr=0.5, reuse_thr=0.45,
                          proto_feats=proto_feats, syn_mean=syn_mean)
        elif method in ("iam", "m2"):
            ckpt = (f"{ROOT}/runs/orbit_iam/iam_i2_v3/model.pth" if method == "iam"
                    else f"{ROOT}/runs/orbit_mdc/mdc_m2/model.pth")
            model, ck = load_iam_model(ckpt, args.device)
            logs = replay(method, model, ck, rows, feats, labels, args.device,
                          args.stream, gate_thr=0.5, compat_thr=0.45,
                          compat_margin=0.05, proto_feats=proto_feats,
                          syn_mean=syn_mean)
        else:
            raise ValueError(method)
        for l in logs:
            l["_method"] = method
        aggregate(logs, out_dir, args.stream)
        dump_prototypes_from_logs(logs, out_dir, method, args.stream)
        print(f"{method.upper()} done", len(logs), flush=True)
    merge_canonical(out_dir)


def dump_prototypes_from_logs(logs, out_dir, method, stream):
    """Build per-prototype rows from trajectory logs (no memory object)."""
    assigned = defaultdict(list)
    for l in logs:
        if l["predicted_action"] in ("EXISTING_NOVEL", "NEW_NOVEL"):
            vid = l.get("predicted_virtual_novel_id")
            if vid is None:
                # NEW_NOVEL rows carry the new vid in predicted_virtual_novel_id
                vid = l.get("predicted_virtual_novel_id")
            if vid is not None:
                assigned[int(vid)].append(l)
    # map vid -> origin from first row's true_role
    proto_rows = []
    for vid, ls in assigned.items():
        ls_sorted = sorted(ls, key=lambda x: x["arrival_index"])
        origin_role = ls_sorted[0]["true_role"] if ls_sorted else "?"
        novel_ls = [l for l in ls if l["true_role"] == "novel"]
        classes = Counter(int(l["true_class"]) for l in novel_ls)
        primary = classes.most_common(1)[0][0] if classes else None
        wrong_existing = sum(1 for l in ls
                             if l["predicted_action"] == "EXISTING_NOVEL"
                             and primary is not None
                             and int(l["true_class"]) != primary)
        proto_rows.append({
            "method": method, "stream": stream, "virtual_id": vid,
            "origin_role": origin_role,
            "support": len(novel_ls),
            "dispersion": ls_sorted[-1].get("mean_dispersion"),
            "lifetime": ls_sorted[-1]["arrival_index"] - ls_sorted[0]["arrival_index"],
            "future_assignments": len(ls) - 1,
            "absorbed_novel_classes": len(classes),
            "distinct_classes_total": len(set(int(l["true_class"]) for l in ls)),
            "hub_risk": int(len(classes) >= 2),
            "wrong_existing_contribution": wrong_existing,
            "first_merge_count": sum(1 for l in ls if l["first_occurrence"]
                                     and l["predicted_action"] == "EXISTING_NOVEL"),
        })
    fn = list(proto_rows[0].keys())
    with open(f"{out_dir}/known_origin_prototype_analysis_{method}_{stream}.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        w.writerows(proto_rows)
    return proto_rows


if __name__ == "__main__":
    main()
