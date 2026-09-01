"""Phase 4E identity-error audit.

Replays ORBIT-MSR C1/C2 on the long-stream proxy and official Pure Full
seed1027 with the exact causal decision loop used by the frozen evaluator,
while additionally logging per-record similarity/margin/radius/support/age/
dispersion/memory-scale features and per-prototype confidence statistics.

All GT usage is offline audit only; the replay decision path is unchanged
from ``src.orbit_msr.evaluate.run_stream_msr`` (verified against the frozen
official/proxy result files).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.orbit.evaluate import build_known
from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_stream,
    load_train_labels,
)
from src.orbit_msr.evaluate import (
    embed_many,
    evaluate_split,
    load_msr_model,
)
from src.orbit_msr.protocol import known_stats, novel_stats, stats_to_tensor
from src.orbit_fc.causal_memory import CausalNovelMemory
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)


def _log_row(r, i, action, kid, vid, active, support, bk, bn, gl, rl,
             extra=None):
    row = {
        "sample_id": r["sample_id"],
        "track_id": r.get("track_id", r["sample_id"]),
        "video_id": r.get("video_id", ""),
        "domain": r.get("_domain", ""),
        "arrival_index": i,
        "role": r["role"],
        "true_class": r["class"],
        "first_occurrence": r["first_occurrence"],
        "predicted_action": action,
        "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active,
        "prototype_support": support,
        "best_known_similarity": bk,
        "best_novel_similarity": bn,
        "gate_logit": gl,
        "reuse_logit": rl,
        "stage": stage_of(i, r.get("_n", 5255)),
        "bucket": active_bucket(active),
    }
    if extra:
        row.update(extra)
    return row


def run_audit_stream(model, ck, rows, feats, labels, device, gate_thr=0.5,
                     reuse_thr=0.5, syn_mean=None, proto_feats=None):
    """Causal replay with rich per-record and per-prototype logging.

    The decision sequence is a line-for-line copy of
    ``src.orbit_msr.evaluate.run_stream_msr`` (verified by metric
    self-checks); only bookkeeping is added.
    """
    known_classes = sorted(set(labels.values()))
    proto_feats = proto_feats if proto_feats is not None else feats
    protos, radii = build_known(model, proto_feats, labels,
                                set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                          device)
    mem = CausalNovelMemory(protos, radii,
                            novel_update_rate=ck.get("novel_update_rate", 0.2))
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    n = len(rows)
    logs = []
    proto_state = {}

    for i, r in enumerate(rows):
        r = dict(r)
        r["_n"] = n
        z = zs[r["sample_id"]]
        rel = rels[r["sample_id"]]
        ks = P_known @ z
        kid = int(known_ids[int(np.argmax(ks))]) if ks.shape[0] else None
        best_k = float(ks.max()) if ks.shape[0] else -1.0
        k_order = np.argsort(ks)[::-1] if ks.shape[0] else np.array([], dtype=int)
        second_k = float(ks[k_order[1]]) if ks.shape[0] >= 2 else best_k
        margin_k = best_k - second_k if ks.shape[0] >= 2 else 0.0

        P_novel = np.stack([mem.novel[c]["proto"] for c in sorted(mem.novel)]
                           ).astype(np.float32) if mem.novel else np.empty(
                               (0, 768), dtype=np.float32)
        nid = None
        best_n = second_n = -1.0
        margin_n = 0.0
        dist_n = 1.0
        ns_ids = []
        ns_sims = []
        best_nid = None
        second_nid = None
        radius_n = float("nan")
        support_n = 0
        age_n = 0
        dispersion_n = float("nan")
        if P_novel.shape[0]:
            ns = P_novel @ z
            order = np.argsort(ns)[::-1]
            best_n = float(ns[order[0]])
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
            novel_ids = sorted(mem.novel)
            ns_ids = [novel_ids[int(j)] for j in order]
            ns_sims = [float(ns[j]) for j in order]
            best_nid = int(novel_ids[int(order[0])])
            second_nid = int(novel_ids[int(order[1])]) if ns.shape[0] >= 2 else None
            nid = best_nid
            radius_n = float(mem.novel_radii.get(nid, 0.3))
            dist_n = (1.0 - best_n) / max(radius_n, 1e-6)
            support_n = mem.support(nid)
            age_n = mem.age(nid, i)
            st = proto_state.get(nid)
            if st and st["dispersion_count"] > 0:
                dispersion_n = st["dispersion_sum"] / st["dispersion_count"]
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_logit = float(model.gate_forward(
                stats_to_tensor(gs, device))[0])
        prob_gate = float(torch.sigmoid(torch.as_tensor(gate_logit)))
        if prob_gate >= gate_thr and kid is not None:
            logs.append(_log_row(
                r, i, "KNOWN", kid, None, len(mem.novel), 0,
                best_k, best_n, gate_logit, float("nan"),
                {"second_known_similarity": second_k,
                 "known_margin": margin_k,
                 "best_novel_proto_id": best_nid,
                 "second_novel_proto_id": second_nid,
                 "second_novel_similarity": second_n,
                 "novel_margin": margin_n,
                 "novel_radius": radius_n,
                 "prototype_dispersion": dispersion_n,
                 "prob_gate": prob_gate, "prob_reuse": float("nan"),
                 "track_length": len(feats[r["sample_id"]]),
                 "feature_reliability": rel,
                 "ns_ids": json.dumps(ns_ids),
                 "ns_sims": json.dumps(ns_sims)}))
            continue
        rs = novel_stats(z, P_novel, mem.novel_counts, mem.novel_radii,
                         novel_ids=sorted(mem.novel) if mem.novel else None,
                         best_k=best_k, margin_k=margin_k,
                         rel=rel, track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel),
                         age_norm=mem.age(nid, i) if nid is not None else 0.0,
                         mem_scale_norm=ck.get("mem_scale_norm", False))
        with torch.no_grad():
            reuse_logit = float(model.reuse_forward(
                stats_to_tensor(rs, device))[0])
        prob_reuse = float(torch.sigmoid(torch.as_tensor(reuse_logit)))
        extra = {
            "second_known_similarity": second_k,
            "known_margin": margin_k,
            "best_novel_proto_id": best_nid,
            "second_novel_proto_id": second_nid,
            "second_novel_similarity": second_n,
            "novel_margin": margin_n,
            "novel_radius": radius_n,
            "novel_dist_norm": dist_n,
            "prototype_dispersion": dispersion_n,
            "prototype_age": age_n,
            "prob_gate": prob_gate, "prob_reuse": prob_reuse,
            "track_length": len(feats[r["sample_id"]]),
            "feature_reliability": rel,
            "ns_ids": json.dumps(ns_ids),
            "ns_sims": json.dumps(ns_sims),
            "true_proto_exists": False,
            "true_proto_rank": None,
            "true_proto_sim": None,
            "created_new_prototype": False,
        }
        if prob_reuse >= reuse_thr and nid is not None:
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False))
            st = proto_state.setdefault(nid, {
                "created_at": i, "support": 0, "dispersion_sum": 0.0,
                "dispersion_count": 0, "margin_sum": 0.0,
                "margin_count": 0, "min_margin": float("inf"),
                "low_margin_count": 0, "recent_margins": [],
                "first_assignment_idx": i, "last_assignment_idx": i,
            })
            st["support"] += 1
            st["dispersion_sum"] += max(1.0 - cos_to_center, 0.0)
            st["dispersion_count"] += 1
            st["margin_sum"] += margin_n
            st["margin_count"] += 1
            st["min_margin"] = min(st["min_margin"], margin_n)
            if margin_n < 0.05:
                st["low_margin_count"] += 1
            st["recent_margins"].append(margin_n)
            if len(st["recent_margins"]) > 12:
                st["recent_margins"] = st["recent_margins"][-12:]
            st["last_assignment_idx"] = i
            logs.append(_log_row(
                r, i, "EXISTING_NOVEL", None, nid, len(mem.novel), support,
                best_k, best_n, gate_logit, reuse_logit, extra))
        else:
            vid = mem.create_novel(z, created_at=i)
            proto_state[vid] = {
                "created_at": i, "support": 1, "dispersion_sum": 0.0,
                "dispersion_count": 0, "margin_sum": 0.0, "margin_count": 0,
                "min_margin": float("inf"), "low_margin_count": 0,
                "recent_margins": [], "first_assignment_idx": i,
                "last_assignment_idx": i,
            }
            extra["created_new_prototype"] = True
            logs.append(_log_row(
                r, i, "NEW_NOVEL", None, vid, len(mem.novel), 0,
                best_k, best_n, gate_logit, reuse_logit, extra))
    return logs, mem, proto_state


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
        path = (r.get("image_paths") or [""])
        domain = path[0].split("/")[1] if path and "/" in path[0] else "?"
        out.append({"sample_id": r["sample_id"], "track_id": r.get("track_id"),
                    "video_id": r.get("video_id"), "_domain": domain,
                    "role": role, "class": cls, "first_occurrence": first})
    return out, gt


def prepare_long_rows():
    rows, gt_rows, feats, syn_mean = load_stream_cache()
    out = []
    for r in rows:
        cls = int(r["class"])
        out.append({"sample_id": r["sample_id"],
                    "track_id": r["sample_id"], "video_id": "",
                    "_domain": "synthetic" if cls >= 1000000 else "real",
                    "role": r["role"], "class": r["class"],
                    "first_occurrence": r["first_occurrence"]})
    return out, gt_rows, feats, syn_mean


def postprocess(logs, gt_rows):
    """Attach offline GT-derived fields (audit only) and compute per-prototype
    confidence / purity / hubness statistics."""
    preds = []
    for l in logs:
        if l["predicted_action"] == "KNOWN" and l["predicted_known_id"] is not None:
            preds.append({"sample_id": l["sample_id"],
                          "stream_order": l["arrival_index"],
                          "prediction_type": "known",
                          "semantic_category_id": int(l["predicted_known_id"])})
        elif l["predicted_virtual_novel_id"] is not None:
            preds.append({"sample_id": l["sample_id"],
                          "stream_order": l["arrival_index"],
                          "prediction_type": "novel",
                          "virtual_category_id": int(l["predicted_virtual_novel_id"])})
        else:
            preds.append({"sample_id": l["sample_id"],
                          "stream_order": l["arrival_index"],
                          "prediction_type": "unresolved"})
    res, _ = evaluate_split(logs, gt_rows)
    hungarian = {int(k): int(v) for k, v in res["hungarian_assignment"].items()}

    created_by_vid = {}
    assigned = defaultdict(list)
    for l in logs:
        if l["predicted_virtual_novel_id"] is not None:
            vid = int(l["predicted_virtual_novel_id"])
            created_by_vid.setdefault(vid, l["arrival_index"])
            assigned[vid].append(l)

    proto_meta = {}
    for vid, ls in assigned.items():
        true_classes = [int(l["true_class"]) for l in ls
                        if l["role"] == "novel"]
        majority = Counter(true_classes).most_common(1)[0][0] if true_classes else None
        primary = hungarian.get(vid, majority)
        n = max(len(true_classes), 1)
        purity = (sum(1 for c in true_classes if c == primary) / n
                  if primary is not None else 0.0)
        attracted = set(true_classes)
        wrong_sims = [l["best_novel_similarity"] for l in ls
                      if l["role"] == "novel"
                      and int(l["true_class"]) != primary]
        hub_score = float(np.mean(wrong_sims)) if wrong_sims else 0.0
        n_first_merge = sum(1 for l in ls
                            if l["first_occurrence"]
                            and l["predicted_action"] == "EXISTING_NOVEL")
        proto_meta[vid] = {
            "primary_class": primary,
            "majority_class": majority,
            "purity": purity,
            "attracted_classes": attracted,
            "hub_score": hub_score,
            "n_first_merge": n_first_merge,
        }

    for l in logs:
        vid = l["predicted_virtual_novel_id"]
        if vid is not None:
            vid = int(vid)
            pm = proto_meta.get(vid, {})
            l["proto_primary_class"] = pm.get("primary_class")
            l["proto_majority_class"] = pm.get("majority_class")
            l["proto_purity"] = pm.get("purity")
            l["proto_hub_score"] = pm.get("hub_score")
            l["proto_attracted_classes"] = len(pm.get("attracted_classes", set()))
            l["proto_n_first_merge"] = pm.get("n_first_merge", 0)
            l["proto_created_at"] = created_by_vid.get(vid)
        else:
            l["proto_primary_class"] = None
            l["proto_purity"] = float("nan")
            l["proto_hub_score"] = float("nan")
            l["proto_attracted_classes"] = 0
            l["proto_n_first_merge"] = 0
            l["proto_created_at"] = None
        if l["role"] == "novel" and vid is not None:
            tc = int(l["true_class"])
            tv = hungarian.get(tc)
            l["true_proto_exists"] = (
                tv is not None and created_by_vid.get(tv, 10**9) < l["arrival_index"])
            l["true_proto_vid"] = tv
            if tv is not None:
                ns_ids = json.loads(l.get("ns_ids", "[]"))
                ns_sims = json.loads(l.get("ns_sims", "[]"))
                if tv in ns_ids:
                    l["true_proto_rank"] = ns_ids.index(tv) + 1
                    l["true_proto_sim"] = ns_sims[ns_ids.index(tv)]
                else:
                    l["true_proto_rank"] = None
                    l["true_proto_sim"] = None
            else:
                l["true_proto_rank"] = None
                l["true_proto_sim"] = None
        else:
            l["true_proto_vid"] = None
            l["true_proto_rank"] = None
            l["true_proto_sim"] = None
    return res, hungarian, proto_meta, created_by_vid


def classify_wrong(l):
    """Exclusive error-source priority classification (descriptive flags)."""
    if not l["true_proto_exists"]:
        return "no_true_proto"
    if l.get("true_proto_rank") is not None and l["true_proto_rank"] >= 2:
        return "true_proto_not_selected"
    if l["proto_attracted_classes"] >= 3:
        return "hub_prototype"
    if l["prototype_support"] >= 8 and l["novel_margin"] < 0.05:
        return "support_reward"
    if l["novel_radius"] >= 0.5:
        return "wide_radius"
    if l["proto_purity"] is not None and l["proto_purity"] < 0.8:
        return "polluted_prototype"
    if l["novel_margin"] < 0.02:
        return "low_margin_forced"
    return "other"


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    choices=["C1", "C2"])
    ap.add_argument("--streams", default="long,official")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--reuse_threshold", type=float, default=0.45)
    args = ap.parse_args()

    ckpt_path = (ROOT / "runs/orbit_msr/msr_nr2/model.pth" if args.checkpoint == "C1"
                 else ROOT / "runs/orbit_msr/msr_c2/model.pth")
    device = args.device
    model, ck = load_msr_model(ckpt_path, device=device)
    labels = load_train_labels()
    out_dir = ROOT / "outputs" / "iclr27_phase4e" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_wrong = []
    all_first = []
    all_hub = []
    all_conf = []
    all_corr = []
    for stream in args.streams.split(","):
        if stream == "long":
            rows, gt_rows, feats, syn_mean = prepare_long_rows()
            proto_feats = None
        elif stream == "official":
            rows, gt_rows = prepare_official_rows()
            feats = {sid: f[:8] for sid, f in
                     load_frame_features("gt_tracks_mean").items()}
            train_feats = {sid: f[:8] for sid, f in
                           load_frame_features("train_known_mean").items()}
            proto_feats = train_feats
            syn_mean = None
        else:
            raise ValueError(stream)
        logs, mem, proto_state = run_audit_stream(
            model, ck, rows, feats, labels, device,
            gate_thr=args.gate_threshold, reuse_thr=args.reuse_threshold,
            syn_mean=syn_mean, proto_feats=proto_feats)
        res, hungarian, proto_meta, created_by_vid = postprocess(logs, gt_rows)

        # self-check against frozen results
        got = {k: res[k] for k in
               ["overall_known_acc", "route_aware_novel_acc",
                "conditional_novel_acc", "novel_routing_recall",
                "novel_only_nmi", "novel_only_ari", "novel_count_abs_error"]}
        if stream == "official":
            frozen = json.loads(
                (ROOT / f"runs/orbit_msr/candidate_{1 if args.checkpoint=='C1' else 2}_seed1027.json"
                 ).read_text())
            expect = {k: frozen[k] for k in got}
        else:
            comp = list(csv.DictReader(open(
                ROOT / "outputs/orbit_msr/meta_dev/long_stream_all_configs.csv")))
            row = [r for r in comp if r["ckpt"] ==
                   ("NR2" if args.checkpoint == "C1" else "C2")
                   and r["gate_thr"] == str(args.gate_threshold)
                   and r["reuse_thr"] == str(args.reuse_threshold)
                   and r["scope"] == "overall"][0]
            expect = {"overall_known_acc": float(row["known_acc"]),
                      "route_aware_novel_acc": float(row["rn_acc"]),
                      "conditional_novel_acc": float(row["cond_novel_acc"]),
                      "novel_routing_recall": float(row["routing_recall"]),
                      "novel_only_nmi": float(row["nmi"]),
                      "novel_only_ari": float(row["ari"]),
                      "novel_count_abs_error": float(row["count_error"])}
        diffs = {k: abs(got[k] - expect[k]) for k in got}
        maxdiff = max(diffs.values())
        print(f"[{args.checkpoint}/{stream}] self-check maxdiff={maxdiff:.6f}",
              flush=True)
        if maxdiff > 1e-4:
            raise SystemExit(f"SELF-CHECK FAILED {args.checkpoint}/{stream}: {diffs}")

        prefix = f"{args.checkpoint}_{stream}"
        write_csv(out_dir / f"per_track_{prefix}.csv", logs)

        for l in logs:
            if l["role"] != "novel" or l["predicted_action"] != "EXISTING_NOVEL":
                continue
            vid = int(l["predicted_virtual_novel_id"])
            wrong = int(l["true_class"]) != l["proto_primary_class"]
            base = {k: l.get(k) for k in [
                "sample_id", "track_id", "video_id", "domain", "arrival_index",
                "first_occurrence", "active_novel_prototypes", "bucket", "stage",
                "prototype_support", "prototype_age", "best_known_similarity",
                "second_known_similarity", "known_margin",
                "best_novel_similarity", "second_novel_similarity",
                "novel_margin", "novel_radius", "novel_dist_norm",
                "prototype_dispersion", "prob_gate", "prob_reuse",
                "track_length", "feature_reliability",
                "true_proto_exists", "true_proto_rank", "true_proto_sim",
                "proto_purity", "proto_hub_score", "proto_attracted_classes",
                "proto_created_at"]}
            base.update({
                "model": args.checkpoint, "stream": stream,
                "true_class": l["true_class"],
                "assigned_virtual_id": vid,
                "assigned_proto_primary_class": l["proto_primary_class"],
                "wrong_existing": wrong,
                "first_merge": bool(l["first_occurrence"]),
                "error_source": classify_wrong(l) if wrong else "",
            })
            if wrong:
                all_wrong.append(base)
            if l["first_occurrence"]:
                all_first.append(base)

        for vid, pm in proto_meta.items():
            st = proto_state.get(vid, {})
            support = st.get("support", 0)
            dispersion = (st["dispersion_sum"] / st["dispersion_count"]
                          if st.get("dispersion_count", 0) else float("nan"))
            mean_margin = (st["margin_sum"] / st["margin_count"]
                           if st.get("margin_count", 0) else float("nan"))
            recent = st.get("recent_margins", [])
            stability = (float(np.std(recent)) if len(recent) >= 2 else float("nan"))
            n_low = st.get("low_margin_count", 0)
            n_wrong = sum(1 for l in logs
                          if l["predicted_virtual_novel_id"] == vid
                          and l["role"] == "novel"
                          and l["proto_primary_class"] is not None
                          and int(l["true_class"]) != l["proto_primary_class"])
            n_first = pm.get("n_first_merge", 0)
            all_conf.append({
                "model": args.checkpoint, "stream": stream,
                "virtual_id": vid,
                "primary_class": pm.get("primary_class"),
                "created_at": st.get("created_at"),
                "support": support,
                "final_radius": mem.novel_radii.get(vid),
                "dispersion": dispersion,
                "mean_margin": mean_margin,
                "min_margin": st.get("min_margin"),
                "low_margin_count": n_low,
                "recent_stability": stability,
                "age": (st.get("last_assignment_idx", st.get("created_at"))
                        - st.get("created_at", 0)),
                "purity": pm.get("purity"),
                "attracted_classes": len(pm.get("attracted_classes", set())),
                "hub_score": pm.get("hub_score"),
                "n_wrong_assignments": n_wrong,
                "n_false_merges": n_first,
            })

    write_csv(out_dir / "wrong_existing_assignments.csv", all_wrong)
    write_csv(out_dir / "first_occurrence_false_merge.csv", all_first)

    def agg(rows, keyfn, fields):
        out = []
        groups = defaultdict(list)
        for r in rows:
            groups[keyfn(r)].append(r)
        for k in sorted(groups, key=str):
            g = groups[k]
            row = {"group": k, "n": len(g)}
            for f in fields:
                vals = [r[f] for r in g if r.get(f) is not None
                        and not (isinstance(r[f], float) and math.isnan(r[f]))]
                row[f + "_mean"] = float(np.mean(vals)) if vals else ""
                row[f + "_median"] = float(np.median(vals)) if vals else ""
            row["error_sources"] = "; ".join(
                f"{c}:{n}" for c, n in Counter(
                    r["error_source"] for r in g).most_common())
            out.append(row)
        return out

    ms_rows = agg(all_wrong, lambda r: f"{r['model']}|{r['stream']}|{r['bucket']}",
                  ["best_novel_similarity", "novel_margin", "prototype_support",
                   "novel_radius", "proto_purity"])
    for r in ms_rows:
        r["model"], r["stream"], r["bucket"] = r["group"].split("|")
        del r["group"]
    write_csv(out_dir / "wrong_existing_by_memory_scale.csv", ms_rows)

    def support_bucket(s):
        if s <= 0:
            return "0"
        if s <= 2:
            return "1-2"
        if s <= 7:
            return "3-7"
        if s <= 15:
            return "8-15"
        if s <= 31:
            return "16-31"
        return "32+"

    sp_rows = agg(all_wrong, lambda r: f"{r['model']}|{r['stream']}|{support_bucket(r['prototype_support'])}",
                  ["best_novel_similarity", "novel_margin", "proto_purity"])
    for r in sp_rows:
        r["model"], r["stream"], r["support_bucket"] = r["group"].split("|")
        del r["group"]
    write_csv(out_dir / "wrong_existing_by_prototype_support.csv", sp_rows)

    def margin_bucket(m):
        if m < 0.01:
            return "<0.01"
        if m < 0.03:
            return "0.01-0.03"
        if m < 0.05:
            return "0.03-0.05"
        if m < 0.1:
            return "0.05-0.1"
        return ">=0.1"

    mg_rows = agg(all_wrong, lambda r: f"{r['model']}|{r['stream']}|{margin_bucket(r['novel_margin'])}",
                  ["best_novel_similarity", "novel_margin", "proto_purity"])
    for r in mg_rows:
        r["model"], r["stream"], r["margin_bucket"] = r["group"].split("|")
        del r["group"]
    write_csv(out_dir / "wrong_existing_by_margin.csv", mg_rows)

    hub_rows = agg(all_first, lambda r: f"{r['model']}|{r['stream']}|{r['assigned_virtual_id']}",
                   ["best_novel_similarity", "novel_margin", "proto_purity",
                    "proto_hub_score"])
    for r in hub_rows:
        r["model"], r["stream"], r["prototype_id"] = r["group"].split("|")
        del r["group"]
    write_csv(out_dir / "false_merge_by_hub_prototype.csv", hub_rows)

    write_csv(out_dir / "prototype_confidence_analysis.csv", all_conf)

    corr_rows = []
    conf_by_cfg = defaultdict(list)
    for c in all_conf:
        conf_by_cfg[(c["model"], c["stream"])].append(c)
    feats = ["support", "dispersion", "mean_margin", "min_margin",
             "low_margin_count", "recent_stability", "age", "hub_score"]
    targets = ["purity", "n_wrong_assignments", "n_false_merges"]
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        pearsonr = spearmanr = None
    for (model, stream), cs in conf_by_cfg.items():
        for f in feats:
            for t in targets:
                xs = [c[f] for c in cs if c[f] is not None
                      and not (isinstance(c[f], float) and math.isnan(c[f]))]
                ys = [c[t] for c in cs if c[f] is not None
                      and not (isinstance(c[f], float) and math.isnan(c[f]))]
                if len(xs) >= 5 and len(set(xs)) > 1 and len(set(ys)) > 1:
                    pr = pearsonr(xs, ys) if pearsonr else (float("nan"), 1.0)
                    sr = spearmanr(xs, ys) if spearmanr else (float("nan"), 1.0)
                    corr_rows.append({"model": model, "stream": stream,
                                      "feature": f, "target": t,
                                      "pearson_r": pr[0], "pearson_p": pr[1],
                                      "spearman_r": sr[0], "spearman_p": sr[1],
                                      "n": len(xs)})
    write_csv(out_dir / "confidence_purity_correlation.csv", corr_rows)

    summary = {
        "checkpoint": args.checkpoint,
        "gate_threshold": args.gate_threshold,
        "reuse_threshold": args.reuse_threshold,
        "wrong_existing_rows": len(all_wrong),
        "first_merge_rows": len(all_first),
        "prototype_rows": len(all_conf),
    }
    (out_dir / f"audit_summary_{args.checkpoint}.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
