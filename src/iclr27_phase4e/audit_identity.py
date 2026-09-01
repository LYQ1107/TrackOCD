"""Phase 4E identity error audit.

Replays frozen ORBIT-MSR C1/C2 on the long-stream proxy and the official
Pure Full seed1027 stream with an instrumented causal memory.  For every
decision we record the state of the candidate novel prototype (radius,
support, dispersion, age, margin history, offline primary class and purity)
so the wrong-existing / first-occurrence / confidence audits can attribute
errors to concrete mechanisms.  GT labels are used only for offline audit
columns (explicitly marked), never as inputs to the decision policy.

The multimodality audit embeds real novel classes with the frozen C1
adapter and fits 1/2/3 centers per class (offline diagnosis only).
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

from src.orbit.protocol import (
    load_frame_features,
    load_gt,
    load_stream,
    load_train_labels,
)
from src.orbit_fc.causal_memory import CausalNovelMemory
from src.orbit_msr.evaluate import (
    embed_many,
    load_msr_model,
    mechanism_rates,
)
from src.orbit_msr.protocol import known_stats, novel_stats
from src.iclr27_phase4c.audit_common import assignment_from_preds, emit_preds
from src.iclr27_phase4d.long_stream import (
    active_bucket,
    load_stream_cache,
    stage_of,
)


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


class AuditMemory(CausalNovelMemory):
    """CausalNovelMemory plus per-prototype audit bookkeeping.

    All GT-derived fields are audit-only.  The decision policy reads exactly
    the same fields as run_stream_msr (protos, counts, radii, age).
    """

    def __init__(self, known_protos, known_radii=None, novel_update_rate=0.2,
                 radius_percentile=50.0, radius_update_rate=0.2):
        super().__init__(known_protos, known_radii, novel_update_rate,
                         radius_percentile, radius_update_rate)
        self.meta = defaultdict(lambda: {
            "created_at": None,
            "dispersion": 0.0,
            "disp_n": 0,
            "mean_margin": 0.0,
            "min_margin": 1.0,
            "low_margin_count": 0,
            "margin_n": 0,
            "history": [],
        })

    def create_novel(self, z, created_at=0):
        vid = super().create_novel(z, created_at)
        self.meta[vid]["created_at"] = created_at
        return vid

    def update_novel(self, vid, z, cos_to_center=None, update_radius=False,
                     margin=None):
        m = self.meta[vid]
        if cos_to_center is None:
            cos_to_center = float(np.dot(self.novel[vid]["proto"], z))
        d = max(1.0 - cos_to_center, 0.0)
        m["disp_n"] += 1
        m["dispersion"] = ((m["disp_n"] - 1) * m["dispersion"] + d) / m["disp_n"]
        if margin is not None:
            n = m["margin_n"] + 1
            m["mean_margin"] = (m["mean_margin"] * m["margin_n"] + margin) / n
            m["margin_n"] = n
            m["min_margin"] = min(m["min_margin"], margin)
            if margin < 0.02:
                m["low_margin_count"] += 1
        super().update_novel(vid, z, cos_to_center=cos_to_center,
                             update_radius=update_radius)

    def note_assignment(self, vid, true_class, arrival):
        self.meta[vid]["history"].append((int(arrival), int(true_class)))

    def state(self, vid, arrival):
        """Decision-time state of prototype vid (before current assignment)."""
        m = self.meta[vid]
        hist = m["history"]
        cnt = Counter(h for _, h in hist)
        primary = max(cnt, key=cnt.get) if cnt else None
        purity = (max(cnt.values()) / len(hist)) if hist else 1.0
        distinct = len(cnt)
        recent = hist[-10:]
        if recent:
            rcnt = Counter(h for _, h in recent)
            rp = max(rcnt, key=rcnt.get)
            stability = sum(1 for _, h in recent if h == rp) / len(recent)
        else:
            stability = 1.0
        low_margin_rate = m["low_margin_count"] / max(m["margin_n"], 1)
        conf_legal = (math.log1p(self.support(vid)) / math.log1p(20.0)
                      * math.exp(-m["dispersion"] / 0.3)
                      * (1.0 - low_margin_rate) * stability)
        return {
            "radius": float(self.novel_radii.get(vid, 0.3)),
            "support": self.support(vid),
            "dispersion": float(m["dispersion"]),
            "age": max(arrival - (m["created_at"] or 0), 0),
            "mean_margin": float(m["mean_margin"]),
            "min_margin": float(m["min_margin"]),
            "low_margin_count": int(m["low_margin_count"]),
            "assignment_consistency": float(purity),
            "primary_class": primary,
            "distinct_classes": distinct,
            "recent_stability": float(stability),
            "conf_legal": float(conf_legal),
        }


def _domain_of(r):
    paths = r.get("image_paths") or []
    if paths:
        parts = str(paths[0]).split("/")
        return parts[1] if len(parts) > 2 else parts[0]
    return "synthetic" if str(r.get("sample_id", "")).startswith("syn_") else "real"


def _stage_n(stream):
    return 5255 if stream == "official" else 5255


def run_audit_stream(model, ck, rows, feats, labels, device, stream,
                     gate_thr=0.5, reuse_thr=0.45, syn_mean=None,
                     proto_feats=None):
    known_classes = sorted(set(labels.values()))
    proto_feats = proto_feats if proto_feats is not None else feats
    from src.orbit.evaluate import build_known
    protos, radii = build_known(model, proto_feats, labels,
                                set(known_classes), device)
    zs, rels = embed_many(model, feats, [r["sample_id"] for r in rows],
                          device)
    mem = AuditMemory(protos, radii,
                      novel_update_rate=ck.get("novel_update_rate", 0.2))
    P_known = np.stack([protos[c] for c in sorted(protos)]).astype(np.float32)
    known_ids = sorted(protos)
    logs = []
    n = len(rows)
    for i, r in enumerate(rows):
        r.setdefault("_track_len", len(feats[r["sample_id"]]))
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
        best_state = {}
        true_states = []
        if P_novel.shape[0]:
            ns = P_novel @ z
            best_n = float(ns.max())
            order = np.argsort(ns)[::-1]
            second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
            margin_n = best_n - second_n
            nid = int(sorted(mem.novel)[int(order[0])])
            r_n = mem.novel_radii.get(nid, 0.3)
            dist_n = (1.0 - best_n) / max(r_n, 1e-6)
            best_state = mem.state(nid, i)
            # offline: does a prototype whose primary class equals the true
            # class already exist, and how similar is the best such prototype?
            true_cls = int(r["class"])
            for vid in sorted(mem.novel):
                st = mem.state(vid, i)
                if st["primary_class"] == true_cls:
                    s = float(np.dot(mem.novel[vid]["proto"], z))
                    true_states.append((s, st))
        gs = known_stats(z, P_known, radii, known_ids=known_ids,
                         best_n=best_n, second_n=second_n, margin_n=margin_n,
                         dist_n=dist_n, rel=rel,
                         track_len=len(feats[r["sample_id"]]),
                         n_novel=len(mem.novel), include_anchor=False)
        with torch.no_grad():
            gate_logit = float(model.gate_forward(
                torch.as_tensor([gs], dtype=torch.float32, device=device))[0])
        if float(torch.sigmoid(torch.as_tensor(gate_logit))) >= gate_thr and kid is not None:
            logs.append(_audit_log(r, i, "KNOWN", kid, None, len(mem.novel),
                                   0, best_k, best_n, gate_logit, float("nan"),
                                   rel, stream, {}, {}, None, n,
                                   second_n=second_n))
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
        prob_reuse = float(torch.sigmoid(torch.as_tensor(reuse_logit)))
        if prob_reuse >= reuse_thr and nid is not None:
            support = mem.support(nid)
            cos_to_center = float(np.dot(mem.novel[nid]["proto"], z))
            mem.update_novel(nid, z, cos_to_center=cos_to_center,
                             update_radius=ck.get("update_radius", False),
                             margin=margin_n)
            logs.append(_audit_log(r, i, "EXISTING_NOVEL", None, nid,
                                   len(mem.novel), support, best_k, best_n,
                                   gate_logit, reuse_logit, rel, stream,
                                   best_state, true_states, margin_n, n,
                                   second_n=second_n))
        else:
            vid = mem.create_novel(z, created_at=i)
            logs.append(_audit_log(r, i, "NEW_NOVEL", None, vid, len(mem.novel),
                                   0, best_k, best_n, gate_logit, reuse_logit,
                                   rel, stream, best_state, true_states, margin_n, n,
                                   second_n=second_n))
        if nid is not None and logs[-1]["predicted_action"] == "EXISTING_NOVEL":
            mem.note_assignment(nid, int(r["class"]), i)
        elif logs[-1]["predicted_action"] == "NEW_NOVEL":
            mem.note_assignment(logs[-1]["predicted_virtual_novel_id"],
                                int(r["class"]), i)
    return logs


def _audit_log(r, i, action, kid, vid, active, support, bk, bn, gl, rl,
               rel, stream, best_state, true_states, margin_n, n_stream,
               second_n=float("nan")):
    true_cls = int(r["class"])
    st = best_state
    true_best_sim = max((s for s, _ in true_states), default=-1.0)
    true_best_support = max((s["support"] for _, s in true_states), default=0)
    true_best_state = None
    for s, stt in true_states:
        if s == true_best_sim:
            true_best_state = stt
            break
    primary_of_assigned = st.get("primary_class") if st else None
    return {
        "sample_id": r["sample_id"], "arrival_index": i,
        "role": r["role"], "class": true_cls,
        "first_occurrence": bool(r["first_occurrence"]),
        "predicted_action": action, "predicted_known_id": kid,
        "predicted_virtual_novel_id": vid,
        "active_novel_prototypes": active, "prototype_support": support,
        "best_known_similarity": bk, "best_novel_similarity": bn,
        "second_novel_similarity": second_n,
        "novel_margin": margin_n,
        "gate_logit": gl, "reuse_logit": rl,
        "track_length": int(r.get("_track_len", 8)),
        "feature_reliability": rel,
        "stage": stage_of(i, n_stream),
        "memory_bucket": active_bucket(active),
        "domain": _domain_of(r),
        "assigned_primary_class": primary_of_assigned,
        "assigned_support": support,
        "assigned_radius": st.get("radius", float("nan")),
        "assigned_dispersion": st.get("dispersion", float("nan")),
        "assigned_age": st.get("age", float("nan")),
        "assigned_mean_margin": st.get("mean_margin", float("nan")),
        "assigned_min_margin": st.get("min_margin", float("nan")),
        "assigned_low_margin_count": st.get("low_margin_count", float("nan")),
        "assigned_consistency": st.get("assignment_consistency", float("nan")),
        "assigned_distinct_classes": st.get("distinct_classes", float("nan")),
        "assigned_recent_stability": st.get("recent_stability", float("nan")),
        "assigned_conf_legal": st.get("conf_legal", float("nan")),
        "true_class_prototype_exists": bool(true_states),
        "true_class_proto_best_sim": true_best_sim,
        "true_class_proto_best_support": true_best_support,
        "true_class_proto_consistency":
            true_best_state.get("assignment_consistency") if true_best_state else float("nan"),
        "true_class_proto_dispersion":
            true_best_state.get("dispersion") if true_best_state else float("nan"),
        "true_class_proto_distinct_classes":
            true_best_state.get("distinct_classes") if true_best_state else float("nan"),
        "novel_margin_used": margin_n if margin_n is not None else float("nan"),
    }


def _margin(ks):
    if ks.shape[0] >= 2:
        order = np.argsort(ks)[::-1]
        return float(ks[order[0]] - ks[order[1]])
    return 0.0


def _primary_category(flags):
    order = ["same_class_proto_unselected", "hub_pollution", "radius_wide",
             "support_reward", "low_margin_forced", "other"]
    for k in order:
        if flags.get(k):
            return k
    return "other"


def error_flags(l, wrong=True):
    f = {
        "same_class_proto_unselected": bool(l.get("true_class_prototype_exists")),
        "hub_pollution": bool(l.get("assigned_consistency", 1.0) < 0.6)
                          or bool(l.get("assigned_distinct_classes", 0) >= 2),
        "radius_wide": bool(l.get("assigned_radius", 0.0) >= 0.5)
                        and bool(l.get("best_novel_similarity", 1.0) < 0.6),
        "support_reward": bool(l.get("assigned_support", 0) >= 8)
                           and bool(l.get("best_novel_similarity", 1.0) < 0.55),
        "low_margin_forced": bool(l.get("novel_margin_used", 1.0) < 0.02),
        "other": True,
    }
    return f


def _write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    fn = fieldnames
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate_audit(logs, gt_rows, out_dir, name, stream):
    routed = [l for l in logs if l["role"] == "novel" and l["predicted_action"] != "KNOWN"]
    wrong = [l for l in routed if l["predicted_action"] == "EXISTING_NOVEL"
             and l.get("assigned_primary_class") != l["class"]]
    first_merge = [l for l in logs if l["role"] == "novel" and l["first_occurrence"]
                   and l["predicted_action"] == "EXISTING_NOVEL"]
    # official-style wrong existing using the final Hungarian assignment
    res, _ = assignment_from_preds(emit_preds(logs), gt_rows)
    hung = res["hungarian_assignment"]
    wrong_hungarian = [l for l in routed
                       if l["predicted_action"] == "EXISTING_NOVEL"
                       and hung.get(int(l["predicted_virtual_novel_id"])) != int(l["class"])]

    # ---- wrong existing records ----
    wrong_rows = []
    for l in wrong:
        fl = error_flags(l)
        row = dict(l)
        row["primary_category"] = _primary_category(fl)
        row.update({f"flag_{k}": int(v) for k, v in fl.items()})
        wrong_rows.append(row)
    _write_csv(out_dir / f"wrong_existing_assignments_{name}.csv", wrong_rows)

    def bucket_table(rows, key, bins, label):
        table = []
        for b in bins:
            sel = [r for r in rows if r.get(key) is not None and b[0] <= r[key] < b[1]]
            denom = [r for r in routed if r.get(key) is not None and b[0] <= r[key] < b[1]]
            table.append({
                label: b[2], "wrong_existing": len(sel),
                "routed_novel": len(denom),
                "wrong_existing_rate": len(sel) / max(len(denom), 1),
            })
        return table

    _write_csv(out_dir / f"wrong_existing_by_memory_scale_{name}.csv",
               bucket_table(wrong, "active_novel_prototypes",
                            [(0, 33, "0-32"), (33, 129, "33-128"),
                             (129, 257, "129-256"), (257, 10**9, "257+")],
                            "memory_bucket"))
    _write_csv(out_dir / f"wrong_existing_by_prototype_support_{name}.csv",
               bucket_table(wrong, "assigned_support",
                            [(0, 1, "1"), (1, 5, "2-4"), (5, 10, "5-9"),
                             (10, 20, "10-19"), (20, 10**9, "20+")],
                            "support_bucket"))
    _write_csv(out_dir / f"wrong_existing_by_margin_{name}.csv",
               bucket_table(wrong, "novel_margin_used",
                            [(-1e-9, 0.01, "<0.01"), (0.01, 0.02, "0.01-0.02"),
                             (0.02, 0.05, "0.02-0.05"), (0.05, 0.1, "0.05-0.10"),
                             (0.1, 1.0, ">0.10")], "margin_bucket"))

    # ---- first occurrence false merge ----
    fm_rows = []
    for l in first_merge:
        fl = error_flags(l)
        row = dict(l)
        row["primary_category"] = _primary_category(fl)
        row.update({f"flag_{k}": int(v) for k, v in fl.items()})
        fm_rows.append(row)
    _write_csv(out_dir / f"first_occurrence_false_merge_{name}.csv", fm_rows)

    hubs = {}
    for l in first_merge:
        vid = l["predicted_virtual_novel_id"]
        h = hubs.setdefault(vid, {"prototype": vid, "first_merge_count": 0,
                                  "absorbed_classes": set(), "supports": [],
                                  "consistencies": [], "dispersions": [],
                                  "radii": []})
        h["first_merge_count"] += 1
        h["absorbed_classes"].add(l["class"])
        h["supports"].append(l.get("assigned_support") or 0)
        h["consistencies"].append(l.get("assigned_consistency") or 0)
        h["dispersions"].append(l.get("assigned_dispersion") or 0)
        h["radii"].append(l.get("assigned_radius") or 0)
    hub_rows = []
    for vid, h in sorted(hubs.items()):
        hub_rows.append({
            "prototype": vid,
            "first_merge_count": h["first_merge_count"],
            "absorbed_class_count": len(h["absorbed_classes"]),
            "absorbed_classes": ",".join(str(c) for c in sorted(h["absorbed_classes"])),
            "mean_support": float(np.mean(h["supports"])),
            "mean_consistency": float(np.mean(h["consistencies"])),
            "mean_dispersion": float(np.mean(h["dispersions"])),
            "mean_radius": float(np.mean(h["radii"])),
        })
    _write_csv(out_dir / f"false_merge_by_hub_prototype_{name}.csv", hub_rows)

    # ---- prototype confidence audit ----
    proto_rows = []
    by_proto = defaultdict(list)
    for l in wrong:
        by_proto[l["predicted_virtual_novel_id"]].append(l)
    for l in first_merge:
        by_proto[l["predicted_virtual_novel_id"]].append(l)
    proto_ids = sorted({int(l["predicted_virtual_novel_id"]) for l in logs
                        if l["predicted_action"] in ("EXISTING_NOVEL", "NEW_NOVEL")})
    for vid in proto_ids:
        sel = [l for l in logs if l["predicted_virtual_novel_id"] == vid]
        cls_cnt = Counter(l["class"] for l in sel)
        purity = max(cls_cnt.values()) / max(len(sel), 1)
        wrong_n = sum(1 for l in wrong if l["predicted_virtual_novel_id"] == vid)
        fm_n = sum(1 for l in first_merge if l["predicted_virtual_novel_id"] == vid)
        last = sel[-1] if sel else {}
        proto_rows.append({
            "prototype": vid,
            "support": last.get("assigned_support", len(sel)),
            "dispersion": last.get("assigned_dispersion", float("nan")),
            "mean_margin": last.get("assigned_mean_margin", float("nan")),
            "min_margin": last.get("assigned_min_margin", float("nan")),
            "low_margin_count": last.get("assigned_low_margin_count", float("nan")),
            "recent_stability": last.get("assigned_recent_stability", float("nan")),
            "age": last.get("assigned_age", float("nan")),
            "radius": last.get("assigned_radius", float("nan")),
            "conf_legal": last.get("assigned_conf_legal", float("nan")),
            "distinct_classes": len(cls_cnt),
            "purity_offline": purity,
            "wrong_existing_caused": wrong_n,
            "first_merge_caused": fm_n,
            "primary_class_offline": max(cls_cnt, key=cls_cnt.get),
        })
    _write_csv(out_dir / f"prototype_confidence_analysis_{name}.csv", proto_rows)

    # ---- correlation ----
    try:
        from scipy.stats import spearmanr
    except Exception:
        spearmanr = None
    corr_rows = []
    feats = ["support", "dispersion", "mean_margin", "min_margin",
             "low_margin_count", "recent_stability", "age", "radius",
             "conf_legal"]
    for feat in feats:
        xs = [r[feat] for r in proto_rows if isinstance(r.get(feat), (int, float))]
        ys_p = [r["purity_offline"] for r in proto_rows
                if isinstance(r.get(feat), (int, float))]
        ys_w = [r["wrong_existing_caused"] for r in proto_rows
                if isinstance(r.get(feat), (int, float))]
        row = {"feature": feat}
        if spearmanr is not None and len(xs) > 2:
            rho_p, p_p = spearmanr(xs, ys_p)
            rho_w, p_w = spearmanr(xs, ys_w)
            row.update({"corr_purity": float(rho_p), "p_purity": float(p_p),
                        "corr_wrong_existing": float(rho_w),
                        "p_wrong_existing": float(p_w)})
        else:
            row.update({"corr_purity": float("nan"), "p_purity": float("nan"),
                        "corr_wrong_existing": float("nan"),
                        "p_wrong_existing": float("nan")})
        corr_rows.append(row)
    _write_csv(out_dir / f"confidence_purity_correlation_{name}.csv", corr_rows)

    return {
        "name": name,
        "stream": stream,
        "routed_novel": len(routed),
        "wrong_existing": len(wrong),
        "wrong_existing_rate": len(wrong) / max(len(routed), 1),
        "wrong_existing_hungarian": len(wrong_hungarian),
        "wrong_existing_hungarian_rate": len(wrong_hungarian) / max(len(routed), 1),
        "first_merge": len(first_merge),
        "first_merge_rate": len(first_merge) / max(sum(1 for l in logs
                        if l["role"] == "novel" and l["first_occurrence"]), 1),
        "hub_prototypes": len(hubs),
    }


def replay(configs, stream, device="cuda"):
    out_dir = ROOT / "outputs" / "iclr27_phase4e" / "audit"
    labels = load_train_labels()
    gt_rows = None
    if stream == "long":
        rows, gt_rows, feats, syn_mean = load_stream_cache()
        proto_feats = None
    else:
        gt = load_gt("pure")
        gt_rows = gt
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
            r["_track_len"] = len(r["frame_ids"])
        feats = {sid: f[:8] for sid, f in load_frame_features("gt_tracks_mean").items()}
        syn_mean = None
        proto_feats = {sid: f[:8] for sid, f in
                       load_frame_features("train_known_mean").items()}
    for cname, path, gt_thr, rt_thr in configs:
        model, ck = load_msr_model(ROOT / path, device)
        logs = run_audit_stream(model, ck, rows, feats, labels, device, stream,
                                gate_thr=gt_thr, reuse_thr=rt_thr,
                                syn_mean=syn_mean, proto_feats=proto_feats)
        # overwrite second/margin fields for routed rows from the decision loop
        # (already exact at decision time)
        summary = aggregate_audit(logs, gt_rows, out_dir, cname, stream)
        print(summary, flush=True)
        (out_dir / f"audit_log_{cname}_{stream}.json").write_text(
            json.dumps(logs, indent=1, default=str))
    _merge_audit_files(configs, stream, out_dir)


def _merge_audit_files(configs, stream, out_dir):
    """Merge per-config audit CSVs into the canonical Phase 4E files."""
    names = ["wrong_existing_assignments", "first_occurrence_false_merge",
             "false_merge_by_hub_prototype", "prototype_confidence_analysis",
             "confidence_purity_correlation", "wrong_existing_by_memory_scale",
             "wrong_existing_by_prototype_support", "wrong_existing_by_margin"]
    for stem in names:
        all_rows = []
        for cname, _, _, _ in configs:
            p = out_dir / f"{stem}_{cname}.csv"
            if not p.exists():
                continue
            for r in csv.DictReader(open(p)):
                r["config"] = cname
                r["stream"] = stream
                all_rows.append(r)
        _write_csv(out_dir / f"{stem}.csv", all_rows)


def _kmeans_cos(X, k, iters=30, restarts=3, seed=2026):
    best = None
    rng = np.random.RandomState(seed)
    for _ in range(restarts):
        idx = rng.choice(len(X), k, replace=False)
        centers = X[idx].copy()
        assign = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            sims = centers @ X.T
            assign = np.argmax(sims, axis=0)
            new_centers = []
            for c in range(k):
                sel = X[assign == c]
                if len(sel) == 0:
                    new_centers.append(X[rng.randint(len(X))])
                else:
                    new_centers.append(_norm(sel.mean(axis=0)))
            centers = np.stack(new_centers)
        sse = float(sum(1.0 - float(np.dot(X[i], centers[assign[i]]))
                        for i in range(len(X))))
        if best is None or sse < best[0]:
            best = (sse, centers, assign)
    return best


def multimodal_audit(device="cuda"):
    out_dir = ROOT / "outputs" / "iclr27_phase4e" / "audit"
    labels = load_train_labels()
    model, ck = load_msr_model(ROOT / "runs/orbit_msr/msr_nr2/model.pth", device)

    # official novel tracks (real classes)
    gt = load_gt("pure")
    gt_novel = [g for g in gt if g["protocol_role"] == "novel"]
    feats_all = load_frame_features("gt_tracks_mean")
    sids = [g["sample_id"] for g in gt_novel]
    feats = {sid: feats_all[sid][:8] for sid in sids if sid in feats_all}
    zs, _ = embed_many(model, feats, list(feats.keys()), device)
    by_class = defaultdict(list)
    for g in gt_novel:
        if g["sample_id"] in zs:
            by_class[int(g["ground_truth_category_id"])].append(zs[g["sample_id"]])
    rows = _multimodal_rows(by_class, "official")
    _write_csv(out_dir / "multimodality_by_class.csv",
               [dict(r, source="official") for r in rows])

    # long-stream real-only classes
    lrows, _, lfeats, _ = load_stream_cache()
    real_rows = [r for r in lrows if r["role"] == "novel" and int(r["class"]) < 1000000]
    lzs, _ = embed_many(model, lfeats, [r["sample_id"] for r in real_rows], device)
    by_class_l = defaultdict(list)
    for r in real_rows:
        by_class_l[int(r["class"])].append(lzs[r["sample_id"]])
    rows_l = _multimodal_rows(by_class_l, "long_real_only")
    all_rows = rows + rows_l
    if not all_rows:
        all_rows = [{"source": "none", "class": -1, "n": 0, "insufficient": True}]
    fieldnames = []
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_dir / "single_vs_multi_center_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print("multimodal rows:", len(rows), len(rows_l))


def _multimodal_rows(by_class, source):
    rows = []
    centers = {}
    for cls, xs_list in sorted(by_class.items()):
        X = np.stack(xs_list).astype(np.float32)
        n = len(X)
        if n < 6:
            rows.append({"source": source, "class": cls, "n": n,
                         "insufficient": True})
            continue
        c1 = _norm(X.mean(axis=0))
        centers[cls] = (c1, X)
        d1 = 1.0 - X @ c1
        sse1 = float(d1.sum())
        sse2, centers2, a2 = _kmeans_cos(X, 2)
        sse3, centers3, a3 = _kmeans_cos(X, 3)
        sizes2 = [int((a2 == k).sum()) for k in range(2)]
        sizes3 = [int((a3 == k).sum()) for k in range(3)]
        # mean silhouette for 2 clusters (distance = 1-cos)
        sil = []
        for i in range(n):
            a = 1.0 - float(np.dot(X[i], centers2[a2[i]]))
            other = 1 - a2[i]
            b = 1.0 - float(np.dot(X[i], centers2[other]))
            sil.append((b - a) / max(a, b, 1e-9))
        mean_sil2 = float(np.mean(sil))
        bic1 = n * math.log(max(sse1 / n, 1e-12)) + 2 * math.log(n)
        bic2 = n * math.log(max(sse2 / n, 1e-12)) + 4 * math.log(n)
        bic3 = n * math.log(max(sse3 / n, 1e-12)) + 6 * math.log(n)
        rows.append({
            "source": source, "class": cls, "n": n,
            "insufficient": False,
            "sse1": sse1, "sse2": sse2, "sse3": sse3,
            "sse_reduction_2": 1.0 - sse2 / max(sse1, 1e-12),
            "sse_reduction_3": 1.0 - sse3 / max(sse1, 1e-12),
            "mean_dist_1": float(d1.mean()), "p90_dist_1": float(np.percentile(d1, 90)),
            "max_dist_1": float(d1.max()),
            "cluster_sizes_2": str(sizes2), "cluster_sizes_3": str(sizes3),
            "min_cluster_share_2": min(sizes2) / n,
            "mean_silhouette_2": mean_sil2,
            "bic1": bic1, "bic2": bic2, "bic3": bic3,
            "bimodal_candidate": bool(min(sizes2) / n >= 0.25
                                      and 1.0 - sse2 / max(sse1, 1e-12) >= 0.15
                                      and mean_sil2 >= 0.0),
        })
    # cross-class single-center overlap (Analysis A)
    for r in rows:
        if r["insufficient"]:
            continue
        cls = r["class"]
        c1, X = centers[cls]
        other_centers = [centers[c2][0] for c2 in centers if c2 != cls]
        if not other_centers:
            continue
        P_other = np.stack(other_centers).astype(np.float32)
        d_own = 1.0 - X @ c1
        d_cross = 1.0 - X @ P_other.T
        d_cross_min = d_cross.min(axis=1)
        r["cross_nearest_mean"] = float(d_cross_min.mean())
        r["cross_nearest_p50"] = float(np.percentile(d_cross_min, 50))
        r["overlap_own_gt_cross"] = float((d_own > d_cross_min).mean())
        r["single_center_confused"] = bool((d_own > d_cross_min).mean() >= 0.2
                                           or float(np.percentile(d_own, 90))
                                           >= float(np.percentile(d_cross_min, 50)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "multimodal", "all"],
                    default="all")
    ap.add_argument("--stream", choices=["long", "official"], default="official")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    configs = [
        ("c1", "runs/orbit_msr/msr_nr2/model.pth", 0.5, 0.45),
        ("c2", "runs/orbit_msr/msr_c2/model.pth", 0.5, 0.45),
    ]
    if args.mode in ("replay", "all"):
        replay(configs, args.stream, args.device)
    if args.mode in ("multimodal", "all"):
        multimodal_audit(args.device)


if __name__ == "__main__":
    main()
