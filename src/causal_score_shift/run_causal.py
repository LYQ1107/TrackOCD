#!/usr/bin/env python3
"""Part B: causal score-shift adaptation (C0-C5)."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.causal_score_shift.causal_routers import (
    C0Legacy, C1Global, C2Translation, C3LocationScale, C4Reliability, C5AllTrack,
)
from src.causal_score_shift.data.causal_stream import load_video_streams
from src.domain_router.data.proxy_builder import build_p1_folds
from src.domain_router.evaluation.run_router import (
    load_frame_dict, load_train_meta, val_meta, load_train_known,
)
from src.domain_router.features.router_features import compute_router_features
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import load_mean_features, build_prototypes
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt, subset_ids

OUT = PROJECT_ROOT / "outputs" / "causal_score_shift"
RUNS = PROJECT_ROOT / "runs" / "causal_score_shift"
STREAMS = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
SEEDS = ("main_seed1027", "main_seed1028", "main_seed1029")
SUBSETS = ("full", "repeated", "balanced")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def reference_stats():
    oof = json.loads((PROJECT_ROOT / "runs/router_audit/oof_scores.json").read_text())
    known = np.array([r["score"] for r in oof["R1"] if r["label"] == 1])
    all_s = np.array([r["score"] for r in oof["R1"]])
    all_y = np.array([r["label"] for r in oof["R1"]])
    ref_median = float(np.median(known))
    ref_mad = float(np.median(np.abs(known - ref_median)))
    hc = {}
    for pt in (0.95, 0.975):
        best = None
        for thr in np.arange(0.45, 0.96, 0.005):
            m = all_s >= thr
            if m.sum() < 20:
                continue
            prec = all_y[m].mean()
            if prec >= pt and (best is None or thr < best):
                best = float(thr)
        hc[pt] = best if best is not None else 0.9
    return {
        "ref_known_median": ref_median,
        "ref_known_mad": ref_mad,
        "hc_score": hc,
        "oof_known_count": int(len(known)),
    }


def fold_features(fold, feats_tr, labels, frames, meta):
    src_pos = [s for s in fold["source_positive_ids"] if s in feats_tr]
    src_classes = {labels[s] for s in src_pos}
    protos = build_prototypes(feats_tr, labels, src_classes)
    knn = np.stack([feats_tr[s] for s in src_pos]) if src_pos else None
    feats = {}
    meta_f = {}
    for s in fold["target_positive_ids"] + fold["target_negative_ids"]:
        if s not in feats_tr:
            continue
        f = compute_router_features(feats_tr[s], protos, knn, frames.get(s), meta.get(s))
        feats[s] = f
        meta_f[s] = meta.get(s, {})
    labels_f = {s: 1 for s in fold["target_positive_ids"]}
    labels_f.update({s: 0 for s in fold["target_negative_ids"]})
    return feats, labels_f, protos


def run_video_stream(vstreams, feats, labels_f, make_router, ref):
    """Causal replay; returns decisions per sample id."""
    out = {}
    router = None
    for vid, vrows in vstreams.items():
        router = make_router()
        router.reset_video(vid)
        for r in vrows:
            sid = r["sample_id"]
            if sid not in feats:
                continue
            state = dict(feats[sid])
            is_known = router.predict(state)
            out[sid] = int(is_known)
            router.update_after_prediction(state, is_known)
    return out


def routing_metrics(out, labels_f):
    y = np.array([labels_f[s] for s in out])
    p = np.array([out[s] for s in out])
    kr = (p[y == 1] == 1).mean() if (y == 1).any() else 0.0
    nr = (p[y == 0] == 0).mean() if (y == 0).any() else 0.0
    hm = 2 * kr * nr / (kr + nr) if kr + nr > 0 else 0.0
    return kr, nr, hm


def build_router(method, thr, ref, hc_pt=0.95, min_anchors=10, ema=0.25):
    if method == "C0":
        return C0Legacy()
    if method == "C1":
        return C1Global(thr)
    if method == "C2":
        return C2Translation(thr, ref["ref_known_median"], ref["hc_score"][hc_pt],
                             0.05, min_anchors, ema,
                             max_shift=2 * ref["ref_known_mad"])
    if method == "C3":
        return C3LocationScale(thr, ref["ref_known_median"], ref["ref_known_mad"],
                               ref["hc_score"][hc_pt], 0.05, min_anchors,
                               min_anchors, ema, max_shift=2 * ref["ref_known_mad"])
    if method == "C4":
        return C4Reliability(thr, ref["ref_known_median"], ref["hc_score"][hc_pt],
                             0.05, min_anchors, ema, max_shift=2 * ref["ref_known_mad"])
    return C5AllTrack(thr)


def proxy_simulation(feats_tr, labels, frames, meta, ref):
    folds = build_p1_folds(seed=1027)
    thr_c1 = 0.42  # pooled OOF threshold from audit
    grid = []
    for mc in (5, 10, 20):
        for ema in (0.1, 0.25, 0.5):
            for pt in (0.95, 0.975):
                grid.append((mc, ema, pt))
    rows = []
    # C0/C1 fixed
    for method in ("C0", "C1"):
        fold_res = []
        for fold in folds:
            feats, labels_f, _ = fold_features(fold, feats_tr, labels, frames, meta)
            vstreams = build_target_vstreams(fold)
            out = run_video_stream(vstreams, feats, labels_f,
                                   lambda m=method: build_router(m, thr_c1, ref), ref)
            kr, nr, hm = routing_metrics(out, labels_f)
            fold_res.append(hm)
            rows.append({"method": method, "target_domain": fold["target_domain"],
                         "known_recall": kr, "novel_recall": nr, "hmean": hm})
        print(method, "mean H", round(float(np.mean(fold_res)), 4), flush=True)
    best_per_method = {}
    for method in ("C2", "C3", "C4"):
        best = None
        for mc, ema, pt in grid:
            fold_res = []
            krs, nrs = [], []
            for fold in folds:
                feats, labels_f, _ = fold_features(fold, feats_tr, labels, frames, meta)
                vstreams = build_target_vstreams(fold)
                out = run_video_stream(
                    vstreams, feats, labels_f,
                    lambda m=method, mc=mc, ema=ema, pt=pt:
                        build_router(m, thr_c1, ref, pt, mc, ema), ref)
                kr, nr, hm = routing_metrics(out, labels_f)
                fold_res.append(hm); krs.append(kr); nrs.append(nr)
            score = float(np.mean(fold_res))
            if best is None or score > best[0]:
                best = (score, mc, ema, pt, float(np.mean(krs)), float(np.mean(nrs)),
                        float(np.min(fold_res)))
        best_per_method[method] = best
        print(method, "best", best, flush=True)
    return rows, best_per_method


def build_target_vstreams(fold):
    # proxy target videos: group target tracks by video via train_known rows
    from src.domain_router.data.domain_metadata import load_train_known_rows
    rows = load_train_known_rows()
    tgt_ids = set(fold["target_positive_ids"] + fold["target_negative_ids"])
    vstreams = {}
    for r in rows:
        if r["sample_id"] in tgt_ids:
            vstreams.setdefault(r["video_id"], []).append({
                "sample_id": r["sample_id"], "video_id": r["video_id"],
                "frame_ids": list(range(len(r["boxes_xyxy"]))),
            })
    for v in vstreams:
        vstreams[v].sort(key=lambda r: r["frame_ids"][0])
    return vstreams


def stress_tests(ref, thr_c1):
    tests = []
    # synthetic known-dominant / novel-dominant / all-novel / all-known videos
    cases = {
        "all_known": [0.80, 0.82, 0.78, 0.85],
        "all_novel": [0.30, 0.32, 0.28, 0.35],
        "known_dominant": [0.80, 0.78, 0.30, 0.82],
        "novel_dominant": [0.30, 0.32, 0.80, 0.28],
        "single_track": [0.75],
    }
    for cname, scores in cases.items():
        for method in ("C0", "C1", "C2", "C3", "C4"):
            router = build_router(method, thr_c1, ref, 0.95, 5, 0.25)
            router.reset_video(0)
            decisions = []
            for i, s in enumerate(scores):
                state = {"s1": s, "margin": 0.1}
                d = router.predict(state)
                decisions.append(int(d))
                router.update_after_prediction(state, d)
            tests.append({"case": cname, "method": method,
                          "decisions": decisions,
                          "known_ratio": round(float(np.mean(decisions)), 3)})
    return tests


def full_evaluation(selected, ref, thr_c1):
    feats_tr, labels = load_train_known("dinov2")
    feats_val = load_mean_features("dinov2", "gt_tracks_mean")
    frames_val = load_frame_dict("gt_tracks_mean")
    meta_val = val_meta()
    protos_all = build_prototypes(feats_tr, labels, set(labels.values()))
    knn_all = np.stack([feats_tr[s] for s in feats_tr])
    feats_cache = {}
    for s in feats_val:
        feats_cache[s] = compute_router_features(feats_val[s], protos_all, knn_all,
                                                 frames_val.get(s), meta_val.get(s))
    rows = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in SUBSETS:
            for stream in STREAMS:
                vstreams = load_video_streams(stream)
                sub = subset_ids(proto, subset)
                for method in ("C0", "C1", selected, "C5") if selected != "C1" else ("C0", "C1", "C5"):
                    methods = ("C0", "C1", "C2", "C3", "C4", "C5")
                    if method not in methods:
                        continue
                    preds = []
                    router = build_router(method, thr_c1, ref, 0.95, 10, 0.25)
                    mem = B2Memory(protos_all, threshold=0.45, novel_only=True)
                    for vid, vrows in vstreams.items():
                        router.reset_video(vid)
                        for r in vrows:
                            sid = r["sample_id"]
                            if sid not in feats_cache:
                                continue
                            state = dict(feats_cache[sid])
                            is_known = router.predict(state)
                            if is_known:
                                best_id, _ = max(protos_all.items(),
                                                 key=lambda kv: float(np.dot(feats_val[sid], kv[1])))
                                preds.append(emit(sid, r["stream_order"], "known", known_id=best_id))
                            else:
                                vid2, _ = mem.predict_one(feats_val[sid], sid, r["stream_order"])
                                preds.append(emit(sid, r["stream_order"], "novel", virtual_id=vid2))
                            router.update_after_prediction(state, is_known)
                    ev = TrackOCDEvaluator(gt)
                    res = ev.evaluate(preds, subset_ids=sub)
                    rows.append({
                        "method": method, "protocol": proto, "subset": subset,
                        "seed": stream,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    })
    write_csv(OUT / "metrics/full_results.csv", rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    feats_tr, labels = load_train_known("dinov2")
    frames = load_frame_dict("train_known_mean")
    meta = load_train_meta()
    ref = reference_stats()
    (RUNS / "reference_stats.json").write_text(json.dumps(ref, indent=2))
    print("ref", ref, flush=True)
    proxy_rows, best_per_method = proxy_simulation(feats_tr, labels, frames, meta, ref)
    write_csv(OUT / "metrics/train_proxy_results.csv", proxy_rows)
    # method selection by train-only H-mean with constraints
    c0_hm = float(np.mean([r["hmean"] for r in proxy_rows if r["method"] == "C0"]))
    c0_kr = float(np.mean([r["known_recall"] for r in proxy_rows if r["method"] == "C0"]))
    c0_nr = float(np.mean([r["novel_recall"] for r in proxy_rows if r["method"] == "C0"]))
    selection = []
    for method in ("C1", "C2", "C3", "C4"):
        vals = [r for r in proxy_rows if r["method"] == method]
        hm = float(np.mean([r["hmean"] for r in vals])) if vals else 0.0
        kr = float(np.mean([r["known_recall"] for r in vals])) if vals else 0.0
        nr = float(np.mean([r["novel_recall"] for r in vals])) if vals else 0.0
        worst = float(np.min([r["hmean"] for r in vals])) if vals else 0.0
        ok = (kr >= c0_kr - 0.03 and nr > c0_nr and worst >= c0_hm - 0.03)
        selection.append({"method": method, "hmean": hm, "known_recall": kr,
                          "novel_recall": nr, "worst_domain": worst, "feasible": ok})
    sel_rows = [s for s in selection if s["feasible"]]
    selected = None
    if sel_rows:
        sel_rows.sort(key=lambda s: (-s["hmean"], {"C2": 0, "C3": 1, "C4": 2}[s["method"]],
                                     s["method"]))
        selected = sel_rows[0]["method"]
    write_csv(OUT / "metrics/method_selection.csv", selection)
    print("selection", selection, "selected", selected, flush=True)
    stress = stress_tests(ref, 0.42)
    write_csv(OUT / "metrics/stress_tests.csv", stress)
    rows = full_evaluation(selected, ref, 0.42)
    # summary + gate
    def agg(method, proto="pure", subset="full"):
        vals = [r for r in rows if r["method"] == method and r["protocol"] == proto
                and r["subset"] == subset and r["seed"] in SEEDS]
        if not vals:
            return {}
        out = {}
        for k in ("all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                  "conditional_novel_acc", "novel_only_nmi", "novel_only_ari",
                  "novel_routing_recall", "novel_count_abs_error"):
            v = [float(r[k]) for r in vals]
            out[k] = {"mean": statistics.mean(v),
                      "std": statistics.stdev(v) if len(v) > 1 else 0.0}
        return out
    summary = []
    for method in ("C0", "C1", "C2", "C3", "C4", "C5"):
        for proto in ("pure", "ov_assisted"):
            for subset in SUBSETS:
                a = agg(method, proto, subset)
                if not a:
                    continue
                row = {"method": method, "protocol": proto, "subset": subset}
                for k, v in a.items():
                    row[f"{k}_mean"] = v["mean"]; row[f"{k}_std"] = v["std"]
                summary.append(row)
    write_csv(OUT / "metrics/final_summary.csv", summary)
    c0 = agg("C0")
    bname = selected if selected is not None else "C1"
    b = agg(bname)
    criteria = {
        "route_novel_gain": b["route_aware_novel_acc"]["mean"] - c0["route_aware_novel_acc"]["mean"],
        "routing_recall_gain": b["novel_routing_recall"]["mean"] - c0["novel_routing_recall"]["mean"],
        "known_drop": c0["overall_known_acc"]["mean"] - b["overall_known_acc"]["mean"],
        "conditional_novel_acc": b["conditional_novel_acc"]["mean"],
        "novel_nmi": b["novel_only_nmi"]["mean"],
        "novel_ari": b["novel_only_ari"]["mean"],
        "count_error": b["novel_count_abs_error"]["mean"],
    }
    passed = all([
        criteria["route_novel_gain"] >= 0.030,
        criteria["routing_recall_gain"] >= 0.050,
        criteria["known_drop"] <= 0.030,
        criteria["conditional_novel_acc"] >= 0.640,
        criteria["novel_nmi"] >= 0.890,
        criteria["novel_ari"] >= 0.480,
        criteria["count_error"] <= 90,
    ])
    gate = {
        "status": "PASS_CAUSAL_SCORE_SHIFT" if passed else "NO_CLEAR_CALIBRATION_GAIN",
        "selected_method": bname,
        "selection_source": "train_only_proxy",
        "criteria": criteria,
        "c0": {k: v["mean"] for k, v in c0.items()},
        "best": {k: v["mean"] for k, v in b.items()},
    }
    (RUNS / "causal_gate.json").write_text(json.dumps(gate, indent=2))
    (RUNS / "status.txt").write_text(gate["status"] + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
