#!/usr/bin/env python3
"""Part A: audit the previous domain-router results and recompute corrected
router results with pooled out-of-fold thresholds."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_router.calibration.nested_calibration import hmean, build_feature_matrix
from src.domain_router.data.proxy_builder import build_p1_folds
from src.domain_router.evaluation.run_router import (
    load_frame_dict, load_train_meta, val_meta, run_router_stream, load_train_known,
)
from src.domain_router.features.router_features import compute_router_features
from src.domain_router.models.routers import FEATURE_SETS, make_router
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.data.track_stream_dataset import load_stream_rows
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import load_mean_features, build_prototypes, proxy_split
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt, subset_ids

OUT = PROJECT_ROOT / "outputs" / "router_audit"
RUNS = PROJECT_ROOT / "runs" / "router_audit"
OLD = PROJECT_ROOT / "outputs" / "domain_router"
STREAMS = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
SEEDS = ("main_seed1027", "main_seed1028", "main_seed1029")
SUBSETS = ("full", "repeated", "balanced")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def audit_inputs():
    files = [
        "outputs/domain_router/metrics/proxy_outer_folds.csv",
        "outputs/domain_router/metrics/router_selection.csv",
        "outputs/domain_router/metrics/r0_reproduction.csv",
        "outputs/domain_router/metrics/router_full_results.csv",
        "outputs/domain_router/metrics/router_subgroups.csv",
        "outputs/domain_router/metrics/router_final_summary.csv",
        "runs/domain_router/router_gate.json",
        "runs/domain_router/final_routers.json",
        "data/domain_router/proxy_protocol/manifest.json",
        "src/domain_router/evaluation/run_router.py",
        "src/domain_router/calibration/nested_calibration.py",
    ]
    out = {}
    for f in files:
        p = PROJECT_ROOT / f
        out[f] = sha256_file(p) if p.exists() else "MISSING"
    (OUT / "audit_input_hashes.json").write_text(json.dumps(out, indent=2))
    return out


def method_registry():
    rows = []
    for name, cfg, cls, feats, nfeat, model, nparam, thr_src, thr in [
        ("R0", "r0_legacy.yaml", "LegacyRouter", ["s1"], 1, "rule", 1,
         "frozen", 0.45),
        ("R1", "r1_margin.yaml", "MarginRouter", ["s1", "margin"], 2, "rule", 2,
         "fold-mean", 0.45),
        ("R2", "r2_relative.yaml", "LogisticRouter", ["s1", "margin", "z_top1", "entropy"],
         4, "logistic", 5, "fold-mean", 0.4271),
        ("R3", "r3_knn.yaml", "LogisticRouter", ["s1", "margin", "k1", "k5", "k10", "nearest_dist"],
         6, "logistic", 7, "default", 0.5),
        ("R4", "r4_logistic.yaml", "LogisticRouter", ["s1", "s2", "margin", "z_top1", "entropy",
         "k1", "k5", "k10", "proto_consistency", "log_len", "frame_consistency", "log_area"],
         12, "logistic", 13, "default", 0.5),
        ("R5", "r5_mlp.yaml", "not_run", [], 0, "mlp", 0, "n/a", None),
    ]:
        rows.append({
            "method_id": name, "config_file": f"configs/domain_router/{cfg}",
            "registry_name": name, "python_class": cls,
            "training_function": "nested_fold_logistic" if model == "logistic" else "none",
            "feature_names": ",".join(feats), "feature_count": nfeat,
            "model_type": model, "parameter_count": nparam,
            "threshold_source": thr_src, "threshold_value": thr,
            "prediction_file": "router_full_results.csv",
            "summary_file": "router_final_summary.csv",
        })
    write_csv(OUT / "method_registry_audit.csv", rows)
    return rows


def regenerate_oof_scores():
    """Fit each method on source folds and score target tracks (OOF)."""
    feats, labels = load_train_known("dinov2")
    frames = load_frame_dict("train_known_mean")
    meta = load_train_meta()
    folds = build_p1_folds(seed=1027)
    oof = {name: [] for name in ("R1", "R2", "R3", "R4")}
    feas = []
    r0_oof_kr_list = []
    for fold in folds:
        src_pos = [s for s in fold["source_positive_ids"] if s in feats]
        src_neg = [s for s in fold["source_negative_ids"] if s in feats]
        src_classes = {labels[s] for s in src_pos}
        protos = build_prototypes(feats, labels, src_classes)
        knn = np.stack([feats[s] for s in src_pos]) if src_pos else None
        tgt_pos = [s for s in fold["target_positive_ids"] if s in feats]
        tgt_neg = [s for s in fold["target_negative_ids"] if s in feats]
        t_ids = tgt_pos + tgt_neg
        y_t = np.array([1] * len(tgt_pos) + [0] * len(tgt_neg))
        # R0 OOF known recall
        r0_kr = 0.0
        if tgt_pos:
            r0_kr = sum(
                1 for s in tgt_pos
                if compute_router_features(feats[s], protos, knn, frames.get(s), meta.get(s))["s1"] >= 0.45
            ) / len(tgt_pos)
        r0_oof_kr_list.append(r0_kr)
        floor = max(0.0, r0_kr - 0.03)
        for name in ("R1", "R2", "R3", "R4"):
            if name == "R1":
                scores = np.array([
                    compute_router_features(feats[s], protos, knn, frames.get(s), meta.get(s))["s1"]
                    for s in t_ids
                ])
            else:
                tr_ids = src_pos + src_neg
                y_tr = np.array([1] * len(src_pos) + [0] * len(src_neg))
                X_tr = build_feature_matrix(tr_ids, feats, protos, knn, frames, meta,
                                            FEATURE_SETS[name])
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
                X_t = build_feature_matrix(t_ids, feats, protos, knn, frames, meta,
                                           FEATURE_SETS[name])
                scores = lr.predict_proba(X_t)[:, 1]
            best = None
            for thr in np.arange(0.20, 0.96, 0.01):
                pred = scores >= thr
                kr = pred[y_t == 1].mean() if (y_t == 1).any() else 0.0
                nr = (~pred[y_t == 0]).mean() if (y_t == 0).any() else 0.0
                if kr < floor:
                    continue
                hm = hmean(kr, nr)
                if best is None or hm > best[0]:
                    best = (hm, kr, nr, float(thr))
            feasible = best is not None
            feas.append({
                "method": name, "target_domain": fold["target_domain"],
                "proxy_known": len(tgt_pos), "proxy_novel": len(tgt_neg),
                "known_recall_floor": floor,
                "feasible": feasible,
                "hmean": best[0] if feasible else 0.0,
                "known_recall": best[1] if feasible else 0.0,
                "novel_recall": best[2] if feasible else 0.0,
                "threshold": best[3] if feasible else None,
            })
            for s, sc, yv in zip(t_ids, scores, y_t):
                oof[name].append({"sample_id": s, "score": float(sc), "label": int(yv),
                                  "target_domain": fold["target_domain"]})
    write_csv(OUT / "fold_feasibility.csv", feas)
    with open(RUNS / "oof_scores.json", "w") as f:
        json.dump(oof, f)
    return oof, feas, r0_oof_kr_list


def pooled_oof_thresholds(oof, r0_oof_kr_list):
    """T-C: pooled OOF threshold per method with known-recall floor."""
    r0_floor = max(0.0, float(np.mean(r0_oof_kr_list)) - 0.03)
    out = {"R0": 0.45, "R1": None, "R2": None, "R3": None, "R4": None}
    for name in ("R1", "R2", "R3", "R4"):
        rows = oof[name]
        scores = np.array([r["score"] for r in rows])
        y = np.array([r["label"] for r in rows])
        best = None
        for thr in np.arange(0.20, 0.96, 0.01):
            pred = scores >= thr
            kr = pred[y == 1].mean(); nr = (~pred[y == 0]).mean()
            if kr < r0_floor:
                continue
            hm = hmean(kr, nr)
            if best is None or hm > best[0]:
                best = (hm, kr, nr, float(thr))
        out[name] = best[3] if best else None
    return out, r0_floor


def threshold_aggregation_comparison(feas, oof):
    rows = []
    for name in ("R1", "R2", "R3", "R4"):
        fold_rows = [r for r in feas if r["method"] == name]
        ta = float(np.mean([r["threshold"] for r in fold_rows if r["feasible"]])) if any(
            r["feasible"] for r in fold_rows) else None
        tb = float(np.median([r["threshold"] for r in fold_rows if r["feasible"]])) if any(
            r["feasible"] for r in fold_rows) else None
        rows.append({
            "method": name,
            "T_A_fold_mean": ta,
            "T_B_sample_weighted_median": tb,
            "T_C_pooled_oof": pooled_oof_cache.get(name),
            "feasible_folds": sum(1 for r in fold_rows if r["feasible"]),
            "infeasible_folds": sum(1 for r in fold_rows if not r["feasible"]),
        })
    write_csv(OUT / "threshold_aggregation_comparison.csv", rows)
    return rows


pooled_oof_cache = {}


def corrected_full_results(oof_thr):
    """Deploy corrected thresholds (T-C) on TrackOCD val and evaluate."""
    feats_tr, labels = load_train_known("dinov2")
    feats_val = load_mean_features("dinov2", "gt_tracks_mean")
    frames_val = load_frame_dict("gt_tracks_mean")
    meta_val = val_meta()
    protos_all = build_prototypes(feats_tr, labels, set(labels.values()))
    knn_all = np.stack([feats_tr[s] for s in feats_tr])
    pk, pn = proxy_split(labels, seed=1027)
    tr_pos = [s for s, c in labels.items() if c in pk and s in feats_tr]
    tr_neg = [s for s, c in labels.items() if c in pn and s in feats_tr]
    routers = {"R0": make_router("R0", protos_all, 0.45)}
    for name in ("R1", "R2", "R3", "R4"):
        thr = oof_thr.get(name)
        if name == "R1":
            routers[name] = make_router("R1", protos_all, thr if thr is not None else 0.45, 0.0)
        else:
            frames_tr = load_frame_dict("train_known_mean")
            meta_tr = load_train_meta()
            X = build_feature_matrix(tr_pos + tr_neg, feats_tr, protos_all, knn_all,
                                     frames_tr, meta_tr, FEATURE_SETS[name])
            y = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
            routers[name] = make_router(name, protos_all, thr if thr is not None else 0.5,
                                        coef=lr.coef_[0], intercept=lr.intercept_[0])
    rows = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in SUBSETS:
            for stream in STREAMS:
                srows = load_stream_rows(stream)
                sub = subset_ids(proto, subset)
                for name in ("R0", "R1", "R2", "R3", "R4"):
                    preds = run_router_stream(name, routers[name], protos_all, feats_val,
                                              frames_val, meta_val, knn_all, srows, gt)
                    ev = TrackOCDEvaluator(gt)
                    res = ev.evaluate(preds, subset_ids=sub)
                    rows.append({
                        "router": name, "protocol": proto, "subset": subset,
                        "seed": stream,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    })
    write_csv(OUT / "corrected_router_full_results.csv", rows)
    return rows


def summary_and_gate(rows):
    def agg(name, proto="pure", subset="full"):
        vals = [r for r in rows if r["router"] == name and r["protocol"] == proto
                and r["subset"] == subset and r["seed"] in SEEDS]
        if not vals:
            return {}
        out = {}
        for k in ("all_track_acc", "overall_known_acc", "route_aware_novel_acc",
                  "conditional_novel_acc", "novel_only_nmi", "novel_only_ari",
                  "novel_routing_recall", "predicted_novel_count",
                  "novel_count_abs_error"):
            v = [float(r[k]) for r in vals]
            out[k] = {"mean": statistics.mean(v),
                      "std": statistics.stdev(v) if len(v) > 1 else 0.0}
        return out
    summary = []
    for name in ("R0", "R1", "R2", "R3", "R4"):
        for proto in ("pure", "ov_assisted"):
            for subset in SUBSETS:
                a = agg(name, proto, subset)
                if not a:
                    continue
                row = {"router": name, "protocol": proto, "subset": subset}
                for k, v in a.items():
                    row[f"{k}_mean"] = v["mean"]; row[f"{k}_std"] = v["std"]
                summary.append(row)
    write_csv(OUT / "corrected_router_summary.csv", summary)
    # corrected train-only selection: mean outer H-mean with infeasible=0
    feas = list(csv.DictReader(open(OUT / "fold_feasibility.csv")))
    old_outer = list(csv.DictReader(open(OLD / "metrics/proxy_outer_folds.csv")))
    r0_hm = statistics.mean(float(r["hmean"]) for r in old_outer if r["router"] == "R0")
    hm = {}
    for name in ("R1", "R2", "R3", "R4"):
        vals = [float(f["hmean"]) for f in feas if f["method"] == name]
        hm[name] = statistics.mean(vals) if vals else 0.0
    feasible_candidates = [n for n in ("R1", "R2", "R3", "R4") if hm[n] > r0_hm + 1e-9]
    # R1 == R0 is not a new contribution; R2/R3/R4 all below or infeasible
    selected = feasible_candidates[0] if feasible_candidates else "NONE"
    bname = "R2" if selected == "NONE" else selected
    r0 = agg("R0"); b = agg(bname)
    criteria = {
        "route_novel_gain": b["route_aware_novel_acc"]["mean"] - r0["route_aware_novel_acc"]["mean"],
        "routing_recall_gain": b["novel_routing_recall"]["mean"] - r0["novel_routing_recall"]["mean"],
        "known_drop": r0["overall_known_acc"]["mean"] - b["overall_known_acc"]["mean"],
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
        "status": "AUDIT_CORRECTION_YIELDS_PASS" if passed else "AUDIT_CONFIRMED_NO_GAIN",
        "selected_router": selected,
        "selected_router_corrected_hmean": hm.get(selected) if selected != "NONE" else None,
        "r0_outer_hmean": r0_hm,
        "corrected_outer_hmeans": hm,
        "selection_source": "train_only_nested_proxy",
        "criteria": criteria,
        "r0": {k: v["mean"] for k, v in r0.items()},
        "best": {k: v["mean"] for k, v in b.items()},
    }
    (RUNS / "audit_gate.json").write_text(json.dumps(gate, indent=2))
    return gate


def r1_reconstruction():
    """Rerun R1 (saved final params) and evaluate - the only true result."""
    feats_tr, labels = load_train_known("dinov2")
    feats_val = load_mean_features("dinov2", "gt_tracks_mean")
    frames_val = load_frame_dict("gt_tracks_mean")
    meta_val = val_meta()
    protos_all = build_prototypes(feats_tr, labels, set(labels.values()))
    knn_all = np.stack([feats_tr[s] for s in feats_tr])
    r1 = make_router("R1", protos_all, 0.45, 0.0)
    rows = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in SUBSETS:
            for stream in STREAMS:
                srows = load_stream_rows(stream)
                preds = run_router_stream("R1", r1, protos_all, feats_val,
                                          frames_val, meta_val, knn_all, srows, gt)
                ev = TrackOCDEvaluator(gt)
                res = ev.evaluate(preds, subset_ids=subset_ids(proto, subset))
                rows.append({
                    "protocol": proto, "subset": subset, "seed": stream,
                    **{k: res[k] for k in res if k != "hungarian_assignment"},
                })
    write_csv(OUT / "r1_result_reconstruction.csv", rows)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "tests").mkdir(parents=True, exist_ok=True)
    audit_inputs()
    method_registry()
    r1_reconstruction()
    oof, feas, r0_kr = regenerate_oof_scores()
    global pooled_oof_cache
    pooled_oof_cache, r0_floor = pooled_oof_thresholds(oof, r0_kr)
    print("pooled OOF thresholds", pooled_oof_cache, "floor", round(r0_floor, 3), flush=True)
    threshold_aggregation_comparison(feas, oof)
    rows = corrected_full_results(pooled_oof_cache)
    gate = summary_and_gate(rows)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
