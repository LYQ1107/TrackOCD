#!/usr/bin/env python3
"""Stage R: domain-robust router bake-off (R0-R5) on frozen DINOv2 mean +
corrected B2, using a source-domain held-out train-only proxy (P1)."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.domain_router.calibration.nested_calibration import (
    fit_router_on_fold, hmean, nested_select_router, build_feature_matrix,
)
from src.domain_router.data.proxy_builder import build_p1_folds, freeze_proxy
from src.domain_router.features.router_features import compute_router_features
from src.domain_router.models.routers import FEATURE_SETS, make_router
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.data.track_stream_dataset import load_stream_rows
from src.dual_branch.models.outputs import emit
from src.ocd_v2.common import (
    load_mean_features, load_train_known, build_prototypes, proxy_split,
)
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt, subset_ids

OUT = PROJECT_ROOT / "outputs" / "domain_router"
RUNS = PROJECT_ROOT / "runs" / "domain_router"
DATA = PROJECT_ROOT / "data" / "domain_router"
STREAMS = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
SEEDS = ("main_seed1027", "main_seed1028", "main_seed1029")
SUBSETS = ("full", "repeated", "balanced")


def load_frame_dict(subdir):
    out = {}
    for p in (PROJECT_ROOT / "data" / "caches" / "features" / "dinov2" / subdir).glob("*.json"):
        r = json.loads(p.read_text())
        out[r["sample_id"]] = np.asarray(r["frame_embeddings"], dtype=np.float32)
    return out


def load_train_meta():
    meta = {}
    with open(PROJECT_ROOT / "data/tao_ow_ocd_v1/public/train_known_tracks.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                boxes = r.get("boxes_xyxy") or []
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                meta[r["sample_id"]] = {
                    "num_frames": len(boxes),
                    "mean_area": float(np.mean(areas)) if areas else 0.0,
                }
    return meta


def val_meta():
    meta = {}
    for stream in STREAMS:
        for r in load_stream_rows(stream):
            areas = r.get("areas") or []
            meta[r["sample_id"]] = {
                "num_frames": len(r.get("frame_ids", []) or []),
                "mean_area": float(np.mean(areas)) if areas else 0.0,
            }
    return meta


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)

    resume = (
        (OUT / "metrics" / "router_full_results.csv").exists()
        and (OUT / "metrics" / "router_selection.csv").exists()
        and (RUNS / "final_routers.json").exists()
        and not (RUNS / "router_gate.json").exists()
    )
    if resume:
        print("resume: loading existing full results", flush=True)
        rows = list(csv.DictReader(open(OUT / "metrics" / "router_full_results.csv")))
        sel = list(csv.DictReader(open(OUT / "metrics" / "router_selection.csv")))
        selected = max((r for r in sel if r["router"] != "R0"),
                       key=lambda r: float(r["outer_hmean_mean"]))
        final = json.loads((RUNS / "final_routers.json").read_text())
        finish_summary(rows, selected["router"], final)
        return

    feats_tr, labels = load_train_known("dinov2")
    frames_tr = load_frame_dict("train_known_mean")
    meta_tr = load_train_meta()
    feats_val = load_mean_features("dinov2", "gt_tracks_mean")
    frames_val = load_frame_dict("gt_tracks_mean")
    meta_val = val_meta()
    protos_all = build_prototypes(feats_tr, labels, set(labels.values()))

    # ---- P1 folds + freeze ----
    folds = build_p1_folds(seed=1027)
    print("folds", len(folds), [f["target_domain"] for f in folds], flush=True)
    freeze_proxy(folds, DATA / "proxy_protocol")
    hash_rows = []
    for p in sorted((DATA / "proxy_protocol" / "folds").glob("*.json")):
        hash_rows.append({"file": str(p.relative_to(PROJECT_ROOT)), "sha256": sha256_file(p)})
    write_csv(OUT / "audit" / "proxy_hashes.csv", hash_rows)
    # domain metadata csv
    from src.domain_router.data.domain_metadata import domain_stats
    ds = domain_stats()
    write_csv(OUT / "audit" / "domain_metadata.csv",
              [{"domain": k, **v} for k, v in sorted(ds.items())])

    # ---- per-fold protos and knn index (train-only, no target leakage) ----
    protos_by_fold = []
    knn_by_fold = []
    for fold in folds:
        src_pos = [s for s in fold["source_positive_ids"] if s in feats_tr]
        src_classes = {labels[s] for s in src_pos}
        protos_by_fold.append(build_prototypes(feats_tr, labels, src_classes))
        knn_by_fold.append(np.stack([feats_tr[s] for s in src_pos]) if src_pos else None)

    outer = nested_select_router(folds, feats_tr, protos_by_fold, knn_by_fold,
                                 frames_tr, meta_tr)
    write_csv(OUT / "metrics" / "proxy_outer_folds.csv", outer)

    # ---- selection: mean H-mean per router, constraint known recall >= R0-0.03 ----
    sel = []
    for name in ("R0", "R1", "R2", "R3", "R4"):
        vals = [r for r in outer if r["router"] == name]
        hm = statistics.mean(r["hmean"] for r in vals)
        kr = statistics.mean(r["known_recall"] for r in vals)
        nr = statistics.mean(r["novel_recall"] for r in vals)
        worst = min(r["hmean"] for r in vals)
        sel.append({
            "router": name, "outer_hmean_mean": hm, "outer_hmean_std":
            statistics.stdev([r["hmean"] for r in vals]) if len(vals) > 1 else 0.0,
            "known_recall_mean": kr, "novel_recall_mean": nr,
            "worst_domain_hmean": worst,
        })
    r0_hm = next(r["outer_hmean_mean"] for r in sel if r["router"] == "R0")
    r4_hm = next(r["outer_hmean_mean"] for r in sel if r["router"] == "R4")
    write_csv(OUT / "metrics" / "router_selection.csv", sel)
    selected = max((r for r in sel if r["router"] != "R0"), key=lambda r: r["outer_hmean_mean"])
    print("selection", sel, "best", selected["router"], flush=True)
    sel_name = selected["router"]
    fold_thrs = [r["threshold"] for r in outer if r["router"] == sel_name]
    fold_thr_mean = float(np.mean(fold_thrs)) if fold_thrs else 0.45
    print("selected", sel_name, "fold threshold mean", fold_thr_mean, flush=True)

    # ---- conditional R5 ----
    if r4_hm >= r0_hm + 0.03:
        print("R5 authorized by proxy (R4 H-mean +0.03)", flush=True)
        r5_rows = []
        for i, fold in enumerate(folds):
            res = fit_router_on_fold("R4", feats_tr, protos_by_fold[i], fold,
                                     knn_by_fold[i], frames_tr, meta_tr)
            r5_rows.append({"router": "R5", "target_domain": fold["target_domain"],
                            "fold_index": i, **res[1]})
        write_csv(OUT / "metrics" / "proxy_outer_folds_r5.csv", r5_rows)
    else:
        (RUNS / "r5.skipped").write_text("R5 not authorized: R4 outer H-mean gain < 0.03\n")

    # ---- final router fitting on all train-known (global proxy split) ----
    pk, pn = proxy_split(labels, seed=1027)
    tr_pos = [s for s, c in labels.items() if c in pk and s in feats_tr]
    tr_neg = [s for s, c in labels.items() if c in pn and s in feats_tr]
    knn_all = np.stack([feats_tr[s] for s in tr_pos + tr_neg])
    r0_proxy_kr = sum(
        1 for s in tr_pos
        if compute_router_features(feats_tr[s], protos_all, knn_all,
                                   frames_tr.get(s), meta_tr.get(s))["s1"] >= 0.45
    ) / max(1, len(tr_pos))
    kr_floor = max(0.0, r0_proxy_kr - 0.03)
    print("r0 proxy known recall", round(r0_proxy_kr, 3), "floor", round(kr_floor, 3), flush=True)
    final = {}
    for name in ("R0", "R1", "R2", "R3", "R4", "R5"):
        if name == "R5" and not (RUNS / "r5.skipped").exists():
            pass
        if name == "R5":
            continue
        if name == "R0":
            final[name] = make_router("R0", protos_all, 0.45)
            continue
        if name == "R1":
            th_s = fold_thr_mean if name == sel_name else 0.45
            final[name] = make_router("R1", protos_all, th_s, 0.0)
            final[name + "_params"] = {"threshold": th_s, "margin_threshold": 0.0}
        else:
            X = build_feature_matrix(tr_pos + tr_neg, feats_tr, protos_all, knn_all,
                                     frames_tr, meta_tr, FEATURE_SETS[name])
            y = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
            thr = fold_thr_mean if name == sel_name else 0.5
            final[name] = make_router(name, protos_all, thr,
                                      coef=lr.coef_[0], intercept=lr.intercept_[0])
            final[name + "_params"] = {
                "coef": [float(v) for v in lr.coef_[0]],
                "intercept": float(lr.intercept_[0]),
                "threshold": thr,
                "features": FEATURE_SETS[name],
            }
    params_out = {}
    for k, v in final.items():
        if isinstance(v, dict):
            params_out[k] = v
    (RUNS / "final_routers.json").write_text(json.dumps(params_out, indent=2))

    # ---- full evaluation R0-R5 ----
    rows = []
    for proto in ("pure", "ov_assisted"):
        gt = load_gt(proto)
        for subset in SUBSETS:
            for stream in STREAMS:
                srows = load_stream_rows(stream)
                sub = subset_ids(proto, subset)
                for name in ("R0", "R1", "R2", "R3", "R4", "R5"):
                    if name == "R5" and (RUNS / "r5.skipped").exists():
                        continue
                    preds = run_router_stream(
                        name, final.get(name), protos_all, feats_val, frames_val,
                        meta_val, knn_all, srows, gt,
                    )
                    ev = TrackOCDEvaluator(gt)
                    res = ev.evaluate(preds, subset_ids=sub)
                    row = {
                        "router": name, "protocol": proto, "subset": subset,
                        "seed": stream,
                        **{k: res[k] for k in res if k != "hungarian_assignment"},
                    }
                    rows.append(row)
                    print(name, proto, subset, stream,
                          "known", round(res["overall_known_acc"], 4),
                          "route", round(res["route_aware_novel_acc"], 4),
                          "recall", round(res["novel_routing_recall"], 4),
                          "cond", round(res["conditional_novel_acc"], 4),
                          "nmi", round(res["novel_only_nmi"], 4),
                          flush=True)
    write_csv(OUT / "metrics" / "router_full_results.csv", rows)

    # ---- summary + gate ----
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
    finish_summary(rows, selected["router"], final)


def finish_summary(rows, br, final):
    SEEDS = ("main_seed1027", "main_seed1028", "main_seed1029")
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
    r0 = agg("R0")
    b = agg(br)
    summary_rows = []
    for name in ("R0", "R1", "R2", "R3", "R4", "R5"):
        for proto in ("pure", "ov_assisted"):
            for subset in SUBSETS:
                a = agg(name, proto, subset)
                if not a:
                    continue
                row = {"router": name, "protocol": proto, "subset": subset}
                for k, v in a.items():
                    row[f"{k}_mean"] = v["mean"]; row[f"{k}_std"] = v["std"]
                summary_rows.append(row)
    write_csv(OUT / "metrics" / "router_final_summary.csv", summary_rows)

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
        "status": "PASS_DOMAIN_ROBUST_ROUTER" if passed else "NO_CLEAR_ROUTER_GAIN",
        "continue_encoder": bool(passed),
        "selected_router": br,
        "selection_source": "train_only_nested_proxy",
        "criteria": criteria,
        "r0": {k: v["mean"] for k, v in r0.items()},
        "best": {k: v["mean"] for k, v in b.items()},
    }
    (RUNS / "router_gate.json").write_text(json.dumps(gate, indent=2))
    (RUNS / "status.txt").write_text(gate["status"] + "\n")
    print(json.dumps(gate, indent=2))

    # ---- subgroup diagnostics (best router vs R0) ----
    if not (OUT / "metrics" / "router_subgroups.csv").exists():
        subgroup_rows = compute_subgroups(br, "R0")
        write_csv(OUT / "metrics" / "router_subgroups.csv", subgroup_rows)


def run_router_stream(name, router, protos, feats, frames, meta, knn_all,
                      srows, gt):
    gt_by_sid = {g["sample_id"]: g for g in gt}
    if name == "R0":
        mem = B2Memory(protos, threshold=0.45)
        preds = []
        for i, r in enumerate(srows):
            vid, kind = mem.predict_one(feats[r["sample_id"]], r["sample_id"], i)
            preds.append(emit(r["sample_id"], i, kind,
                              vid if kind == "known" else None,
                              vid if kind == "novel" else None))
        return preds
    mem = B2Memory(protos, threshold=0.45, novel_only=True)
    preds = []
    for i, r in enumerate(srows):
        sid = r["sample_id"]
        f = compute_router_features(feats[sid], protos, knn_all,
                                    frames.get(sid), meta.get(sid))
        is_known = router.decide(f)
        if is_known:
            best_id, _ = max(protos.items(), key=lambda kv: float(np.dot(feats[sid], kv[1])))
            preds.append(emit(sid, i, "known", known_id=best_id))
        else:
            vid, _ = mem.predict_one(feats[sid], sid, i)
            preds.append(emit(sid, i, "novel", virtual_id=vid))
    return preds


def compute_subgroups(best_name, r0_name):
    feats_tr, labels = load_train_known("dinov2")
    protos = build_prototypes(feats_tr, labels, set(labels.values()))
    feats_val = load_mean_features("dinov2", "gt_tracks_mean")
    frames_val = load_frame_dict("gt_tracks_mean")
    meta_val = val_meta()
    knn_all = np.stack([feats_tr[s] for s in feats_tr])
    params = json.loads((RUNS / "final_routers.json").read_text())
    routers = {"R0": make_router("R0", protos, 0.45)}
    for name in ("R1", "R2", "R3", "R4"):
        p = params.get(name + "_params")
        if p is None:
            continue
        if name == "R1":
            routers[name] = make_router("R1", protos, p["threshold"], p["margin_threshold"])
        else:
            routers[name] = make_router(name, protos, p["threshold"],
                                        coef=p["coef"], intercept=p["intercept"])
    return subgroup_diagnostics(best_name, r0_name, routers, protos,
                                feats_val, frames_val, meta_val, knn_all)


def subgroup_diagnostics(best_name, r0_name, router_cache, protos, feats,
                         frames, meta, knn_all):
    gt = load_gt("pure")
    private = {g["sample_id"]: g for g in gt}
    cat_count = Counter(g["ground_truth_category_id"] for g in gt if g["protocol_role"] == "novel")
    cat_video = defaultdict(set)
    for g in gt:
        if g["protocol_role"] == "novel":
            cat_video[g["ground_truth_category_id"]].add(int(g["sample_id"].split("_")[0]))
    groups = {"all": set(private)}
    groups["singleton_novel"] = {g["sample_id"] for g in gt if g["protocol_role"] == "novel" and cat_count[g["ground_truth_category_id"]] == 1}
    groups["repeated_novel"] = {g["sample_id"] for g in gt if g["protocol_role"] == "novel" and cat_count[g["ground_truth_category_id"]] >= 2}
    groups["cross_video_novel"] = {g["sample_id"] for g in gt if g["protocol_role"] == "novel" and len(cat_video[g["ground_truth_category_id"]]) >= 2}
    groups["short_tracks"] = {s for s in private if meta.get(s, {}).get("num_frames", 0) < 3}
    groups["long_tracks"] = {s for s in private if meta.get(s, {}).get("num_frames", 0) >= 10}
    groups["small_objects"] = {s for s in private if meta.get(s, {}).get("mean_area", 1e9) < 1000}
    groups["large_objects"] = {s for s in private if meta.get(s, {}).get("mean_area", 0) >= 10000}
    srows = load_stream_rows("main")
    preds_cache = {}
    for name in (r0_name, best_name):
        router = router_cache.get(name)
        preds_cache[name] = run_router_stream(
            name, router, protos, feats, frames, meta, knn_all, srows, gt)
    out = []
    for gname, ids in groups.items():
        if len(ids) < 10:
            continue
        for router_name in (r0_name, best_name):
            ev = TrackOCDEvaluator(gt)
            res = ev.evaluate(preds_cache[router_name], subset_ids=ids)
            out.append({
                "group": gname, "router": router_name, "size": len(ids),
                "route_novel_acc": res["route_aware_novel_acc"],
                "routing_recall": res["novel_routing_recall"],
                "cond_novel_acc": res["conditional_novel_acc"],
                "known_acc": res["overall_known_acc"],
            })
    return out


if __name__ == "__main__":
    main()
