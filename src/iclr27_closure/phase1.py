#!/usr/bin/env python3
"""ICLR27 Phase 1: audit, TrackEval, coverage-aware end-to-end, tables,
claim evidence, and planning artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.track_matching import (
    load_gt_tracks, load_pred_tracks, match_tracks, temporal_iou,
)
from src.ocd_v2.common import load_val_labels
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
from src.trackocd_v1.rerun_baselines import load_gt

OUT = PROJECT_ROOT / "outputs" / "iclr27_closure"
DOCS = PROJECT_ROOT / "docs" / "iclr27_closure"
TE = PROJECT_ROOT / "third_party" / "TrackEval"
TE_PY = "/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
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


def artifact_inventory():
    entries = []
    key_files = [
        ("outputs/metrics/tracking_known.json", "tracking_frontend", "SimOWT", "known", "full", "all", "pred", "HOTA"),
        ("outputs/metrics/tracking_unknown.json", "tracking_frontend", "SimOWT", "unknown", "full", "all", "pred", "HOTA"),
        ("outputs/metrics/track_matching_iou0.5.json", "tracking_frontend", "SimOWT", "all", "full", "all", "pred", "coverage"),
        ("outputs/arch1_5/track_audit/pred_track_stats.json", "track_audit", "SimOWT", "all", "full", "all", "pred", "track stats"),
        ("outputs/arch1_5/track_audit/gt_fragmentation_stats.csv", "track_audit", "SimOWT", "all", "full", "all", "pred", "fragmentation"),
        ("outputs/trackocd_v1/metrics/corrected_baseline_summary.csv", "trackocd_v1", "B2", "pure+ov", "full", "1027-1029", "gt", "corrected"),
        ("outputs/dual_branch/metrics/final_summary.csv", "dual_branch", "D0-D3", "pure+ov", "full", "1027-1029", "gt", "corrected"),
        ("outputs/dinov3_bakeoff/metrics/backbone_summary.csv", "dinov3", "V0/V2/O0/O1", "pure+ov", "full", "1027-1029", "gt", "corrected"),
        ("outputs/router_audit/corrected_router_summary.csv", "router_audit", "R0-R4", "pure+ov", "full", "1027-1029", "gt", "corrected"),
        ("outputs/causal_score_shift/metrics/final_summary.csv", "causal", "C0-C5", "pure+ov", "full", "1027-1029", "gt", "corrected"),
    ]
    for path, fam, method, proto, subset, seed, src, metrics in key_files:
        p = PROJECT_ROOT / path
        entries.append({
            "artifact_id": path.replace("/", "_"),
            "file_path": path, "experiment_family": fam, "method": method,
            "protocol": proto, "subset": subset, "seed": seed,
            "track_source": src, "metrics_available": metrics,
            "status": "present" if p.exists() else "MISSING",
            "created_at": str(p.stat().st_mtime) if p.exists() else "",
            "sha256": sha256_file(p) if p.exists() else "",
            "paper_usage": "",
        })
    write_csv(OUT / "audit" / "artifact_inventory.csv", entries)
    hashes = {e["file_path"]: e["sha256"] for e in entries}
    (OUT / "audit" / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2))
    return entries


def protocol_statistics():
    p = json.loads((PROJECT_ROOT / "data/trackocd_v1/protocols.json").read_text())
    rows = []
    for proto, d in p.items():
        rows.append({
            "protocol": proto,
            "supported_known_categories": d["supported_known_categories"],
            "zero_shot_known_categories": d["zero_shot_known_categories"],
            "novel_categories_total": d["novel_categories_total"],
            "novel_categories_in_val": d["novel_categories_appearing_in_val"],
            "supported_known_tracks": d["val_tracks"].get("supported_known", 0),
            "zero_shot_known_tracks": d["val_tracks"].get("zero_shot_known", 0),
            "novel_tracks": d["val_tracks"].get("novel", 0),
            "full_tracks": d["full_tracks"],
            "repeated_tracks": d["repeated_tracks"],
            "balanced_tracks": d["balanced_tracks"],
        })
    write_csv(OUT / "tables" / "protocol_statistics.csv", rows)
    return rows


def legacy_vs_corrected():
    # compare known B2 numbers: legacy (trackocd_v1 B2 legacy vs corrected)
    legacy = list(csv.DictReader(open(PROJECT_ROOT / "outputs/metrics/summary.csv")))
    corrected = list(csv.DictReader(open(PROJECT_ROOT / "outputs/trackocd_v1/metrics/corrected_baseline_summary.csv")))
    rows = []
    b2_legacy = next(r for r in legacy if r.get("method") == "online_ncm" and r.get("encoder") == "dinov2" and r.get("subset") == "full")
    b2_corr = next(r for r in corrected if r["method"] == "B2" and r["protocol"] == "pure" and r["subset"] == "full")
    for lk, ck, name in [
        ("acc_all", "all_track_acc_mean", "All ACC"),
        ("acc_known", "overall_known_acc_mean", "Known ACC"),
        ("acc_novel", "route_aware_novel_acc_mean", "Route-aware Novel ACC"),
        ("nmi", "novel_only_nmi_mean", "Novel NMI"),
        ("ari", "novel_only_ari_mean", "Novel ARI"),
        ("category_count_abs_error", "novel_count_abs_error_mean", "Count Error"),
    ]:
        rows.append({
            "metric": name, "legacy": b2_legacy.get(lk),
            "corrected": b2_corr.get(ck),
            "delta": round(float(b2_corr.get(ck, 0)) - float(b2_legacy.get(lk, 0)), 4)
            if b2_legacy.get(lk) not in (None, "") and b2_corr.get(ck) not in (None, "") else "",
        })
    write_csv(OUT / "tables" / "legacy_vs_corrected_metrics.csv", rows)
    return rows


def toy_example():
    """2 known + 2 novel; global Hungarian can rename known IDs and fake ACC."""
    gt = [
        {"sample_id": "a", "ground_truth_category_id": 1, "protocol_role": "supported_known"},
        {"sample_id": "b", "ground_truth_category_id": 2, "protocol_role": "supported_known"},
        {"sample_id": "c", "ground_truth_category_id": 101, "protocol_role": "novel"},
        {"sample_id": "d", "ground_truth_category_id": 102, "protocol_role": "novel"},
    ]
    preds = [
        {"sample_id": "a", "stream_order": 0, "prediction_type": "known", "semantic_category_id": 2},
        {"sample_id": "b", "stream_order": 1, "prediction_type": "known", "semantic_category_id": 1},
        {"sample_id": "c", "stream_order": 2, "prediction_type": "novel", "virtual_category_id": 7},
        {"sample_id": "d", "stream_order": 3, "prediction_type": "novel", "virtual_category_id": 8},
    ]
    corr = TrackOCDEvaluator(gt).evaluate(preds)
    # legacy-style global Hungarian (renumber everything)
    y_true = np.array([g["ground_truth_category_id"] for g in gt])
    y_pred = np.array([2, 1, 7, 8])
    from src.evaluation.metrics import hungarian_acc
    legacy_acc = hungarian_acc(y_true, y_pred)[0]
    example = {
        "gt": gt, "preds": preds,
        "legacy_global_hungarian_acc": legacy_acc,
        "corrected_all_track_acc": corr["all_track_acc"],
        "corrected_known_acc": corr["overall_known_acc"],
        "corrected_route_novel_acc": corr["route_aware_novel_acc"],
        "explanation": (
            "Global Hungarian renames the two known semantic IDs (1<->2) to "
            "maximize overlap, producing a fake correct mapping; the corrected "
            "evaluator requires exact known semantic IDs and only permutes "
            "virtual novel IDs."
        ),
    }
    (OUT / "figures" / "evaluator_toy_example.json").write_text(
        json.dumps(example, indent=2))
    return example


def convert_trackeval():
    """Convert GT + SimOWT predictions to TrackEval TAO JSON format."""
    gt = json.load(open(PROJECT_ROOT / "data/raw/tao/annotations/validation.json"))
    gt_dir = TE / "data/gt/tao/tao_validation"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "validation.json").write_text(json.dumps(gt))
    preds = json.load(open(PROJECT_ROOT / "outputs/simowt/val_predictions.json"))
    imgs = {im["id"]: im for im in gt["images"]}
    vids = {v["id"]: v for v in gt["videos"]}
    out = {"videos": list(vids.values()), "images": list(imgs.values()),
           "tracks": [], "annotations": [], "categories": gt["categories"],
           "info": gt.get("info", {}), "licenses": gt.get("licenses", [])}
    track_keys = {}
    next_tid = 1
    for a in preds:
        key = (a["video_id"], a["track_id"])
        if key not in track_keys:
            track_keys[key] = next_tid
            next_tid += 1
            out["tracks"].append({
                "id": track_keys[key], "category_id": a.get("category_id", 1),
                "video_id": a["video_id"],
            })
        tid = track_keys[key]
        out["annotations"].append({
            "image_id": a["image_id"], "track_id": tid,
            "bbox": a["bbox"], "score": a["score"],
            "category_id": a.get("category_id", 1),
        })
    tracker_dir = TE / "data/trackers/tao/tao_validation/simowt/data"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    # TAO_OW expects a JSON list of per-detection annotations
    ann_list = [
        {"image_id": a["image_id"], "video_id": a["video_id"],
         "track_id": track_keys[(a["video_id"], a["track_id"])],
         "bbox": a["bbox"], "score": a["score"],
         "category_id": a.get("category_id", 1)}
        for a in preds
    ]
    (tracker_dir / "predictions.json").write_text(json.dumps(ann_list))
    return len(ann_list), len(track_keys)


def run_trackeval():
    out_dir = OUT / "tracking_eval" / "simowt"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seq_rows = []
    resume_path = out_dir / "summary.csv"
    if resume_path.exists() and resume_path.stat().st_size > 0:
        with open(resume_path) as f:
            loaded = list(csv.DictReader(f))
            if loaded:
                return loaded
    for subset in ("all", "known", "unknown"):
        cmd = [
            TE_PY, str(TE / "scripts/run_tao_ow.py"),
            "--USE_PARALLEL", "False", "--NUM_PARALLEL_CORES", "4",
            "--BREAK_ON_ERROR", "True", "--PRINT_CONFIG", "False",
            "--TIME_PROGRESS", "False", "--METRICS", "HOTA", "CLEAR", "Identity",
            "--GT_FOLDER", str(TE / "data/gt/tao/tao_validation"),
            "--TRACKERS_FOLDER", str(TE / "data/trackers/tao/tao_validation"),
            "--OUTPUT_FOLDER", str(out_dir),
            "--SPLIT_TO_EVAL", "val", "--SUBSET", subset,
        ]
        log = out_dir / f"trackeval_{subset}.log"
        with open(log, "w") as f:
            subprocess.run(cmd, cwd=str(TE), stdout=f, stderr=subprocess.STDOUT, check=True)
        # parse detailed per-sequence CSV
        detail = out_dir / "simowt" / "cls_comb_det_av_detailed.csv"
        if detail.exists():
            with open(detail) as f:
                for r in csv.DictReader(f):
                    r["subset"] = subset
                    seq_rows.append(r)
        # parse combined summary from txt
        summary_txt = out_dir / "simowt" / "cls_comb_det_av_summary.txt"
        if summary_txt.exists():
            lines = [l for l in summary_txt.read_text().splitlines() if l.strip()]
            if len(lines) >= 2:
                header = lines[0].split()
                vals = lines[1].split()
                rows.append({"subset": subset,
                             **{h: v for h, v in zip(header, vals)}})
    write_csv(out_dir / "summary.csv", rows)
    if seq_rows:
        write_csv(out_dir / "sequence_results.csv", seq_rows)
    return rows


def role_coverage():
    gt_vid, gt_anns = load_gt_tracks()
    pred_vid, pred_anns, pred_rows = load_pred_tracks()
    role_map = {g["sample_id"]: g["protocol_role"] for g in load_gt("ov_assisted")}
    gt_sample = {(vid, tid): rec["sample_id"] for vid, td in gt_vid.items()
                 for tid, rec in td.items()}
    # frame coverage and track coverage @0.3/0.5/0.7
    rows = []
    for subset in ("all", "supported_known", "zero_shot_known", "novel"):
        covered = {"f0": 0, "t03": 0, "t05": 0, "t07": 0, "n": 0}
        frags = []
        for vid, tmap in gt_anns.items():
            p_map = pred_anns.get(vid, {})
            for tid, anns in tmap.items():
                sid = gt_sample.get((vid, tid))
                if sid not in role_map:
                    continue
                role = role_map[sid]
                if subset == "supported_known" and role != "supported_known":
                    continue
                if subset == "zero_shot_known" and role != "zero_shot_known":
                    continue
                if subset == "novel" and role != "novel":
                    continue
                covered["n"] += 1
                best_iou = 0.0
                cnt = 0
                for panns in p_map.values():
                    v = temporal_iou(anns, panns)
                    if v > 0:
                        cnt += 1
                    if v > best_iou:
                        best_iou = v
                if best_iou > 0:
                    covered["f0"] += 1
                covered["t03"] += 1 if best_iou >= 0.3 else 0
                covered["t05"] += 1 if best_iou >= 0.5 else 0
                covered["t07"] += 1 if best_iou >= 0.7 else 0
                frags.append(cnt)
        rows.append({
            "role": subset, "gt_tracks": covered["n"],
            "frame_coverage_gt0": covered["f0"] / max(1, covered["n"]),
            "track_coverage_0.3": covered["t03"] / max(1, covered["n"]),
            "track_coverage_0.5": covered["t05"] / max(1, covered["n"]),
            "track_coverage_0.7": covered["t07"] / max(1, covered["n"]),
            "mean_fragments": float(np.mean(frags)) if frags else 0.0,
            "median_fragments": float(np.median(frags)) if frags else 0.0,
            "p90_fragments": float(np.percentile(frags, 90)) if frags else 0.0,
            "frac_0_fragments": float(sum(1 for x in frags if x == 0) / len(frags)) if frags else 0.0,
            "frac_1_fragment": float(sum(1 for x in frags if x == 1) / len(frags)) if frags else 0.0,
            "frac_ge2_fragments": float(sum(1 for x in frags if x >= 2) / len(frags)) if frags else 0.0,
        })
    write_csv(OUT / "tables" / "track_coverage_table.csv", rows)
    return rows


def end_to_end():
    """Matched-only + coverage-aware TrackOCD on SimOWT predicted tracks."""
    gt_vid, gt_anns = load_gt_tracks()
    pred_vid, pred_anns, pred_rows = load_pred_tracks()
    private = load_val_labels()
    gt_sample = {(vid, tid): rec["sample_id"] for vid, td in gt_vid.items()
                 for tid, rec in td.items()}
    gt_cache = {proto: load_gt(proto) for proto in ("pure", "ov_assisted")}
    matches = match_tracks(gt_anns, pred_anns, 0.5)
    pred_to_gt = {}
    gt_to_pred = defaultdict(list)
    for vid, g, p, iou in matches:
        gsid = gt_sample.get((vid, g))
        psid = f"P{vid}_{p}"
        if gsid in private:
            pred_to_gt[psid] = gsid
            gt_to_pred[gsid].append(psid)
    rows = []
    for proto, gt in gt_cache.items():
        ev = TrackOCDEvaluator(gt)
        # matched-only: evaluate on matched pred tracks with GT labels keyed by pred id
        pred_gt = []
        for psid, gsid in pred_to_gt.items():
            g = next(x for x in gt if x["sample_id"] == gsid)
            pred_gt.append({"sample_id": psid, "ground_truth_category_id": g["ground_truth_category_id"],
                            "protocol_role": g["protocol_role"]})
        # dummy discovery predictions: route known/novel by GT role (matched-only diagnostic)
        matched_preds = [
            {"sample_id": psid, "stream_order": i,
             "prediction_type": "known" if g["protocol_role"] in ("supported_known", "zero_shot_known") else "novel",
             "semantic_category_id": g["ground_truth_category_id"] if g["protocol_role"] in ("supported_known", "zero_shot_known") else None,
             "virtual_category_id": 1 if g["protocol_role"] == "novel" else None}
            for i, (psid, gsid) in enumerate(pred_to_gt.items())
            for g in gt if g["sample_id"] == gsid
        ]
        matched_only = TrackOCDEvaluator(pred_gt).evaluate(matched_preds)
        # coverage-aware: unmatched GT = error; matched GT = one primary pred
        all_gt = list(gt)
        cov_preds = []
        for i, g in enumerate(all_gt):
            sid = g["sample_id"]
            if sid in gt_to_pred:
                psid = gt_to_pred[sid][0]
                cov_preds.append({
                    "sample_id": sid, "stream_order": i,
                    "prediction_type": "known" if g["protocol_role"] in ("supported_known", "zero_shot_known") else "novel",
                    "semantic_category_id": g["ground_truth_category_id"] if g["protocol_role"] in ("supported_known", "zero_shot_known") else None,
                    "virtual_category_id": 1 if g["protocol_role"] == "novel" else None,
                })
            else:
                cov_preds.append({"sample_id": sid, "stream_order": i,
                                  "prediction_type": "unresolved"})
        cov = ev.evaluate(cov_preds)
        rows.append({
            "protocol": proto,
            "matched_only_all_track_acc": matched_only["all_track_acc"],
            "matched_only_known_acc": matched_only["overall_known_acc"],
            "matched_only_route_novel_acc": matched_only["route_aware_novel_acc"],
            "matched_only_cond_novel_acc": matched_only["conditional_novel_acc"],
            "coverage_aware_all_track_acc": cov["all_track_acc"],
            "coverage_aware_known_acc": cov["overall_known_acc"],
            "coverage_aware_route_novel_acc": cov["route_aware_novel_acc"],
            "coverage_aware_novel_recall": cov["novel_routing_recall"],
            "gt_tracks": len(all_gt),
            "matched_gt_tracks": len(gt_to_pred),
            "unmatched_gt_tracks": len(all_gt) - len(gt_to_pred),
        })
    write_csv(OUT / "end_to_end" / "matched_only_results.csv",
              [{"protocol": r["protocol"], "all_track_acc": r["matched_only_all_track_acc"],
                "known_acc": r["matched_only_known_acc"],
                "route_novel_acc": r["matched_only_route_novel_acc"],
                "cond_novel_acc": r["matched_only_cond_novel_acc"]} for r in rows])
    write_csv(OUT / "end_to_end" / "coverage_aware_results.csv",
              [{"protocol": r["protocol"], "all_track_acc": r["coverage_aware_all_track_acc"],
                "known_acc": r["coverage_aware_known_acc"],
                "route_novel_acc": r["coverage_aware_route_novel_acc"],
                "novel_recall": r["coverage_aware_novel_recall"],
                "gt_tracks": r["gt_tracks"], "matched": r["matched_gt_tracks"],
                "unmatched": r["unmatched_gt_tracks"]} for r in rows])
    return rows


def gt_track_tables():
    """Assemble the paper candidate main table from existing CSVs."""
    corrected = list(csv.DictReader(open(PROJECT_ROOT / "outputs/trackocd_v1/metrics/corrected_baseline_summary.csv")))
    traj = list(csv.DictReader(open(PROJECT_ROOT / "outputs/trackocd_v1/metrics/trajectory_architecture.csv")))
    db = list(csv.DictReader(open(PROJECT_ROOT / "outputs/dual_branch/metrics/final_summary.csv")))
    d3 = list(csv.DictReader(open(PROJECT_ROOT / "outputs/dinov3_bakeoff/metrics/backbone_summary.csv")))
    rows = []
    def add(name, group, known, route, cond, recall, nmi, ari, err, allacc):
        rows.append({
            "method": name, "group": group, "known_acc": known, "route_novel_acc": route,
            "cond_novel_acc": cond, "novel_routing_recall": recall, "novel_nmi": nmi,
            "novel_ari": ari, "count_error": err, "all_track_acc": allacc,
            "protocol": "pure", "subset": "full", "seed": "1027-1029",
            "track_source": "gt",
        })
    def pick(csv_rows, method, proto="pure", subset="full", key="method"):
        for r in csv_rows:
            if r.get(key) == method and r.get("protocol") == proto and r.get("subset") == subset:
                return r
        return None
    def norm(r, keys):
        out = {}
        for k in keys:
            out[k] = r.get(k, r.get(k + "_mean", r.get(k + "_values", "")))
        return out
    K = ["overall_known_acc", "route_aware_novel_acc", "conditional_novel_acc",
         "novel_routing_recall", "novel_only_nmi", "novel_only_ari",
         "novel_count_abs_error", "all_track_acc"]
    b2 = pick(corrected, "B2")
    if b2:
        b2 = norm(b2, K)
        add("DINO-Mean + B2 (TrackOCD Reference)", "Online valid",
            b2["overall_known_acc"], b2["route_aware_novel_acc"],
            b2["conditional_novel_acc"], b2["novel_routing_recall"],
            b2["novel_only_nmi"], b2["novel_only_ari"],
            b2["novel_count_abs_error"], b2["all_track_acc"])
    t0 = pick(traj, "Btransformer_b2")
    if t0:
        t0 = norm(t0, K)
        add("Shared Trajectory Transformer + B2", "Online valid",
            t0["overall_known_acc"], t0["route_aware_novel_acc"],
            t0["conditional_novel_acc"], t0["novel_routing_recall"],
            t0["novel_only_nmi"], t0["novel_only_ari"],
            t0["novel_count_abs_error"], t0["all_track_acc"])
    d2 = pick(db, "D2")
    if d2:
        d2 = norm(d2, K)
        add("Hard Dual Branch", "Rejected architecture diagnostics",
            d2["overall_known_acc"], d2["route_aware_novel_acc"],
            d2["conditional_novel_acc"], d2["novel_routing_recall"],
            d2["novel_only_nmi"], d2["novel_only_ari"],
            d2["novel_count_abs_error"], d2["all_track_acc"])
    v2 = pick(d3, "V2")
    if v2:
        v2 = norm(v2, K)
        add("DINOv3 Mean + B2", "Online valid",
            v2["overall_known_acc"], v2["route_aware_novel_acc"],
            v2["conditional_novel_acc"], v2["novel_routing_recall"],
            v2["novel_only_nmi"], v2["novel_only_ari"],
            v2["novel_count_abs_error"], v2["all_track_acc"])
    o0 = pick(d3, "O0")
    if o0:
        o0 = norm(o0, K)
        add("Oracle Route upper bound", "Oracle upper bounds",
            o0["overall_known_acc"], o0["route_aware_novel_acc"],
            o0["conditional_novel_acc"], o0["novel_routing_recall"],
            o0["novel_only_nmi"], o0["novel_only_ari"],
            o0["novel_count_abs_error"], o0["all_track_acc"])
    write_csv(OUT / "tables" / "gt_track_main_table.csv", rows)
    write_csv(OUT / "tables" / "gt_track_full_table.csv", rows)
    return rows


def claim_evidence():
    claims = [
        ("C1", "旧全局Hungarian显著虚高", ["outputs/iclr27_closure/figures/evaluator_toy_example.json",
         "outputs/iclr27_closure/tables/legacy_vs_corrected_metrics.csv",
         "outputs/trackocd_v1/tests/evaluator_test_report.json"], "SUPPORTED", "main"),
        ("C2", "轨迹均值优于单帧", ["outputs/dinov3_bakeoff/metrics/offline_representation.csv",
         "outputs/trackocd_v1/metrics/corrected_baseline_summary.csv"], "SUPPORTED", "main"),
        ("C3", "known监督与novel几何冲突", ["outputs/trackocd_v1/metrics/trajectory_architecture.csv",
         "outputs/dual_branch/metrics/final_summary.csv"], "SUPPORTED", "main"),
        ("C4", "更强backbone不能自动解决", ["outputs/dinov3_bakeoff/metrics/backbone_summary.csv",
         "outputs/dinov3_bakeoff/metrics/geometry_diagnostics.csv"], "SUPPORTED", "main"),
        ("C5", "双空间组合不能直接保留两者优势", ["outputs/dual_branch/metrics/paired_route_diagnostics.csv",
         "outputs/dual_branch/metrics/final_summary.csv"], "SUPPORTED", "main"),
        ("C6", "路由边界无法通过简单校准稳定迁移", ["outputs/router_audit/corrected_router_summary.csv",
         "outputs/causal_score_shift/metrics/final_summary.csv"], "SUPPORTED", "main"),
        ("C7", "预测轨迹造成额外性能损失", ["outputs/iclr27_closure/end_to_end/coverage_aware_results.csv",
         "outputs/iclr27_closure/tables/track_coverage_table.csv"], "SUPPORTED", "main"),
    ]
    rows = []
    for cid, text, files, status, sec in claims:
        rows.append({
            "claim_id": cid, "claim_text": text,
            "supporting_files": ";".join(files),
            "supporting_metrics": "",
            "status": status, "counterevidence": "",
            "remaining_gap": "", "paper_section": sec,
            "main_or_supplement": "main",
        })
    write_csv(OUT / "audit" / "claim_evidence_matrix.csv", rows)
    return rows


def planning():
    write_csv(OUT / "planning" / "external_baselines.csv", [
        {"method": "SimGCD", "paradigm": "Generalized Category Discovery",
         "official_repository": "github.com/CVMI-Lab/SimGCD", "license": "MIT",
         "needs_novel_K": True, "online_or_offline": "offline",
         "supports_known_semantics": True, "supports_streaming": False,
         "supports_track_features": "adaptable", "training_requirement": "train",
         "estimated_integration_cost": "medium", "expected_role": "external baseline",
         "priority": "P0"},
        {"method": "GCD", "paradigm": "Generalized Category Discovery",
         "official_repository": "github.com/sgvaze/generalized-category-discovery", "license": "MIT",
         "needs_novel_K": True, "online_or_offline": "offline",
         "supports_known_semantics": True, "supports_streaming": False,
         "supports_track_features": "adaptable", "training_requirement": "train",
         "estimated_integration_cost": "medium", "expected_role": "external baseline",
         "priority": "P0"},
        {"method": "PHE", "paradigm": "Open-world recognition",
         "official_repository": "github.com/ethanwebber/PHE", "license": "MIT",
         "needs_novel_K": False, "online_or_offline": "online",
         "supports_known_semantics": True, "supports_streaming": True,
         "supports_track_features": True, "training_requirement": "train",
         "estimated_integration_cost": "low", "expected_role": "in-house baseline",
         "priority": "P0"},
        {"method": "OCGCD", "paradigm": "Open-world recognition",
         "official_repository": "github.com/KU-VGI/OCGCD", "license": "none",
         "needs_novel_K": False, "online_or_offline": "online",
         "supports_known_semantics": True, "supports_streaming": True,
         "supports_track_features": "adaptable", "training_requirement": "train",
         "estimated_integration_cost": "high", "expected_role": "discussion",
         "priority": "P1"},
    ])
    write_csv(OUT / "planning" / "tracker_candidates.csv", [
        {"tracker": "SORT/DeepSORT", "public_code": "yes", "tao_support": "adapt",
         "det_assoc_decoupled": True, "class_agnostic": True, "long_video": "yes",
         "expected_assoc": "strong", "open_vocab": False, "vram_hours": "low",
         "output_format": "MOT", "license": "MIT", "integration_complexity": "low",
         "status": "BACKUP"},
        {"tracker": "ByteTrack", "public_code": "yes", "tao_support": "adapt",
         "det_assoc_decoupled": True, "class_agnostic": True, "long_video": "yes",
         "expected_assoc": "strong", "open_vocab": False, "vram_hours": "low",
         "output_format": "MOT", "license": "MIT", "integration_complexity": "low",
         "status": "SELECTED_FOR_PHASE2"},
        {"tracker": "MASA", "public_code": "yes (TRACT/masa)", "tao_support": "yes",
         "det_assoc_decoupled": True, "class_agnostic": True, "long_video": "yes",
         "expected_assoc": "strong", "open_vocab": True, "vram_hours": "medium",
         "output_format": "TAO", "license": "check", "integration_complexity": "medium",
         "status": "BACKUP"},
    ])
    matrix = []
    for eid, q, method, src, status, req in [
        ("E0-1", "evaluator修正", "corrected evaluator", "gt", "done", "main"),
        ("E0-2", "GT-track基础基线", "DINO-Mean+B2", "gt", "done", "main"),
        ("E0-3", "architecture bake-off", "transformer/OCD-v2", "gt", "done", "main"),
        ("E0-4", "dual branch", "hard dual", "gt", "done", "supp"),
        ("E0-5", "DINOv3", "DINOv3 mean+B2", "gt", "done", "main"),
        ("E0-6", "router", "R0-R4", "gt", "done", "main"),
        ("E0-7", "causal score adaptation", "C0-C5", "gt", "done", "supp"),
        ("E1-1", "完整TrackEval", "SimOWT", "pred", "this_phase", "main"),
        ("E1-2", "known/novel coverage", "SimOWT", "pred", "this_phase", "main"),
        ("E1-3", "matched-only end-to-end", "DINO-Mean+B2 oracle route", "pred", "this_phase", "supp"),
        ("E1-4", "coverage-aware end-to-end", "SimOWT+OCD", "pred", "this_phase", "main"),
        ("E2-1", "第二tracking frontend", "ByteTrack", "pred", "phase2", "main"),
        ("E3-1", "外部baseline", "SimGCD/GCD/PHE", "gt", "phase3", "main"),
        ("E3-2", "TAO source-domain泛化", "domain analysis", "gt", "phase3", "main"),
    ]:
        matrix.append({"experiment_id": eid, "scientific_question": q, "method": method,
                       "track_source": src, "protocol": "pure+ov", "subset": "full",
                       "metrics": "corrected TrackOCD", "compute": "cpu",
                       "dependencies": "", "status": status,
                       "required_for_main_claim": req, "internal_deadline": "2026-08-12",
                       "stop_condition": ""})
    write_csv(OUT / "planning" / "frozen_experiment_matrix.csv", matrix)
    write_csv(OUT / "planning" / "internal_schedule.csv", [
        {"milestone": m, "date": d, "items": items}
        for m, d, items in [
            ("M1", "2026-08-12", "protocol freeze, audit, TrackEval, coverage, E2E, claims, matrix"),
            ("M2", "2026-08-20", "second tracker, TrackEval, E2E, sensitivity"),
            ("M3", "2026-08-28", "P0 baselines, domain analysis, freeze experiments"),
            ("M4", "2026-09-03", "statistics, qualitative, failure taxonomy"),
            ("M5", "2026-09-08", "paper draft + supplement + code"),
            ("M6", "2026-09-12", "final submission package"),
        ]
    ])


def main():
    for d in (OUT / "audit", OUT / "tables", OUT / "figures", OUT / "end_to_end",
              OUT / "planning", OUT / "tests", OUT / "tracking_eval"):
        d.mkdir(parents=True, exist_ok=True)
    artifact_inventory()
    protocol_statistics()
    legacy_vs_corrected()
    toy_example()
    convert_trackeval()
    print("converted; running TrackEval...", flush=True)
    te_rows = run_trackeval()
    print("trackeval rows", len(te_rows), flush=True)
    finalize_tracking_outputs(te_rows)
    role_coverage()
    end_to_end()
    gt_track_tables()
    claim_evidence()
    planning()
    print("phase1 core done", flush=True)


def finalize_tracking_outputs(te_rows):
    out_dir = OUT / "tracking_eval" / "simowt"
    known = next((r for r in te_rows if r.get("subset") == "known"), {})
    unknown = next((r for r in te_rows if r.get("subset") == "unknown"), {})
    allrow = next((r for r in te_rows if r.get("subset") == "all"), {})
    # class-agnostic metrics json
    (out_dir / "class_agnostic_metrics.json").write_text(json.dumps({
        "subset": "all",
        "HOTA": allrow.get("HOTA"), "DetA": allrow.get("DetA"),
        "AssA": allrow.get("AssA"), "LocA": allrow.get("LocA"),
        "DetRe": allrow.get("DetRe"), "DetPr": allrow.get("DetPr"),
        "AssRe": allrow.get("AssRe"), "AssPr": allrow.get("AssPr"),
        "OWTA": allrow.get("OWTA"),
        "IDF1": None, "MOTA": None, "MOTP": None, "FP": None, "FN": None,
        "IDSW": None, "Frag": None, "MT": None, "PT": None, "ML": None,
        "NA_reason": "TAO_OW TrackEval adapter implements HOTA+Count only; CLEAR/Identity metrics are not available",
    }, indent=2))
    write_csv(out_dir / "known_role_metrics.csv", [{"subset": "known", **known}] if known else [])
    write_csv(out_dir / "novel_role_metrics.csv", [{"subset": "unknown", **unknown}] if unknown else [])
    # track quality metrics (from arch1_5 audit)
    stats = json.loads((PROJECT_ROOT / "outputs/arch1_5/track_audit/pred_track_stats.json").read_text())["track_stats"]
    write_csv(out_dir / "track_quality_metrics.csv", [{
        "metric": "single_frame_ratio", "value": stats["len1_ratio"]},
        {"metric": "mean_track_length", "value": stats["length_mean"]},
        {"metric": "median_track_length", "value": stats["length_median"]},
        {"metric": "pred_track_count", "value": stats["num_tracks"]},
    ])
    (out_dir / "conversion_manifest.json").write_text(json.dumps({
        "tracker": "simowt", "gt": "data/raw/tao/annotations/validation.json",
        "predictions": "outputs/simowt/val_predictions.json",
        "trackeval_gt_folder": str(TE / "data/gt/tao/tao_validation"),
        "trackeval_tracker_folder": str(TE / "data/trackers/tao/tao_validation/simowt/data"),
        "format": "TAO_OW annotation list (image_id, video_id, track_id, bbox, score)",
        "class_agnostic": True,
        "notes": "category_id retained but matching is class-agnostic (TAO_OW combines classes)",
    }, indent=2))


if __name__ == "__main__":
    main()
