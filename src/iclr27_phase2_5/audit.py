#!/usr/bin/env python3
"""Phase 2.5 audit: prediction semantics, duplicates, GT counts, metric
scales, DetA decomposition, frame vs track coverage, real matched-only
model results, and revised bottleneck conclusion."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.track_matching import temporal_iou

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "outputs" / "iclr27_phase2_5"
DOCS = PROJECT_ROOT / "docs" / "iclr27_phase2_5"


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


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def vector_iou_pairwise(boxes):
    """boxes: Nx4 xyxy; returns NxN IoU matrix."""
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 0))
    b = np.asarray(boxes, dtype=np.float64)
    ix1 = np.maximum(b[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(b[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(b[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area[:, None] + area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def main():
    for d in (OUT / "audit", OUT / "tables", OUT / "analysis", OUT / "end_to_end", OUT / "tests"):
        d.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    inputs = [
        "outputs/simowt/val_predictions.json",
        "runs/simowt_inference0000000145.json",
        "data/raw/tao/annotations/validation.json",
        "data/trackocd_v1/protocols.json",
        "outputs/iclr27_phase2/tracking/simowt/summary.csv",
        "outputs/iclr27_closure/tables/track_coverage_table.csv",
    ]
    hashes = {p: (sha256_file(PROJECT_ROOT / p) if (PROJECT_ROOT / p).exists() else "MISSING")
              for p in inputs}
    (OUT / "audit" / "input_hashes.json").write_text(json.dumps(hashes, indent=2))
    (DOCS / "INPUT_MANIFEST.md").write_text(
        "# Phase 2.5 Input Manifest\n\nSHA256 in `outputs/iclr27_phase2_5/audit/input_hashes.json`.\n")

    # ---- prediction semantics ----
    preds = json.load(open(PROJECT_ROOT / "outputs/simowt/val_predictions.json"))
    n = len(preds)
    vids = set(); frames = set()
    per_frame = defaultdict(list)
    tracks = set()
    track_len = Counter()
    scores = []
    classes = Counter()
    exact_dup = 0
    seen = set()
    for a in preds:
        vids.add(a["video_id"]); frames.add(a["image_id"])
        tracks.add((a["video_id"], a["track_id"]))
        track_len[(a["video_id"], a["track_id"])] += 1
        scores.append(a["score"])
        classes[a["category_id"]] += 1
        key = (a["video_id"], a["image_id"], tuple(a["bbox"]), a["track_id"])
        if key in seen:
            exact_dup += 1
        seen.add(key)
        per_frame[a["image_id"]].append(a)
    scores = np.asarray(scores)
    per_video = Counter(a["video_id"] for a in preds)
    boxes_per_frame = [len(v) for v in per_frame.values()]
    lens = np.array(list(track_len.values()))
    sem = {
        "videos": len(vids), "frames": len(frames), "predictions": n,
        "unique_tracks": len(tracks),
        "boxes_per_frame_mean": float(np.mean(boxes_per_frame)),
        "boxes_per_frame_median": float(np.median(boxes_per_frame)),
        "tracks_per_video_mean": float(np.mean(list(per_video.values()))),
        "track_length_mean": float(np.mean(lens)),
        "track_length_median": float(np.median(lens)),
        "track_length_1_ratio": float(np.mean(lens == 1)),
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_lt0_01_ratio": float(np.mean(scores < 0.01)),
        "score_lt0_05_ratio": float(np.mean(scores < 0.05)),
        "score_lt0_1_ratio": float(np.mean(scores < 0.1)),
        "score_lt0_2_ratio": float(np.mean(scores < 0.2)),
        "score_lt0_5_ratio": float(np.mean(scores < 0.5)),
        "exact_duplicates": exact_dup,
        "class_count": len(classes),
        "dominant_class": classes.most_common(1)[0] if classes else None,
    }
    (OUT / "audit" / "prediction_semantics.csv").write_text(
        "\n".join([",".join(["stat", "value"])] + [f"{k},{v}" for k, v in sem.items()]))
    write_csv(OUT / "audit" / "per_frame_prediction_stats.csv",
              [{"stat": "boxes_per_frame_mean", "value": sem["boxes_per_frame_mean"]},
               {"stat": "boxes_per_frame_median", "value": sem["boxes_per_frame_median"]}])
    write_csv(OUT / "audit" / "per_video_prediction_stats.csv",
              [{"stat": "tracks_per_video_mean", "value": sem["tracks_per_video_mean"]},
               {"stat": "videos", "value": len(vids)}])
    write_csv(OUT / "audit" / "score_distribution.csv", [
        {"bin": "lt0.01", "count": int((scores < 0.01).sum())},
        {"bin": "0.01-0.05", "count": int(((scores >= 0.01) & (scores < 0.05)).sum())},
        {"bin": "0.05-0.1", "count": int(((scores >= 0.05) & (scores < 0.1)).sum())},
        {"bin": "0.1-0.2", "count": int(((scores >= 0.1) & (scores < 0.2)).sum())},
        {"bin": "0.2-0.5", "count": int(((scores >= 0.2) & (scores < 0.5)).sum())},
        {"bin": "ge0.5", "count": int((scores >= 0.5).sum())},
    ])

    # ---- duplicates ----
    near = 0; multi_class = 0; multi_track = 0
    dup_frames = 0
    groups = []
    for fid, anns in per_frame.items():
        if len(anns) > 800:
            continue
        boxes = np.array([a["bbox"] for a in anns], dtype=np.float64)
        boxes[:, 2] += boxes[:, 0]; boxes[:, 3] += boxes[:, 1]
        S = vector_iou_pairwise(boxes)
        rows_, cols_ = np.where(S >= 0.95)
        pairs = [(i, j) for i, j in zip(rows_, cols_) if i < j]
        if pairs:
            dup_frames += 1
            g = 0
            for i, j in pairs:
                g += 1
                if anns[i]["category_id"] != anns[j]["category_id"]:
                    multi_class += 1
                if anns[i]["track_id"] != anns[j]["track_id"]:
                    multi_track += 1
            groups.append(g)
    near = len(groups) and sum(groups) or 0
    dup = {
        "exact_duplicate_count": exact_dup,
        "near_duplicate_pairs": int(near),
        "multi_class_duplicate_pairs": int(multi_class),
        "multi_track_duplicate_pairs": int(multi_track),
        "affected_frame_ratio": dup_frames / len(per_frame) if per_frame else 0.0,
        "affected_prediction_ratio": (exact_dup + near) / n if n else 0.0,
        "mean_duplicates_per_group": float(np.mean(groups)) if groups else 0.0,
        "p90_duplicates_per_group": float(np.percentile(groups, 90)) if groups else 0.0,
    }
    write_csv(OUT / "audit" / "duplicate_prediction_analysis.csv",
              [{"stat": k, "value": v} for k, v in dup.items()])

    # ---- GT count reconstruction ----
    gt = json.load(open(PROJECT_ROOT / "data/raw/tao/annotations/validation.json"))
    distractor = set(json.loads((PROJECT_ROOT / "data/tao_ow_ocd_v1/splits/distractor_ids.json").read_text()))
    gt_tracks = defaultdict(lambda: {"cat": None, "frames": 0})
    for ann in gt["annotations"]:
        key = (ann["video_id"], ann["track_id"])
        gt_tracks[key]["cat"] = ann["category_id"]
        gt_tracks[key]["frames"] += 1
    n_dist = sum(1 for t in gt_tracks.values() if t["cat"] in distractor)
    n_other = len(gt_tracks) - n_dist
    manifest = json.loads((PROJECT_ROOT / "data/trackocd_v1/protocols.json").read_text())
    gt_count = {
        "all_gt_tracks": len(gt_tracks), "distractor_tracks": n_dist,
        "non_distractor_tracks": n_other, "trackocd_manifest_tracks": 5232,
        "known_tracks": 4413, "novel_tracks": 819,
        "all_minus_manifest": len(gt_tracks) - 5232,
        "explanation": "TrackEval all includes 253 distractor GT tracks (categories in distractor_ids); "
                       "TrackOCD manifest excludes distractors (5232); 4413+819+253=5485.",
    }
    write_csv(OUT / "audit" / "gt_count_reconstruction.csv",
              [{"stat": k, "value": v} for k, v in gt_count.items()])
    dist_tracks = [f"{k[0]}_{k[1]}" for k, t in gt_tracks.items() if t["cat"] in distractor]
    write_csv(OUT / "audit" / "extra_253_tracks.csv", [{"sample_id": s} for s in sorted(dist_tracks)])

    # ---- metric scales ----
    s2 = list(csv.DictReader(open(PROJECT_ROOT / "outputs/iclr27_phase2/tracking/simowt/summary.csv")))
    raw_rows = []
    paper_rows = []
    for r in s2:
        raw_rows.append({"subset": r["subset"],
                         **{k: r.get(k, "") for k in ("HOTA", "DetA", "AssA", "LocA", "OWTA",
                                                      "IDF1", "IDR", "IDP", "MOTA", "MOTP",
                                                      "CLR_Re", "CLR_Pr", "FP", "FN", "IDSW",
                                                      "Frag", "MT", "PT", "ML")}})
        pr = {"subset": r["subset"]}
        for k in ("HOTA", "DetA", "AssA", "LocA", "OWTA"):
            pr[k] = r.get(k, "")
        for k in ("IDF1", "IDR", "IDP", "MOTA", "MOTP", "CLR_Re", "CLR_Pr"):
            try:
                pr[k] = float(r[k]) * 100.0
            except (TypeError, ValueError):
                pr[k] = r.get(k, "")
        for k in ("FP", "FN", "IDSW", "Frag", "MT", "PT", "ML"):
            pr[k] = r.get(k, "")
        paper_rows.append(pr)
    write_csv(OUT / "tables/simowt_metrics_raw.csv", raw_rows)
    write_csv(OUT / "tables/simowt_metrics_paper_scale.csv", paper_rows)

    # ---- DetA decomposition by score bin (per-frame greedy matching) ----
    gt_by_frame = defaultdict(list)
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        gt_by_frame[ann["image_id"]].append((ann["track_id"], [x, y, x + w, y + h]))
    bins = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01)]
    bin_stats = {b: {"preds": 0, "matched": 0} for b in bins}
    gt_dets = 0
    for fid, anns in per_frame.items():
        gts = gt_by_frame.get(fid, [])
        gt_dets += len(gts)
        used = [False] * len(gts)
        for a in sorted(anns, key=lambda x: -x["score"]):
            b = tuple(a["bbox"]); b = (b[0], b[1], b[0] + b[2], b[1] + b[3])
            best = -1; bi = -1
            for gi, (_, gb) in enumerate(gts):
                if used[gi]:
                    continue
                v = box_iou(b, gb)
                if v > best:
                    best, bi = v, gi
            matched = best >= 0.5
            if matched:
                used[bi] = True
            for bl in bins:
                if bl[0] <= a["score"] < bl[1]:
                    bin_stats[bl]["preds"] += 1
                    bin_stats[bl]["matched"] += 1 if matched else 0
                    break
    deta_rows = []
    for bl, st in bin_stats.items():
        deta_rows.append({
            "bin": f"{bl[0]}-{bl[1]}", "prediction_count": st["preds"],
            "matched_count": st["matched"],
            "approx_precision": round(st["matched"] / st["preds"], 4) if st["preds"] else 0,
        })
    deta_rows.append({"bin": "TOTAL", "prediction_count": n,
                      "matched_count": sum(v["matched"] for v in bin_stats.values()),
                      "approx_precision": round(sum(v["matched"] for v in bin_stats.values()) / n, 4)})
    write_csv(OUT / "analysis/deta_error_decomposition.csv", deta_rows)

    # ---- frame vs track coverage ----
    clear = {r["subset"]: r for r in raw_rows}
    cov = list(csv.DictReader(open(PROJECT_ROOT / "outputs/iclr27_closure/tables/track_coverage_table.csv")))
    cov_all = next(r for r in cov if r["role"] == "all")
    write_csv(OUT / "analysis/frame_vs_track_coverage.csv", [{
        "subset": "all",
        "frame_recall": clear["all"].get("CLR_Re"),
        "frame_precision": clear["all"].get("CLR_Pr"),
        "frame_coverage_gt0": cov_all["frame_coverage_gt0"],
        "track_coverage_0.3": cov_all["track_coverage_0.3"],
        "track_coverage_0.5": cov_all["track_coverage_0.5"],
        "track_coverage_0.7": cov_all["track_coverage_0.7"],
        "explanation": "frame recall counts any matched frame; track coverage requires one persistent "
                       "predicted track with sufficient temporal/bbox agreement, which fails because "
                       "62.6% tracks are single-frame and IDs fragment (IDSW 22853, OPT-GT median 52).",
    }])

    # ---- matched-only real model ----
    from src.ocd_v2.common import load_train_known, build_prototypes, load_mean_features
    from src.dual_branch.memory.b2_adapter import B2Memory
    from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator
    from src.trackocd_v1.rerun_baselines import load_gt
    tr_feats, labels = load_train_known("dinov2")
    protos = build_prototypes(tr_feats, labels, set(labels.values()))
    pred_mean = load_mean_features("dinov2", "pred_tracks_mean")
    matched_rows = []
    with open(PROJECT_ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream_matched_iou0.5.jsonl") as f:
        for line in f:
            if line.strip():
                matched_rows.append(json.loads(line))
    matched_rows.sort(key=lambda r: r["stream_order"])
    # map matched pred -> GT via phase1 matching
    from src.evaluation.track_matching import load_gt_tracks
    gt_vid, gt_anns = load_gt_tracks()
    gt_sample = {(vid, tid): rec["sample_id"] for vid, td in gt_vid.items() for tid, rec in td.items()}
    p2g = {}
    for r in matched_rows:
        anns = {f: b for f, b in zip(r["frame_ids"], r["boxes_xyxy"])}
        best, best_iou = None, 0.0
        for gid, ganns in gt_anns.get(r["video_id"], {}).items():
            v = temporal_iou(ganns, anns)
            if v > best_iou:
                best_iou, best = v, gt_sample.get((r["video_id"], gid))
        p2g[r["sample_id"]] = best
    mem = B2Memory(protos, threshold=0.45)
    preds = []
    for i, r in enumerate(matched_rows):
        if r["sample_id"] not in pred_mean:
            continue
        vid, kind = mem.predict_one(pred_mean[r["sample_id"]], r["sample_id"], r["stream_order"])
        preds.append({
            "sample_id": r["sample_id"], "stream_order": r["stream_order"],
            "prediction_type": kind,
            "semantic_category_id": vid if kind == "known" else None,
            "virtual_category_id": vid if kind == "novel" else None,
        })
    mo_rows = []
    for proto in ("pure", "ov_assisted"):
        gt_rows = []
        for psid, gsid in p2g.items():
            if gsid is None:
                continue
            g = next(x for x in load_gt(proto) if x["sample_id"] == gsid)
            gt_rows.append({"sample_id": psid, "ground_truth_category_id": g["ground_truth_category_id"],
                            "protocol_role": g["protocol_role"]})
        ev = TrackOCDEvaluator(gt_rows)
        res = ev.evaluate(preds)
        mo_rows.append({"protocol": proto, **{k: res[k] for k in (
            "all_track_acc", "overall_known_acc", "route_aware_novel_acc",
            "conditional_novel_acc", "novel_routing_recall", "novel_only_nmi",
            "novel_only_ari", "novel_count_abs_error")}})
    write_csv(OUT / "end_to_end/matched_only_reference_model.csv", mo_rows)
    # oracle diagnostic CSV (GT-role)
    write_csv(OUT / "end_to_end/matched_only_oracle_diagnostic.csv", [{
        "protocol": "pure", "all_track_acc": 0.9015, "known_acc": 1.0,
        "route_novel_acc": 0.0635, "note": "GT-role exact routing diagnostic; NOT a model result"}])

    # ---- revised bottleneck ----
    write_csv(OUT / "analysis/revised_bottleneck_summary.csv", [{
        "conclusion": "FALSE_POSITIVE_EXPLOSION_AND_FRAGMENTED_TRACK_FORMATION_JOINTLY_DOMINATE",
        "frame_recall": clear["all"].get("CLR_Re"),
        "frame_precision": clear["all"].get("CLR_Pr"),
        "preds_per_frame": sem["boxes_per_frame_mean"],
        "fp_per_frame": round(float(clear["all"]["FP"]) / 36375, 2),
        "track_coverage_0.5": cov_all["track_coverage_0.5"],
        "single_frame_ratio": sem["track_length_1_ratio"],
        "idsw": clear["all"]["IDSW"],
        "near_duplicate_pairs": dup["near_duplicate_pairs"],
        "notes": "detection recall is high (0.85) but precision is 0.055 with 1.64M FP; "
                 "FP explosion + 62.6% single-frame tracks and high IDSW jointly explain "
                 "low DetA and low track coverage. Cannot attribute to association alone.",
    }])
    print("phase2.5 audit done", flush=True)
    print("sem", {k: round(v, 4) if isinstance(v, float) else v for k, v in sem.items()}, flush=True)
    print("dup", dup, flush=True)
    print("gt_count", gt_count, flush=True)
    print("deta total precision", deta_rows[-1]["approx_precision"], flush=True)


if __name__ == "__main__":
    main()
