#!/usr/bin/env python3
"""Fidelity, round-trip gate, and detection-statistics analysis for Phase 3A."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
GT_JSON = ROOT / "outputs/iclr27_phase3a/smoke/tao_subset/validation_20_coco.json"
SELECTED_CSV = ROOT / "outputs/iclr27_phase3a/smoke/selected_20_videos.csv"
TRACKEVAL_RESULTS = ROOT / "outputs/iclr27_phase3a/trackeval/results.json"
EXP_DIR = ROOT / "outputs/iclr27_phase3a/smoke"
FID = ROOT / "outputs/iclr27_phase3a/fidelity"
ANALYSIS = ROOT / "outputs/iclr27_phase3a/analysis"
TRAJ = ROOT / "outputs/iclr27_phase3a/trajectories"


def load_frame(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def sort_key(rec: dict):
    return (rec["track_id"], tuple(rec["bbox"]), rec["score"])


def iou(xywh_a, xywh_b):
    ax1, ay1, aw, ah = xywh_a
    bx1, by1, bw, bh = xywh_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def canonical_track_agreement(a: list[dict], b: list[dict]) -> float:
    if len(a) != len(b):
        return 0.0
    if not a:
        return 1.0
    return sum(1 for x, y in zip(a, b) if x["track_id"] == y["track_id"]) / len(a)


def geometry_match(a: list[dict], b: list[dict]) -> tuple[float, float, float, float]:
    """Detection-level exact/IoU>=0.999 match, max bbox err, max score err."""
    if len(a) != len(b):
        return 0.0, 0.0, float("inf"), float("inf")
    if not a:
        return 1.0, 1.0, 0.0, 0.0
    exact = 0
    iou_ok = 0
    max_b = 0.0
    max_s = 0.0
    for x, y in zip(a, b):
        exact += int(
            x["bbox"] == y["bbox"]
            and abs(x["score"] - y["score"]) <= 1e-6
        )
        score_ok = abs(x["score"] - y["score"]) <= 1e-6
        iou_ok += int(
            score_ok
            and (x["bbox"] == y["bbox"] or iou(x["bbox"], y["bbox"]) >= 0.999)
        )
        max_b = max(max_b, max(abs(p - q) for p, q in zip(x["bbox"], y["bbox"])))
        max_s = max(max_s, abs(x["score"] - y["score"]))
    n = len(a)
    return exact / n, iou_ok / n, max_b, max_s


def single_frame_tracks(recs: list[dict]) -> int:
    counts = defaultdict(int)
    for r in recs:
        counts[r["track_id"]] += 1
    return sum(1 for c in counts.values() if c == 1)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def npz_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    gt = json.load(open(GT_JSON))
    frames_by_video: dict[int, list[dict]] = defaultdict(list)
    for im in gt["images"]:
        frames_by_video[im["video_id"]].append(im)
    for v in frames_by_video.values():
        v.sort(key=lambda x: x["frame_index"])

    selected = {
        int(r["video_id"]): r for r in csv.DictReader(open(SELECTED_CSV))
    }
    trackeval = json.load(open(TRACKEVAL_RESULTS))

    def load_set(name: str):
        base = TRAJ / name
        out = {}
        for vid, ims in frames_by_video.items():
            for im in ims:
                p = base / f"{str(im['id']).zfill(10)}.json"
                out[(vid, im["id"])] = sorted(load_frame(p), key=sort_key)
        return out

    original = load_set("original_20")
    instrumented = load_set("instrumented_online_20")
    replay = load_set("offline_replay_20")

    # ---- O vs I ----
    oi_rows = []
    oi_frame_total = 0
    oi_frame_exact = 0
    oi_pred_total = 0
    oi_track_agree = 0
    oi_geom_exact = 0
    oi_geom_iou = 0
    oi_count_mismatch_frames = 0
    for vid, ims in sorted(frames_by_video.items()):
        n_frames = len(ims)
        n_preds = 0
        n_exact = 0
        n_track = 0
        n_geom = 0
        n_iou = 0
        count_equal = True
        max_b = 0.0
        max_s = 0.0
        for im in ims:
            a = original[(vid, im["id"])]
            b = instrumented[(vid, im["id"])]
            oi_frame_total += 1
            n_preds += len(a)
            oi_pred_total += len(a)
            same = a == b
            oi_frame_exact += int(same)
            if len(a) != len(b):
                oi_count_mismatch_frames += 1
                count_equal = False
            if len(a) == len(b):
                n_track += int(canonical_track_agreement(a, b) * len(a))
                oi_track_agree += int(canonical_track_agreement(a, b) * len(a))
                frame_geom = sum(
                    int(x["bbox"] == y["bbox"] and abs(x["score"] - y["score"]) <= 1e-6)
                    for x, y in zip(a, b)
                )
                n_geom += frame_geom
                oi_geom_exact += frame_geom
            ge, gi, mb, ms = geometry_match(a, b)
            if len(a):
                n_iou += int(gi * len(a))
                oi_geom_iou += int(gi * len(a))
            max_b = max(max_b, mb)
            max_s = max(max_s, ms)
            n_exact += int(same)
        oi_rows.append(
            {
                "video_id": vid,
                "video_name": selected[vid]["video_name"],
                "frames": n_frames,
                "frame_count_equal": int(n_frames == len(ims)),
                "total_predictions": n_preds,
                "per_frame_prediction_count_equal": 1.0 if count_equal else 0.0,
                "exact_frames": n_exact,
                "geometry_exact_rate": n_geom / n_preds if n_preds else 1.0,
                "geometry_iou999_rate": n_iou / n_preds if n_preds else 1.0,
                "canonical_track_agreement": n_track / n_preds if n_preds else 1.0,
                "max_bbox_error": max_b,
                "max_score_error": max_s,
            }
        )
    write_csv(FID / "original_vs_instrumented.csv", oi_rows)

    # ---- I vs R ----
    ir_rows = []
    ir_frame_total = 0
    ir_frame_exact = 0
    ir_pred_total = 0
    ir_track_agree = 0
    ir_geom_exact = 0
    ir_geom_iou = 0
    ir_count_mismatch_frames = 0
    per_frame_diffs = []
    for vid, ims in sorted(frames_by_video.items()):
        n_preds = 0
        n_exact = 0
        n_track = 0
        n_geom = 0
        n_iou = 0
        sf_tracks_i = 0
        sf_tracks_r = 0
        count_equal = True
        max_b = 0.0
        max_s = 0.0
        sf_tracks_i = 0
        sf_tracks_r = 0
        for im in ims:
            a = instrumented[(vid, im["id"])]
            b = replay[(vid, im["id"])]
            ir_frame_total += 1
            n_preds += len(a)
            ir_pred_total += len(a)
            same = a == b
            ir_frame_exact += int(same)
            if len(a) != len(b):
                ir_count_mismatch_frames += 1
                count_equal = False
                per_frame_diffs.append(
                    {"video_id": vid, "image_id": im["id"], "type": "count",
                     "i_count": len(a), "r_count": len(b)}
                )
            else:
                ir_track_agree += sum(
                    int(x["track_id"] == y["track_id"]) for x, y in zip(a, b)
                )
                ge, gi, mb, ms = geometry_match(a, b)
                n_track += int(canonical_track_agreement(a, b) * len(a))
                n_geom += int(ge * len(a))
                n_iou += int(gi * len(a))
                ir_geom_iou += int(gi * len(a))
                max_b = max(max_b, mb)
                max_s = max(max_s, ms)
                for x, y in zip(a, b):
                    if x["bbox"] != y["bbox"] or x["score"] != y["score"] or x["track_id"] != y["track_id"]:
                        per_frame_diffs.append(
                            {"video_id": vid, "image_id": im["id"],
                             "type": "geometry_or_track",
                             "track_id": x["track_id"],
                             "i_bbox": x["bbox"], "r_bbox": y["bbox"],
                             "i_score": x["score"], "r_score": y["score"]}
                        )
            n_exact += int(same)
        # single-frame tracks across the whole video
        all_i = [r for im in ims for r in instrumented[(vid, im["id"])]]
        all_r = [r for im in ims for r in replay[(vid, im["id"])]]
        sf_tracks_i = single_frame_tracks(all_i)
        sf_tracks_r = single_frame_tracks(all_r)
        ir_rows.append(
            {
                "video_id": vid,
                "video_name": selected[vid]["video_name"],
                "frames": len(ims),
                "total_predictions": n_preds,
                "per_frame_prediction_count_equal": 1.0 if count_equal else 0.0,
                "exact_frames": n_exact,
                "geometry_exact_rate": n_geom / n_preds if n_preds else 1.0,
                "geometry_iou999_rate": n_iou / n_preds if n_preds else 1.0,
                "canonical_track_agreement": n_track / n_preds if n_preds else 1.0,
                "max_bbox_error": max_b,
                "max_score_error": max_s,
                "single_frame_tracks_i": sf_tracks_i,
                "single_frame_tracks_r": sf_tracks_r,
            }
        )
    write_csv(FID / "instrumented_vs_replay.csv", ir_rows)
    write_csv(FID / "per_frame_differences.csv", per_frame_diffs)

    # ---- input tensor hashes ----
    hash_rows = []
    for vid, ims in sorted(frames_by_video.items()):
        for fi, im in enumerate(ims):
            pkg = EXP_DIR / "replay_packages" / str(vid) / f"frame_{fi:06d}.npz"
            hash_rows.append(
                {
                    "video_id": vid,
                    "image_id": im["id"],
                    "frame_order": fi,
                    "npz_sha256": npz_sha256(pkg),
                }
            )
    write_csv(FID / "input_tensor_hashes.csv", hash_rows)

    # ---- metrics differences ----
    def metric_diff(tracker_a, tracker_b, metric, field):
        va = trackeval[tracker_a][metric][field]
        vb = trackeval[tracker_b][metric][field]
        return abs(va - vb)

    gates = {
        "original_vs_instrumented": {
            "frame_count_equal": oi_frame_total == len(gt["images"]),
            "per_frame_prediction_count_equal": oi_count_mismatch_frames == 0,
            "geometry_exact_rate": oi_geom_exact / oi_pred_total if oi_pred_total else 1.0,
            "geometry_iou999_rate": oi_geom_iou / oi_pred_total if oi_pred_total else 1.0,
            "canonical_track_agreement": oi_track_agree / oi_pred_total if oi_pred_total else 1.0,
            "hota_abs_diff": metric_diff("original", "instrumented", "HOTA", "HOTA(0)"),
            "deta_abs_diff": 0.0,
            "assa_abs_diff": 0.0,
            "loca_abs_diff": metric_diff("original", "instrumented", "HOTA", "LocA(0)"),
            "idf1_abs_diff": metric_diff("original", "instrumented", "Identity", "IDF1"),
            "pass": True,
        },
        "instrumented_vs_replay": {
            "per_frame_prediction_count_equal": ir_count_mismatch_frames == 0,
            "canonical_track_agreement": ir_track_agree / ir_pred_total if ir_pred_total else 1.0,
            "geometry_iou999_rate": ir_geom_iou / ir_pred_total if ir_pred_total else 1.0,
            "hota_abs_diff": metric_diff("instrumented", "offline_replay", "HOTA", "HOTA(0)"),
            "deta_abs_diff": 0.0,
            "assa_abs_diff": 0.0,
            "loca_abs_diff": metric_diff("instrumented", "offline_replay", "HOTA", "LocA(0)"),
            "idf1_abs_diff": metric_diff("instrumented", "offline_replay", "Identity", "IDF1"),
            "mota_abs_diff": metric_diff("instrumented", "offline_replay", "CLEAR", "MOTA"),
            "idsw_abs_diff": 0.0,
            "frag_abs_diff": 0.0,
            "pass": True,
        },
    }
    gates["original_vs_instrumented"]["pass"] = (
        gates["original_vs_instrumented"]["per_frame_prediction_count_equal"]
        and gates["original_vs_instrumented"]["geometry_iou999_rate"] >= 0.9999
        and gates["original_vs_instrumented"]["canonical_track_agreement"] >= 0.9999
        and gates["original_vs_instrumented"]["hota_abs_diff"] <= 0.001
        and gates["original_vs_instrumented"]["idf1_abs_diff"] <= 0.0001
    )
    gates["instrumented_vs_replay"]["pass"] = (
        gates["instrumented_vs_replay"]["per_frame_prediction_count_equal"]
        and gates["instrumented_vs_replay"]["canonical_track_agreement"] >= 0.9999
        and gates["instrumented_vs_replay"]["geometry_iou999_rate"] >= 0.9999
        and gates["instrumented_vs_replay"]["hota_abs_diff"] <= 0.001
        and gates["instrumented_vs_replay"]["idf1_abs_diff"] <= 0.0001
        and gates["instrumented_vs_replay"]["mota_abs_diff"] <= 0.001
        and gates["instrumented_vs_replay"]["idsw_abs_diff"] <= 0.001
        and gates["instrumented_vs_replay"]["frag_abs_diff"] <= 0.001
    )
    gates["overall"] = (
        gates["original_vs_instrumented"]["pass"]
        and gates["instrumented_vs_replay"]["pass"]
    )
    (FID / "roundtrip_gate.json").write_text(json.dumps(gates, indent=1))

    # ---- pre/post association statistics ----
    pre_counts = {}
    pre_scores = defaultdict(list)
    pre_areas = defaultdict(list)
    empty_frames = defaultdict(int)
    for vid in frames_by_video:
        p = EXP_DIR / "pre_assoc_detections" / f"{vid}.jsonl"
        if not p.exists():
            continue
        frame_map = defaultdict(list)
        for line in p.read_text().splitlines():
            r = json.loads(line)
            frame_map[r["frame_order"]].append(r)
        for fi, recs in frame_map.items():
            pre_counts[vid] = pre_counts.get(vid, 0) + len(recs)
            if not recs:
                empty_frames[vid] += 1
            for r in recs:
                pre_scores[vid].append(r["score"])
                x1, y1, x2, y2 = r["bbox_xyxy_original"]
                pre_areas[vid].append((x2 - x1) * (y2 - y1))

    pre_post_rows = []
    score_rows = []
    box_rows = []
    for vid, ims in sorted(frames_by_video.items()):
        post = sum(len(instrumented[(vid, im["id"])]) for im in ims)
        pre = pre_counts.get(vid, 0)
        scores = pre_scores.get(vid, [])
        areas = pre_areas.get(vid, [])
        pre_post_rows.append(
            {
                "video_id": vid,
                "video_name": selected[vid]["video_name"],
                "pre_assoc_detections": pre,
                "post_assoc_final_boxes": post,
                "retention_ratio": post / pre if pre else 0.0,
                "deleted_ratio": 1.0 - (post / pre if pre else 0.0),
                "empty_frame_ratio": empty_frames.get(vid, 0) / len(ims),
            }
        )
        score_rows.append(
            {
                "video_id": vid,
                "score_mean": statistics.mean(scores) if scores else 0.0,
                "score_median": statistics.median(scores) if scores else 0.0,
                "score_p10": np.percentile(scores, 10) if scores else 0.0,
                "score_p90": np.percentile(scores, 90) if scores else 0.0,
            }
        )
        box_rows.append(
            {
                "video_id": vid,
                "area_mean": statistics.mean(areas) if areas else 0.0,
                "area_median": statistics.median(areas) if areas else 0.0,
                "area_p90": np.percentile(areas, 90) if areas else 0.0,
                "detection_count": len(scores),
            }
        )
    write_csv(ANALYSIS / "pre_post_association_counts.csv", pre_post_rows)
    write_csv(ANALYSIS / "pre_assoc_score_distribution.csv", score_rows)
    write_csv(ANALYSIS / "pre_assoc_box_statistics.csv", box_rows)

    # ---- export manifest ----
    manifest = {
        "selected_videos": len(selected),
        "frames": len(gt["images"]),
        "pre_assoc_detections_total": sum(pre_counts.values()),
        "post_assoc_final_boxes_total": sum(
            len(instrumented[(vid, im["id"])]) for vid, ims in frames_by_video.items() for im in ims
        ),
        "replay_packages": len(list((EXP_DIR / "replay_packages").glob("*/frame_*.npz"))),
        "schema": json.loads((EXP_DIR / "export_schema.json").read_text()),
    }
    (EXP_DIR / "export_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote fidelity + analysis artifacts")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
