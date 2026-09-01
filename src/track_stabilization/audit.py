#!/usr/bin/env python3
"""Architecture 1.5 Stage B: SimOWT prediction-track audit.

Reads the raw merged `outputs/simowt/val_predictions.json`, verifies track ID
semantics, computes track-level statistics, compares the adapter-built stream,
and computes per-GT fragmentation against predicted tracks (temporal IoU).

Usage:
  audit.py --stage basic    # semantics + stats + matched/unmatched
  audit.py --stage gtfrag   # per-GT fragmentation (needs basic compact cache)
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import resource
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "outputs" / "arch1_5" / "track_audit"
RAW_PREDS = PROJECT_ROOT / "outputs" / "simowt" / "val_predictions.json"
ADAPTER_STREAM = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "pred_track_stream.jsonl"
MATCHED_STREAM = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "pred_track_stream_matched_iou0.5.jsonl"
MATCHED_GT_IDS = PROJECT_ROOT / "outputs" / "metrics" / "matched_gt_ids_iou0.5.json"
GT_ANNOTATIONS = PROJECT_ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json"
PRIVATE_LABELS = PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "private" / "val_gt_track_labels.jsonl"
COMPACT_CACHE = OUT / "pred_tracks_compact.pkl"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def box_area(b):
    x, y, w, h = b
    return max(0.0, float(w)) * max(0.0, float(h))


def load_raw_predictions():
    """Load raw merged JSON and group per video/track, dropping segmentation
    blobs as soon as possible to bound memory."""
    with open(RAW_PREDS) as f:
        data = json.load(f)
    print(f"raw records: {len(data)}", flush=True)
    per_video = defaultdict(dict)   # video_id -> track_id -> list of anns
    dup_keys = 0
    bad_boxes = 0
    bad_scores = 0
    seen = set()
    for a in data:
        a.pop("segmentations", None)
        vid = a.get("video_id")
        tid = a.get("track_id")
        img = a.get("image_id")
        if vid is None or tid is None or img is None:
            continue
        key = (vid, tid, img)
        if key in seen:
            dup_keys += 1
            continue
        seen.add(key)
        bb = a.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4 or min(bb) < 0:
            bad_boxes += 1
            continue
        sc = a.get("score")
        if sc is None or sc < 0 or sc > 1.0001:
            bad_scores += 1
        per_video[vid].setdefault(tid, []).append(a)
    print(f"dup (vid,tid,img): {dup_keys}, bad boxes: {bad_boxes}, bad scores: {bad_scores}", flush=True)
    return per_video


def reconstruct_tracks(per_video):
    """Return dict video_id -> track_id -> {frames, boxes, scores, cats}."""
    tracks = {}
    for vid, tmap in per_video.items():
        tracks[vid] = {}
        for tid, anns in tmap.items():
            anns.sort(key=lambda a: a["image_id"])
            tracks[vid][tid] = {
                "frames": [a["image_id"] for a in anns],
                "boxes": [a["bbox"] for a in anns],
                "scores": [float(a["score"]) for a in anns],
                "cats": [a.get("category_id") for a in anns],
            }
    return tracks


def track_stats(tracks):
    lens = []
    mean_scores = []
    mean_areas = []
    per_video_count = Counter()
    for vid, tmap in tracks.items():
        per_video_count[vid] += len(tmap)
        for tid, t in tmap.items():
            L = len(t["frames"])
            lens.append(L)
            mean_scores.append(float(np.mean(t["scores"])) if t["scores"] else 0.0)
            mean_areas.append(float(np.mean([box_area(b) for b in t["boxes"]])) if t["boxes"] else 0.0)
    lens = np.asarray(lens, dtype=np.int64)
    stats = {
        "num_tracks": int(len(lens)),
        "num_videos_with_tracks": len(per_video_count),
        "length_mean": float(lens.mean()),
        "length_median": float(np.median(lens)),
        "length_p75": float(np.percentile(lens, 75)),
        "length_p90": float(np.percentile(lens, 90)),
        "length_p95": float(np.percentile(lens, 95)),
        "length_p99": float(np.percentile(lens, 99)),
        "length_max": int(lens.max()) if len(lens) else 0,
        "len1_ratio": float((lens == 1).mean()) if len(lens) else 0.0,
        "len_lt3_ratio": float((lens < 3).mean()) if len(lens) else 0.0,
        "len_lt5_ratio": float((lens < 5).mean()) if len(lens) else 0.0,
        "len_lt10_ratio": float((lens < 10).mean()) if len(lens) else 0.0,
        "score_mean_track_mean": float(np.mean(mean_scores)) if mean_scores else 0.0,
        "area_mean_track_mean": float(np.mean(mean_areas)) if mean_areas else 0.0,
        "tracks_per_video_mean": float(np.mean(list(per_video_count.values()))) if per_video_count else 0.0,
        "tracks_per_video_median": float(np.median(list(per_video_count.values()))) if per_video_count else 0.0,
    }
    # joint distributions: length x score and length x area buckets
    hist = Counter()
    joint_sa = Counter()
    for L, s, a in zip(lens.tolist(), mean_scores, mean_areas):
        lb = "1" if L == 1 else ("2" if L == 2 else ("3-4" if L < 5 else ("5-9" if L < 10 else ("10-29" if L < 30 else "30+"))))
        sb = "s<0.3" if s < 0.3 else ("s0.3-0.5" if s < 0.5 else ("s0.5-0.7" if s < 0.7 else "s>=0.7"))
        ab = "a<500" if a < 500 else ("a500-2k" if a < 2000 else ("a2k-10k" if a < 10000 else "a>=10k"))
        hist[(lb, sb)] += 1
        joint_sa[(lb, ab)] += 1
    stats["length_score_joint"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(hist.items())}
    stats["length_area_joint"] = {f"{k[0]}|{k[1]}": v for k, v in sorted(joint_sa.items())}
    # length histogram CSV
    with open(OUT / "pred_track_length_hist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["length", "count"])
        for L, c in sorted(Counter(lens.tolist()).items()):
            w.writerow([L, c])
    return stats


def semantics_check(tracks):
    # track_id namespaces: report per-video uniqueness and whether ids reset
    global_key_count = 0
    reused_track_ids = 0
    per_video_tid_set = Counter()
    for vid, tmap in tracks.items():
        per_video_tid_set[vid] = len(tmap)
        global_key_count += len(tmap)
    tid_hist = Counter()
    for vid, tmap in tracks.items():
        for tid in tmap:
            tid_hist[tid] += 1
    reused_track_ids = sum(1 for tid, c in tid_hist.items() if c > 1)
    return {
        "track_id_reused_across_videos": reused_track_ids,
        "per_video_track_count_min": min(per_video_tid_set.values()) if per_video_tid_set else 0,
        "per_video_track_count_max": max(per_video_tid_set.values()) if per_video_tid_set else 0,
        "per_video_track_ids_are_unique": True,
        "global_track_key_count": global_key_count,
    }


def adapter_consistency(tracks):
    """Compare reconstructed raw tracks with the adapter-built stream."""
    adapter = {}
    with open(ADAPTER_STREAM) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                adapter[(r["video_id"], r["track_id"])] = len(r["frame_ids"])
    raw_keys = {(vid, tid) for vid, tmap in tracks.items() for tid in tmap}
    adapter_keys = set(adapter)
    missing = sorted(adapter_keys - raw_keys)
    extra = sorted(raw_keys - adapter_keys)
    mismatch_frames = sum(
        1 for k in raw_keys & adapter_keys
        if adapter[k] != len(tracks[k[0]][k[1]]["frames"])
    )
    return {
        "raw_track_keys": len(raw_keys),
        "adapter_track_keys": len(adapter_keys),
        "adapter_missing_keys": len(missing),
        "adapter_extra_keys": len(extra),
        "frame_count_mismatches": mismatch_frames,
        "missing_sample": missing[:5],
        "extra_sample": extra[:5],
    }


def load_gt():
    gt = json.load(open(GT_ANNOTATIONS))
    img_to_video = {im["id"]: im["video_id"] for im in gt["images"]}
    private = {}
    with open(PRIVATE_LABELS) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                private[r["sample_id"]] = r
    by_video = defaultdict(dict)
    for ann in gt["annotations"]:
        vid = ann["video_id"]
        tid = ann["track_id"]
        rec = by_video[vid].setdefault(
            tid,
            {"frames": [], "boxes": [], "category_id": ann["category_id"]},
        )
        rec["frames"].append(ann["image_id"])
        rec["boxes"].append([ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]])
    private_tracks = {}
    for vid, tmap in by_video.items():
        for tid, rec in tmap.items():
            sid = f"{vid}_{tid}"
            if sid not in private:
                continue
            rec["sample_id"] = sid
            rec["is_known"] = private[sid]["is_known"]
            rec["category_id"] = private[sid]["ground_truth_category_id"]
            private_tracks[sid] = (vid, tid, rec)
    return by_video, private_tracks


def matched_unmatched(tracks, private_tracks):
    matched_gt_ids = set(json.loads(MATCHED_GT_IDS.read_text()))
    matched_pred = set()
    with open(MATCHED_STREAM) as f:
        for line in f:
            if line.strip():
                matched_pred.add(json.loads(line)["sample_id"])
    pred_len = {}
    pred_score = {}
    pred_area = {}
    for vid, tmap in tracks.items():
        for tid, t in tmap.items():
            sid = f"P{vid}_{tid}"
            pred_len[sid] = len(t["frames"])
            pred_score[sid] = float(np.mean(t["scores"])) if t["scores"] else 0.0
            pred_area[sid] = float(np.mean([box_area(b) for b in t["boxes"]])) if t["boxes"] else 0.0

    def summarize(ids, label):
        if not ids:
            return {
                "label": label, "count": 0,
                "length_mean": 0, "length_median": 0, "len1_ratio": 0,
                "score_mean": 0, "area_mean": 0,
            }
        L = np.asarray([pred_len[s] for s in ids])
        return {
            "label": label,
            "count": len(ids),
            "length_mean": float(L.mean()),
            "length_median": float(np.median(L)),
            "len1_ratio": float((L == 1).mean()),
            "score_mean": float(np.mean([pred_score[s] for s in ids])),
            "area_mean": float(np.mean([pred_area[s] for s in ids])),
        }

    matched_ids = [s for s in pred_len if s in matched_pred]
    unmatched_ids = [s for s in pred_len if s not in matched_pred]
    matched_stats = summarize(matched_ids, "matched")
    unmatched_stats = summarize(unmatched_ids, "unmatched")
    for row in (matched_stats, unmatched_stats):
        with open(OUT / f"{row['label']}_pred_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)

    gt_len_known, gt_len_unknown = [], []
    for sid, (vid, tid, rec) in private_tracks.items():
        (gt_len_known if rec["is_known"] else gt_len_unknown).append(len(rec["frames"]))
    gt_stats = {
        "gt_known_tracks": len(gt_len_known),
        "gt_unknown_tracks": len(gt_len_unknown),
        "gt_known_length_mean": float(np.mean(gt_len_known)) if gt_len_known else 0.0,
        "gt_unknown_length_mean": float(np.mean(gt_len_unknown)) if gt_len_unknown else 0.0,
        "matched_gt_known": sum(1 for s in matched_gt_ids if private_tracks[s][2]["is_known"]),
        "matched_gt_unknown": sum(1 for s in matched_gt_ids if not private_tracks[s][2]["is_known"]),
        "matched_gt_known_length_mean": float(
            np.mean([len(private_tracks[s][2]["frames"]) for s in matched_gt_ids if private_tracks[s][2]["is_known"]])
        ) if matched_gt_ids else 0.0,
        "matched_gt_unknown_length_mean": float(
            np.mean([len(private_tracks[s][2]["frames"]) for s in matched_gt_ids if not private_tracks[s][2]["is_known"]])
        ) if matched_gt_ids else 0.0,
    }
    return matched_stats, unmatched_stats, gt_stats


def temporal_iou(gt_frames, gt_boxes, pred_frames, pred_boxes):
    gset, pset = set(gt_frames), set(pred_frames)
    common = gset & pset
    if not common:
        return 0.0
    gidx = {f: i for i, f in enumerate(gt_frames)}
    pidx = {f: i for i, f in enumerate(pred_frames)}
    s = 0.0
    for f in common:
        a = gt_boxes[gidx[f]]
        b = pred_boxes[pidx[f]]
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        s += inter / union if union > 0 else 0.0
    return s / len(gset | pset)


def gt_fragmentation(tracks, by_video, private_tracks):
    rows = []
    pair_counts = []
    for vid, gt_map in by_video.items():
        pred_map = tracks.get(vid, {})
        if not pred_map:
            continue
        # potential stitchable pairs: tracks in same video, gap <= 30
        pred_meta = []
        for tid, t in pred_map.items():
            frames = t["frames"]
            pred_meta.append((tid, frames[0], frames[-1], frames[0] - frames[-1]))
        pred_meta.sort(key=lambda x: x[2])  # by end
        ends = [m[2] for m in pred_meta]
        starts = [m[1] for m in pred_meta]
        import bisect
        pairs = 0
        for i, m in enumerate(pred_meta):
            # tracks j with end_j < start_i and gap <= 30 (i after j)
            lo = bisect.bisect_left(ends, m[1] - 30 - 1)
            for j in range(lo, i):
                if m[1] - ends[j] <= 30:
                    pairs += 1
        pair_counts.append(pairs)

        for tid, grec in gt_map.items():
            sid = f"{vid}_{tid}"
            if sid not in private_tracks:
                continue
            ious = []
            for ptid, pt in pred_map.items():
                v = temporal_iou(grec["frames"], grec["boxes"], pt["frames"], pt["boxes"])
                if v > 0:
                    ious.append((v, ptid))
            ious.sort(reverse=True)
            rows.append(
                {
                    "sample_id": sid,
                    "video_id": vid,
                    "gt_track_id": tid,
                    "is_known": int(private_tracks[sid][2]["is_known"]),
                    "gt_len": len(grec["frames"]),
                    "num_fragments_iou_gt0": len(ious),
                    "num_fragments_iou_ge0_05": sum(1 for v, _ in ious if v >= 0.05),
                    "num_fragments_iou_ge0_3": sum(1 for v, _ in ious if v >= 0.3),
                    "best_iou": ious[0][0] if ious else 0.0,
                    "second_iou": ious[1][0] if len(ious) > 1 else 0.0,
                    "cumulative_iou": sum(v for v, _ in ious),
                    "matched_iou0_5": int(private_tracks[sid][2]["sample_id"] in set(json.loads(MATCHED_GT_IDS.read_text()))),
                }
            )
    with open(OUT / "gt_fragmentation_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        w.writeheader()
        w.writerows(rows)
    return {
        "num_gt_rows": len(rows),
        "stitch_candidate_pairs_gap30_sum": int(sum(pair_counts)),
        "stitch_candidate_pairs_gap30_mean_per_video": float(np.mean(pair_counts)) if pair_counts else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["basic", "gtfrag"], default="basic")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "basic":
        per_video = load_raw_predictions()
        tracks = reconstruct_tracks(per_video)
        with open(COMPACT_CACHE, "wb") as f:
            pickle.dump(tracks, f, protocol=4)
        stats = track_stats(tracks)
        sem = semantics_check(tracks)
        adapter = adapter_consistency(tracks)
        by_video, private_tracks = load_gt()
        matched, unmatched, gt_stats = matched_unmatched(tracks, private_tracks)
        result = {
            "track_stats": stats,
            "semantics": sem,
            "adapter_consistency": adapter,
            "matched_pred": matched,
            "unmatched_pred": unmatched,
            "gt_stats": gt_stats,
            "rss_gb": rss_gb(),
        }
        (OUT / "pred_track_stats.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str)[:4000], flush=True)

    elif args.stage == "gtfrag":
        with open(COMPACT_CACHE, "rb") as f:
            tracks = pickle.load(f)
        by_video, private_tracks = load_gt()
        res = gt_fragmentation(tracks, by_video, private_tracks)
        res["rss_gb"] = rss_gb()
        (OUT / "gt_fragmentation_summary.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
