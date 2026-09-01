#!/usr/bin/env python3
"""Candidate architecture C: causal tracklet stitching feasibility.

C0: motion-only stitching over all 649,378 SimOWT tracklets (per-video,
causal, greedy one-predecessor/one-successor, gap<=max_gap, no overlap).
C1: motion + identity (matched-subset diagnostic, DINO track feature).
C2: motion + identity + weak mature-category compatibility (matched subset).
Oracle category-cue diagnostic: same pipeline with GT category as the cue
(diagnostic only).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.track_matching import (
    load_gt_tracks, match_tracks, temporal_iou,
)
from src.ocd_v2.common import load_mean_features, load_val_labels

COMPACT = PROJECT_ROOT / "outputs" / "arch1_5" / "track_audit" / "pred_tracks_compact.pkl"
OUT = PROJECT_ROOT / "outputs" / "trackocd_v1" / "metrics"
RUNS = PROJECT_ROOT / "runs" / "trackocd_v1"


def load_tracks():
    with open(COMPACT, "rb") as f:
        return pickle.load(f)


def track_arrays(vid, tracks):
    """Vectorized per-video tracklet metadata."""
    tids = sorted(tracks.keys())
    n = len(tids)
    start = np.zeros(n, dtype=np.int64)
    end = np.zeros(n, dtype=np.int64)
    cx0 = np.zeros(n)
    cy0 = np.zeros(n)
    w0 = np.zeros(n)
    h0 = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    prev_w = np.zeros(n)
    prev_h = np.zeros(n)
    for i, tid in enumerate(tids):
        t = tracks[tid]
        frames = t["frames"]
        boxes = t["boxes"]
        start[i] = frames[0]
        end[i] = frames[-1]
        b0 = boxes[-1]
        x0, y0, x1, y1 = b0
        w0[i], h0[i] = x1 - x0, y1 - y0
        cx0[i], cy0[i] = (x0 + x1) / 2, (y0 + y1) / 2
        if len(boxes) >= 2:
            b1 = boxes[-2]
            px1, py1, px2, py2 = b1
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            dt = max(1, frames[-1] - frames[-2])
            vx[i] = (cx0[i] - pcx) / dt
            vy[i] = (cy0[i] - pcy) / dt
            prev_w[i], prev_h[i] = px2 - px1, py2 - py1
        else:
            prev_w[i], prev_h[i] = w0[i], h0[i]
    return tids, start, end, cx0, cy0, w0, h0, vx, vy


def stitch_video(tracks, max_gap=30, dist_scale=3.0):
    tids, start, end, cx0, cy0, w0, h0, vx, vy = track_arrays(None, tracks)
    n = len(tids)
    succ = np.full(n, -1, dtype=np.int64)
    costs = {}
    order = np.argsort(start)
    for i in order:
        si = start[i]
        lo = np.searchsorted(end, si - max_gap - 1, side="left")
        hi = np.searchsorted(end, si, side="left")
        if lo >= hi:
            continue
        js = np.arange(lo, hi)
        valid = (succ[js] == -1)
        if not valid.any():
            continue
        js = js[valid]
        gap = si - end[js]
        scale = (w0[js] + h0[js] + w0[i] + h0[i]) / 4.0 + 1e-6
        pred_cx = cx0[js] + vx[js] * gap
        pred_cy = cy0[js] + vy[js] * gap
        dist = np.hypot(pred_cx - cx0[i], pred_cy - cy0[i])
        ok = (dist <= dist_scale * scale) & (gap >= 1)
        # size compatibility
        size_j = np.sqrt(w0[js] * h0[js] + 1e-6)
        size_i = np.sqrt(w0[i] * h0[i] + 1e-6)
        ratio = size_j / size_i
        ok &= (ratio >= 0.3) & (ratio <= 3.0)
        if not ok.any():
            continue
        js = js[ok]
        gap = gap[ok]
        dist = dist[ok]
        cost = dist / (scale[ok] + 1e-6) + 0.15 * gap
        best = int(js[np.argmin(cost)])
        succ[best] = i
        costs[(int(tids[best]), int(tids[i]))] = float(cost.min())
    # build stitched tracks
    pred_by_first = {}
    root = {}
    for i in range(n):
        root[i] = i
    # union-find along succ
    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x
    for j in range(n):
        if succ[j] >= 0:
            root[find(int(succ[j]))] = find(j)
    chains = defaultdict(list)
    for i in range(n):
        chains[find(i)].append(i)
    stitched = {}
    new_tid = 0
    for members in chains.values():
        members = sorted(members, key=lambda i: start[i])
        frames = []
        boxes = []
        scores = []
        for i in members:
            t = tracks[tids[i]]
            frames.extend(t["frames"])
            boxes.extend(t["boxes"])
            scores.extend(t["scores"])
        stitched[new_tid] = {
            "frames": frames, "boxes": boxes, "scores": scores,
            "source_tracklets": [tids[i] for i in members],
        }
        new_tid += 1
    return stitched, len(costs)


def to_anns(stitched):
    anns = {}
    for tid, t in stitched.items():
        d = {}
        for f, b in zip(t["frames"], t["boxes"]):
            x, y, w, h = b
            d[f] = [x, y, x + w, y + h]
        anns[tid] = d
    return anns


def coverage(gt_anns, pred_anns, private):
    matches = match_tracks(gt_anns, pred_anns, 0.5)
    gt_vid, _ = load_gt_tracks()
    gt_sample = {
        (vid, tid): rec["sample_id"] for vid, td in gt_vid.items() for tid, rec in td.items()
    }
    matched_gt = {gt_sample[(vid, g)] for vid, g, p, iou in matches if (vid, g) in gt_sample}
    matched_gt &= set(private)
    all_gt = set(private)
    known = sum(1 for s in matched_gt if private[s]["is_known"])
    unknown = sum(1 for s in matched_gt if not private[s]["is_known"])
    all_known = sum(1 for s in all_gt if private[s]["is_known"])
    all_unknown = sum(1 for s in all_gt if not private[s]["is_known"])
    return {
        "pred_track_count": int(sum(len(v) for v in pred_anns.values())),
        "matched": len(matches),
        "gt_coverage": len(matched_gt) / len(all_gt),
        "known_coverage": known / all_known,
        "unknown_coverage": unknown / all_unknown,
    }


def run_c0(max_gap=30):
    tracks = load_tracks()
    gt_vid, gt_anns_all = load_gt_tracks()
    private = load_val_labels()
    gt_sample = {
        (vid, tid): rec["sample_id"] for vid, td in gt_vid.items() for tid, rec in td.items()
    }
    gt_anns = {
        vid: {tid: anns for tid, anns in tmap.items() if gt_sample.get((vid, tid)) in private}
        for vid, tmap in gt_anns_all.items()
    }
    all_stitched = {}
    all_anns = {}
    total_edges = 0
    for vid, tmap in tracks.items():
        if not tmap:
            continue
        stitched, edges = stitch_video(tmap, max_gap=max_gap)
        total_edges += edges
        all_stitched[vid] = stitched
        all_anns[vid] = to_anns(stitched)
    res = coverage(gt_anns, all_anns, private)
    res["total_stitch_edges"] = total_edges
    # fragment stats vs GT (IoU>0)
    frags = []
    for vid, gt_map in gt_anns.items():
        p_map = all_anns.get(vid, {})
        if not p_map:
            continue
        for g, ganns in gt_map.items():
            cnt = sum(1 for panns in p_map.values() if temporal_iou(ganns, panns) > 0)
            if cnt > 0:
                frags.append(cnt)
    res["overlapped_gt_mean_fragments"] = float(np.mean(frags)) if frags else 0.0
    res["overlapped_gt_count"] = len(frags)
    lens = [len(t["frames"]) for v in all_stitched.values() for t in v.values()]
    res["stitched_length_mean"] = float(np.mean(lens)) if lens else 0.0
    res["stitched_length_median"] = float(np.median(lens)) if lens else 0.0
    res["single_frame_ratio"] = float(sum(1 for L in lens if L == 1) / len(lens)) if lens else 0.0
    (RUNS / "c0_stitched_tracks.pkl").write_bytes(pickle.dumps(all_stitched))
    (OUT / "bidirectional_feasibility_c0.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def run_c1_c2(identity_thr=0.60, max_gap=30):
    """Matched-subset causal stitching with identity (C1), weak category
    compatibility (C2) and oracle category cue (diagnostic)."""
    from src.trackocd_v1.modular import simulate_ncm
    from src.ocd_v2.common import load_train_known, build_prototypes

    rows = []
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "pred_track_stream_matched_iou0.5.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["stream_order"])
    feats = load_mean_features("dinov2", "pred_tracks_mean")
    # recompute per-pred matched GT
    gt_vid, gt_anns_all = load_gt_tracks()
    private = load_val_labels()
    gt_sample = {
        (vid, tid): rec["sample_id"] for vid, td in gt_vid.items() for tid, rec in td.items()
    }
    pred_anns = {}
    for r in rows:
        d = {}
        for f, b in zip(r["frame_ids"], r["boxes_xyxy"]):
            d[f] = b
        pred_anns.setdefault(r["video_id"], {})[r["track_id"]] = d
    pred_to_gt = {}
    for vid, tmap in pred_anns.items():
        gmap = gt_anns_all.get(vid, {})
        for ptid, panns in tmap.items():
            best = None
            best_iou = 0.0
            for gtid, ganns in gmap.items():
                if gt_sample.get((vid, gtid)) not in private:
                    continue
                v = temporal_iou(ganns, panns)
                if v > best_iou:
                    best_iou, best = v, gt_sample[(vid, gtid)]
            pred_to_gt[f"P{vid}_{ptid}"] = best
    # online discovery over the matched stream (corrected B2 memory)
    tr_feats, labels = load_train_known("dinov2")
    protos = build_prototypes(tr_feats, labels, set(labels.values()))
    preds_log = simulate_ncm(rows, feats, protos, 0.45)
    cat_of = {p["sample_id"]: (
        p["semantic_category_id"] if p["prediction_type"] == "known" else p["virtual_category_id"]
    ) for p in preds_log}

    def study(cue):
        merged_correct = 0
        merged_wrong = 0
        accepted = 0
        candidates = 0
        motion_gated = 0
        stitched = {}
        succ = {}
        groups = defaultdict(list)
        for r in rows:
            groups[r["video_id"]].append(r)
        for vid, g in groups.items():
            g = sorted(g, key=lambda r: r["frame_ids"][-1])
            n = len(g)
            for i in range(n):
                si = g[i]["frame_ids"][0]
                best_j, best_cost = None, 1e18
                for j in range(i):
                    if succ.get(j) is not None:
                        continue
                    ej = g[j]["frame_ids"][-1]
                    gap = si - ej
                    if gap < 1 or gap > max_gap:
                        continue
                    candidates += 1
                    b0 = g[j]["boxes_xyxy"][-1]
                    b1 = g[j]["boxes_xyxy"][-2] if len(g[j]["boxes_xyxy"]) >= 2 else b0
                    v = ((b0[0] + b0[2]) / 2 - (b1[0] + b1[2]) / 2,
                         (b0[1] + b0[3]) / 2 - (b1[1] + b1[3]) / 2)
                    dt = max(1, g[j]["frame_ids"][-1] - g[j]["frame_ids"][-2]) if len(g[j]["frame_ids"]) >= 2 else 1
                    pcx = (b0[0] + b0[2]) / 2 + v[0] / dt * gap
                    pcy = (b0[1] + b0[3]) / 2 + v[1] / dt * gap
                    ci = ((g[i]["boxes_xyxy"][0][0] + g[i]["boxes_xyxy"][0][2]) / 2,
                          (g[i]["boxes_xyxy"][0][1] + g[i]["boxes_xyxy"][0][3]) / 2)
                    dist = ((pcx - ci[0]) ** 2 + (pcy - ci[1]) ** 2) ** 0.5
                    scale = (b0[2] - b0[0] + b0[3] - b0[1] +
                             g[i]["boxes_xyxy"][0][2] - g[i]["boxes_xyxy"][0][0] +
                             g[i]["boxes_xyxy"][0][3] - g[i]["boxes_xyxy"][0][1]) / 4.0
                    if dist > 3.0 * max(scale, 1e-6):
                        continue
                    motion_gated += 1
                    cost = dist / max(scale, 1e-6) + 0.15 * gap
                    sid_i = g[i]["sample_id"]
                    sid_j = g[j]["sample_id"]
                    id_sim = float(np.dot(feats[sid_i], feats[sid_j]))
                    if id_sim < identity_thr:
                        continue
                    cost += (1.0 - id_sim) * 2.0
                    if cue is not None:
                        cj = cue.get(sid_j)
                        ci2 = cue.get(sid_i)
                        if cj is not None and ci2 is not None and cj == ci2:
                            cost -= 0.5
                    if cost < best_cost:
                        best_cost, best_j = cost, j
                if best_j is not None:
                    succ[best_j] = i
                    accepted += 1
                    gt_i = pred_to_gt.get(g[i]["sample_id"])
                    gt_j = pred_to_gt.get(g[best_j]["sample_id"])
                    if gt_i is not None and gt_j is not None and gt_i == gt_j:
                        merged_correct += 1
                    else:
                        merged_wrong += 1
        return {
            "motion_candidates": candidates,
            "motion_gated": motion_gated,
            "accepted_stitches": accepted,
            "correct_merges": merged_correct,
            "wrong_merges": merged_wrong,
            "correct_rate": merged_correct / accepted if accepted else 0.0,
        }

    out = {
        "c1_identity_only": study(None),
        "c2_identity_category": study(cat_of),
        "oracle_category_cue": study({sid: pred_to_gt.get(sid) for sid in pred_to_gt}),
    }
    (OUT / "bidirectional_feasibility_c1c2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-gap", type=int, default=30)
    ap.add_argument("--video-limit", type=int, default=0)
    ap.add_argument("--c1c2", action="store_true")
    ap.add_argument("--identity-thr", type=float, default=0.60)
    args = ap.parse_args()
    if args.c1c2:
        run_c1_c2(identity_thr=args.identity_thr, max_gap=args.max_gap)
    else:
        run_c0(max_gap=args.max_gap)


if __name__ == "__main__":
    main()
