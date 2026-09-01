#!/usr/bin/env python3
"""Reproduce the historical OVTR Q0 full-sequence output without retraining.

The Q0 checkpoint was trained for seven complete epochs in Phase4Q.  This
script treats its score-corrected TAO JSON as a read-only lineage artifact,
computes class-agnostic proposal recall from the TAO validation annotations,
and prepares a TrackEval-compatible symlink layout.  No GT is passed to the
model (there is no model invocation here); annotations are used only for the
offline evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = pathlib.Path(__file__).resolve().parents[2]
Q0_JSON = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
Q0_CSV = ROOT / "outputs/iclr27_phase4q/q0_long/proposals_dev.csv"
GT_JSON = ROOT / "data/external_annotations/ovtr/validation_ours_v1.json"
TE_GT_JSON = ROOT / "third_party/TrackEval/data/gt/tao/tao_validation/validation.json"
OUT = ROOT / "outputs/iclr27_phase68"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def xywh_to_xyxy(b: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, b[:4])
    return x, y, x + max(w, 0.0), y + max(h, 0.0)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def build_recall(gt: Dict[int, List[dict]], pred: Dict[int, List[dict]], topks: Sequence[int], thresholds: Sequence[float]) -> Dict[str, Any]:
    # One record per annotated object.  The denominator is all annotated rows;
    # no hard event or image is removed.
    totals = len([a for v in gt.values() for a in v])
    out: Dict[str, Any] = {"gt_rows": totals, "topk": {}}
    for k in topks:
        by_t: Dict[str, Any] = {}
        hits = {t: 0 for t in thresholds}
        best_values: List[float] = []
        images_with_gt = 0
        images_with_prediction = 0
        for image_id, anns in gt.items():
            if not anns:
                continue
            images_with_gt += 1
            ps = sorted(pred.get(image_id, []), key=lambda z: float(z.get("score", 0.0)), reverse=True)
            if ps:
                images_with_prediction += 1
            ps = ps[:k] if k > 0 else ps
            for ann in anns:
                best = max((iou(ann["bbox"], p["bbox"]) for p in ps), default=0.0)
                best_values.append(best)
                for t in thresholds:
                    if best >= t:
                        hits[t] += 1
        n = max(totals, 1)
        by_t = {f"{t:.1f}": {"matched_rows": hits[t], "recall": hits[t] / n} for t in thresholds}
        out["topk"][str(k)] = {
            "thresholds": by_t,
            "mean_best_iou": statistics.fmean(best_values) if best_values else 0.0,
            "median_best_iou": statistics.median(best_values) if best_values else 0.0,
            "images_with_gt": images_with_gt,
            "images_with_prediction": images_with_prediction,
        }
    return out


def load_records(gt_path: pathlib.Path, pred_path: pathlib.Path) -> Tuple[Dict[int, List[dict]], Dict[int, List[dict]], Dict[int, dict]]:
    with gt_path.open() as f:
        d = json.load(f)
    gt: Dict[int, List[dict]] = defaultdict(list)
    for ann in d.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt[int(ann["image_id"])].append(ann)
    with pred_path.open() as f:
        p = json.load(f)
    pred: Dict[int, List[dict]] = defaultdict(list)
    for x in p:
        if "image_id" in x and "bbox" in x:
            pred[int(x["image_id"])].append(x)
    images = {int(x["id"]): x for x in d.get("images", [])}
    return gt, pred, images


def prepare_trackeval_layout(q0: pathlib.Path) -> Dict[str, str]:
    tracker_dir = OUT / "trackeval" / "trackers" / "OVTR_Q0" / "data"
    tracker_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = OUT / "trackeval" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    links = {
        str(tracker_dir / "tao_track.json"): str(q0.resolve()),
        str(gt_dir / "validation.json"): str(TE_GT_JSON.resolve()),
    }
    for dst_s, src_s in links.items():
        dst = pathlib.Path(dst_s)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and os.path.realpath(dst) == src_s:
                continue
            raise RuntimeError(f"refusing to overwrite existing TrackEval path: {dst}")
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(src_s, tmp)
        os.replace(tmp, dst)
    return links


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q0-json", type=pathlib.Path, default=Q0_JSON)
    ap.add_argument("--gt-json", type=pathlib.Path, default=GT_JSON)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    q0, gt_path = args.q0_json.resolve(), args.gt_json.resolve()
    if not q0.exists() or not gt_path.exists() or not Q0_CSV.exists():
        raise FileNotFoundError(f"missing read-only lineage asset: {q0}, {gt_path}, {Q0_CSV}")
    gt, pred, images = load_records(gt_path, q0)
    recall = build_recall(gt, pred, topks=(1, 5, 20, 100, 0), thresholds=(0.3, 0.5, 0.7))
    # Track-level continuity proxy: for each GT track, count frames with any
    # class-agnostic prediction at IoU>=.5 and longest contiguous run.  This is
    # explicitly diagnostic; official HOTA/CLEAR/Identity come from TrackEval.
    by_track: Dict[Tuple[int, int], List[Tuple[int, float]]] = defaultdict(list)
    anns_by_image = gt
    for image_id, anns in anns_by_image.items():
        ps = pred.get(image_id, [])
        frame = int(images.get(image_id, {}).get("frame_id", images.get(image_id, {}).get("frame_index", 0)))
        for ann in anns:
            best = max((iou(ann["bbox"], p["bbox"]) for p in ps), default=0.0)
            by_track[(int(images.get(image_id, {}).get("video_id", ann.get("video_id", -1))), int(ann.get("track_id", ann.get("instance_id", -1))))].append((frame, best))
    track_stats = []
    for key, vals in by_track.items():
        vals.sort()
        hits = [int(v >= 0.5) for _, v in vals]
        longest = cur = 0
        for h in hits:
            cur = cur + 1 if h else 0
            longest = max(longest, cur)
        track_stats.append({"video_id": key[0], "track_id": key[1], "frames": len(vals), "reliable_frames": sum(hits), "reliable_fraction": sum(hits) / max(len(vals), 1), "longest_reliable_run": longest})
    track_summary = {
        "gt_tracks": len(track_stats),
        "mean_reliable_fraction": statistics.fmean(x["reliable_fraction"] for x in track_stats) if track_stats else 0.0,
        "median_reliable_fraction": statistics.median(x["reliable_fraction"] for x in track_stats) if track_stats else 0.0,
        "mean_longest_reliable_run": statistics.fmean(x["longest_reliable_run"] for x in track_stats) if track_stats else 0.0,
        "tracks_reliable_at_least_once": sum(x["reliable_frames"] > 0 for x in track_stats),
        "details": track_stats,
        "definition": "diagnostic class-agnostic IoU>=0.5 frame proxy; not a TrackEval score",
    }
    links = prepare_trackeval_layout(q0)
    manifest = {
        "protocol": "trackocd_phase68_ovtr_q0_full_sequence_reproduction",
        "q0_json": {"path": str(q0), "bytes": q0.stat().st_size, "sha256": sha256(q0), "mtime": q0.stat().st_mtime},
        "q0_proposals_csv": {"path": str(Q0_CSV.resolve()), "bytes": Q0_CSV.stat().st_size, "sha256": sha256(Q0_CSV), "mtime": Q0_CSV.stat().st_mtime},
        "gt_json": {"path": str(gt_path), "bytes": gt_path.stat().st_size, "sha256": sha256(gt_path)},
        "trackeval_gt_json": {"path": str(TE_GT_JSON.resolve()), "bytes": TE_GT_JSON.stat().st_size, "sha256": sha256(TE_GT_JSON)},
        "prediction_count": len([x for v in pred.values() for x in v]),
        "prediction_images": len(pred),
        "gt_images": len(gt),
        "gt_annotations": sum(len(v) for v in gt.values()),
        "recall": recall,
        "track_continuity_proxy": track_summary,
        "trackeval_symlinks": links,
        "score_source": "q0 score-corrected tao_track.json; score is detection score, not bbox y2",
        "labels_used_for_model": False,
        "sealed_public_q1_accessed": False,
        "held_event_gt_used_for_model": False,
    }
    atomic_json(OUT / "audit/full_sequence_baseline.json", manifest)
    atomic_json(OUT / "metrics/ovtr_baseline/proposal_recall.json", {"protocol": manifest["protocol"], "recall": recall, "source_sha256": manifest["q0_json"]["sha256"]})
    atomic_json(OUT / "metrics/ovtr_baseline/track_continuity_proxy.json", track_summary)
    (OUT / "completion").mkdir(parents=True, exist_ok=True)
    (OUT / "completion/phase68_reproduction.done").write_text("complete\n")
    print(json.dumps({"q0_sha256": manifest["q0_json"]["sha256"], "predictions": manifest["prediction_count"], "gt_rows": manifest["gt_annotations"], "recall_top20_iou05": recall["topk"]["20"]["thresholds"]["0.5"]["recall"], "trackeval_gt": links[str(OUT / 'trackeval' / 'gt' / 'validation.json')]}, indent=2))


if __name__ == "__main__":
    main()
