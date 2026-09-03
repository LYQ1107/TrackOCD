#!/usr/bin/env python3
"""Evaluate a frozen Phase81P association stream and physical-track proxies.

The learned stream is produced causally from the frozen Q0 proposal rows.  GT
annotations are joined only after inference to measure diagnostic tracking
proxies; they are never passed to the model or used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(
    "/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl"
)
TRAIN_ANN = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json"
)
REPLAY = ROOT / "scripts/iclr27_phase81p/replay_association.py"


def _iou(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    ab = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def _xywh_to_xyxy(b: list[float]) -> list[float]:
    return [float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])]


def load_gt() -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Load TRAIN GT for post-inference scoring only."""
    data = json.loads(TRAIN_ANN.read_text())
    out: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for a in data.get("annotations", []):
        out[(int(a.get("video_id", -1)), int(a["image_id"]))].append(
            {
                "track_id": int(a.get("track_id", -1)),
                "bbox": _xywh_to_xyxy([float(x) for x in a["bbox"]]),
            }
        )
    return out


def load_native() -> list[dict[str, Any]]:
    rows = []
    with NATIVE.open() as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                if x.get("bbox_xyxy") is not None:
                    rows.append(x)
    return rows


def physical_summary(rows: list[dict[str, Any]], gt_by_image: dict[tuple[int, int], list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute conservative GT-joined physical proxies after inference.

    A GT annotation is assigned to the highest-IoU prediction in its image. A
    switch is counted only between consecutive visible GT frames with reliable
    IoU (>=0.5), avoiding assumptions about absent detections. This is a
    diagnostic comparison, not a replacement for full TrackEval.
    """
    by_image: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    by_video: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    track_ids = set()
    lifecycle = collections.Counter()
    for r in rows:
        v, im = int(r.get("video_id", -1)), int(r.get("image_id", -1))
        by_image[(v, im)].append(r)
        by_video[v].append(r)
        track_ids.add(int(r.get("physical_track_id", -1)))
        lifecycle[str(r.get("lifecycle_action", r.get("lifecycle", "unknown")))] += 1

    gt_sequences: dict[tuple[int, int], list[tuple[int, int | None, float]]] = collections.defaultdict(list)
    pred_to_gt: dict[int, set[tuple[int, int]]] = collections.defaultdict(set)
    gt_to_pred: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    reliable_assignments = 0
    for key, gts in gt_by_image.items():
        candidates = by_image.get(key, [])
        for gt in gts:
            best = max(candidates, key=lambda r: _iou(r.get("bbox_xyxy"), gt["bbox"]), default=None)
            score = _iou(best.get("bbox_xyxy"), gt["bbox"]) if best else 0.0
            pid = int(best["physical_track_id"]) if best is not None and score >= 0.5 else None
            tid = (key[0], int(gt["track_id"]))
            # frame index is not guaranteed to be dense; image_id is causal in
            # the native stream and is sufficient for ordered diagnostics.
            gt_sequences[tid].append((key[1], pid, score))
            if pid is not None:
                reliable_assignments += 1
                pred_to_gt[pid].add(tid)
                gt_to_pred[tid].add(pid)

    switches = 0
    fragmented_tracks = 0
    reliable_gt_tracks = 0
    for tid, seq in gt_sequences.items():
        seq.sort(key=lambda x: x[0])
        ids = [pid for _, pid, _ in seq if pid is not None]
        if not ids:
            continue
        reliable_gt_tracks += 1
        if len(set(ids)) > 1:
            fragmented_tracks += 1
        prev = ids[0]
        for pid in ids[1:]:
            if pid != prev:
                switches += 1
            prev = pid
    merged_pred_tracks = sum(1 for ids in pred_to_gt.values() if len(ids) > 1)
    duplicate_births = sum(max(0, len(ids) - 1) for ids in gt_to_pred.values())
    return {
        "rows": len(rows),
        "videos": len(by_video),
        "unique_physical_tracks": len(track_ids),
        "lifecycle_counts": dict(lifecycle),
        "gt_annotations": sum(len(v) for v in gt_by_image.values()),
        "reliable_gt_assignments_iou_ge_0.5": reliable_assignments,
        "reliable_gt_tracks": reliable_gt_tracks,
        "gt_track_switches": switches,
        "fragmented_gt_tracks": fragmented_tracks,
        "merged_pred_tracks": merged_pred_tracks,
        "duplicate_birth_proxy": duplicate_births,
        "definition": "post-inference highest-IoU GT join; diagnostic proxy, not TrackEval",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="physical_formal")
    ap.add_argument("--motion", action="store_true", help="use the registered causal velocity route")
    args = ap.parse_args()

    # Import lazily so this evaluator shares exactly the Phase81P runtime.
    import importlib.util

    spec = importlib.util.spec_from_file_location("phase81p_replay", REPLAY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {REPLAY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpt = Path(args.checkpoint)
    stream = mod.run_stream(ckpt, args.device, use_motion=args.motion)
    gt_by_image = load_gt()
    native = load_native()
    result = {
        "schema_version": "phase81p.physical_replay.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "native_lineage": str(NATIVE),
        "native_lineage_sha256": hashlib.sha256(NATIVE.read_bytes()).hexdigest(),
        "train_annotations": str(TRAIN_ANN),
        "protocol": {
            "labels_joined_before_inference": False,
            "future_rows_or_tracks": False,
            "ids_as_model_input": False,
            "positive_denominator": 76,
            "negative_denominator": 76,
            "causal_motion_prediction": bool(args.motion),
        },
        "learned": physical_summary(stream, gt_by_image),
        "q0_native": physical_summary(native, gt_by_image),
    }
    out = ROOT / f"outputs/iclr27_phase81p/metrics/physical_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("." + out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(json.dumps({"learned": result["learned"], "q0_native": result["q0_native"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
