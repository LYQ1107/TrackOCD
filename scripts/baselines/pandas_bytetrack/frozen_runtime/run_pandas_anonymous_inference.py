#!/usr/bin/env python3
"""Run frozen PANDAS and write compact per-video anonymous detections."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _quantile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def write_video_npz(
    path: Path,
    rows: list[dict],
    foreground_floor: float | None,
) -> dict[str, int]:
    frame_index = np.asarray([row["frame_index"] for row in rows], dtype=np.int64)
    image_id = np.asarray([row["image_id"] for row in rows], dtype=np.int64)
    filtered_boxes = []
    filtered_scores = []
    filtered_foreground_scores = []
    filtered_prototype_ids = []
    before_detections = 0
    after_detections = 0
    zero_detection_frames = 0
    for row in rows:
        boxes_row = np.asarray(row["boxes"], dtype=np.float32).reshape(-1, 4)
        scores_row = np.asarray(row["scores"], dtype=np.float32).reshape(-1)
        foreground_row = np.asarray(row["foreground_scores"], dtype=np.float32).reshape(-1)
        prototype_row = np.asarray(row["prototype_id"], dtype=np.int32).reshape(-1)
        if not (len(boxes_row) == len(scores_row) == len(foreground_row) == len(prototype_row)):
            raise RuntimeError("PANDAS output fields have different detection counts")
        if not np.isfinite(foreground_row).all() or not ((0.0 <= foreground_row).all() and (foreground_row <= 1.0).all()):
            raise ValueError("foreground_scores must be finite and within [0, 1]")
        before_detections += len(boxes_row)
        if foreground_floor is None:
            keep = np.ones(len(boxes_row), dtype=bool)
        else:
            keep = foreground_row >= foreground_floor
        filtered_boxes.append(boxes_row[keep])
        filtered_scores.append(scores_row[keep])
        filtered_foreground_scores.append(foreground_row[keep])
        filtered_prototype_ids.append(prototype_row[keep])
        after_detections += int(np.sum(keep))
        if not np.any(keep):
            zero_detection_frames += 1

    counts = [len(value) for value in filtered_boxes]
    offsets = np.concatenate(([0], np.cumsum(np.asarray(counts, dtype=np.int64))))
    if offsets[-1]:
        boxes = np.concatenate(filtered_boxes, axis=0).astype(np.float32)
        scores = np.concatenate(filtered_scores, axis=0).astype(np.float32)
        foreground_scores = np.concatenate(filtered_foreground_scores, axis=0).astype(np.float32)
        prototype_id = np.concatenate(filtered_prototype_ids, axis=0).astype(np.int32)
    else:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        foreground_scores = np.empty((0,), dtype=np.float32)
        prototype_id = np.empty((0,), dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_index=frame_index,
        image_id=image_id,
        frame_offsets=offsets,
        boxes=boxes,
        # ``scores`` remains the PANDAS semantic/prototype score for the
        # existing detection evaluator. Tracking must read the separate
        # ``foreground_scores`` field.
        scores=scores,
        foreground_scores=foreground_scores,
        prototype_id=prototype_id,
    )
    return {
        "frames": len(rows),
        "detections_before_floor": before_detections,
        "detections_after_floor": after_detections,
        "zero_detection_frames_after_floor": zero_detection_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["annotated", "full"], required=True)
    parser.add_argument("--pandas-root", type=Path, required=True)
    parser.add_argument("--tao-annotation", type=Path, required=True)
    parser.add_argument("--tao-frame-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit-records", type=int, default=None,
                        help="annotated-mode smoke limit; omitted for the full universe")
    parser.add_argument("--limit-videos", type=int, default=None,
                        help="full-mode smoke limit; omitted for the full universe")
    parser.add_argument("--skip-existing", action="store_true",
                        help="full-mode: reuse existing per-video NPZ outputs")
    parser.add_argument("--tracking-export-floor", type=float, default=None,
                        help="full-mode foreground floor; defaults to 0.01 (annotated defaults to 0)")
    parser.add_argument("--score-sample-limit", type=int, default=200000,
                        help="deterministic cap for score quantile audit samples")
    args = parser.parse_args()
    if args.score_sample_limit < 0:
        raise ValueError("score-sample-limit must be non-negative")
    foreground_floor = args.tracking_export_floor
    if foreground_floor is None:
        foreground_floor = 0.0 if args.mode == "annotated" else 0.01
    if not 0.0 <= foreground_floor <= 1.0:
        raise ValueError("tracking-export-floor must be within [0, 1]")
    existing_video_ids = set()

    script_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_root))
    from pandas_runtime import load_anonymous_model
    from tao_data import TaoImageDataset, annotated_image_records, full_video_frame_records

    sys.path.insert(0, str(args.pandas_root))
    from detection_utils import utils_od
    from detection_utils.presets import get_transforms

    model, prototype_state = load_anonymous_model(
        args.pandas_root,
        args.base_checkpoint,
        args.prototype_checkpoint,
        args.device,
    )
    eval_transform = get_transforms(False, None)
    if args.mode == "annotated":
        records = annotated_image_records(args.tao_annotation, args.tao_frame_root)
        if args.limit_records is not None:
            records = records[:args.limit_records]
    else:
        per_video = full_video_frame_records(args.tao_annotation, args.tao_frame_root)
        video_ids = sorted(per_video)
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        # Greedy load balancing by full-frame count, deterministic across runs.
        bins = [[] for _ in range(args.num_shards)]
        loads = [0] * args.num_shards
        for video_id in sorted(video_ids, key=lambda value: (-len(per_video[value]), value)):
            shard = min(range(args.num_shards), key=lambda index: (loads[index], index))
            bins[shard].append(video_id)
            loads[shard] += len(per_video[video_id])
        selected_video_ids = sorted(bins[args.shard_index])
        if args.limit_videos is not None:
            selected_video_ids = selected_video_ids[:args.limit_videos]
        mode_root = args.output_root / "anonymous" / args.mode
        if args.skip_existing:
            existing_video_ids = {
                int(path.stem.split("_")[-1])
                for path in mode_root.glob("video_*.npz")
            }
            existing_video_ids &= set(selected_video_ids)
            selected_video_ids = [
                video_id for video_id in selected_video_ids
                if video_id not in existing_video_ids
            ]
        records = [record for video_id in selected_video_ids for record in per_video[video_id]]

    dataset = TaoImageDataset(records, transform=eval_transform, include_ground_truth=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=utils_od.collate_fn,
    )
    mode_root = args.output_root / "anonymous" / args.mode
    mode_root.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    start = time.time()
    current_video = None
    current_rows: list[dict] = []
    video_count = 0
    detections_before_floor = 0
    detections_after_floor = 0
    zero_detection_frames_after_floor = 0
    image_count = 0
    semantic_sample: list[float] = []
    foreground_sample: list[float] = []
    batch_start = 0

    def extend_sample(target: list[float], values: np.ndarray) -> None:
        if len(target) >= args.score_sample_limit:
            return
        remaining = args.score_sample_limit - len(target)
        target.extend(np.asarray(values, dtype=np.float32).reshape(-1)[:remaining].tolist())

    def flush_video(video_id: int, rows: list[dict]) -> None:
        nonlocal video_count, image_count
        nonlocal detections_before_floor, detections_after_floor
        nonlocal zero_detection_frames_after_floor
        stats = write_video_npz(
            mode_root / f"video_{video_id:04d}.npz",
            rows,
            foreground_floor=foreground_floor,
        )
        video_count += 1
        image_count += stats["frames"]
        detections_before_floor += stats["detections_before_floor"]
        detections_after_floor += stats["detections_after_floor"]
        zero_detection_frames_after_floor += stats["zero_detection_frames_after_floor"]

    with torch.inference_mode():
        for batch_index, (images, _targets) in enumerate(loader):
            outputs = model.od_model([image.to(args.device) for image in images])
            for output_index, output in enumerate(outputs):
                record = records[batch_start + output_index]
                video_id = int(record["video_id"])
                if current_video is None:
                    current_video = video_id
                if video_id != current_video:
                    flush_video(current_video, current_rows)
                    current_rows = []
                    current_video = video_id
                if "foreground_scores" not in output:
                    raise KeyError("PANDAS output is missing foreground_scores")
                semantic_scores = output["scores"].detach().cpu().numpy()
                foreground_scores = output["foreground_scores"].detach().cpu().numpy()
                if len(output["boxes"]) != len(semantic_scores) or len(semantic_scores) != len(foreground_scores):
                    raise RuntimeError("PANDAS output fields have different detection counts")
                extend_sample(semantic_sample, semantic_scores)
                extend_sample(foreground_sample, foreground_scores)
                current_rows.append({
                    "frame_index": int(record["frame_index"]),
                    "image_id": int(record.get("image_id", -1)),
                    "boxes": output["boxes"].detach().cpu().numpy(),
                    "scores": semantic_scores,
                    "foreground_scores": foreground_scores,
                    "prototype_id": (output["labels"].detach().cpu().numpy() - 1),
                })
            batch_start += len(outputs)
            if batch_index % 100 == 0:
                print(f"{batch_start}/{len(dataset)}", flush=True)
    if current_video is not None:
        flush_video(current_video, current_rows)

    device = torch.device(args.device)
    result = {
        "status": "PASS",
        "mode": args.mode,
        "videos": video_count,
        "images_or_frames": image_count,
        "detections_before_floor": detections_before_floor,
        "detections_after_floor": detections_after_floor,
        "zero_detection_frames_after_floor": zero_detection_frames_after_floor,
        "foreground_score_floor": foreground_floor,
        "semantic_score_quantiles_sample": _quantile_summary(semantic_sample),
        "foreground_score_quantiles_sample": _quantile_summary(foreground_sample),
        "score_quantile_sample_size": len(semantic_sample),
        "prototype_count": int(prototype_state["class_prototypes"].shape[0]),
        "output_root": str(mode_root),
        "elapsed_seconds": time.time() - start,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "peak_cuda_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            if device.type == "cuda" else None
        ),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    suffix = "" if args.num_shards == 1 else f"_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    result["shard_index"] = args.shard_index
    result["num_shards"] = args.num_shards
    result["limit_records"] = args.limit_records
    result["limit_videos"] = args.limit_videos
    result["skip_existing"] = args.skip_existing
    result["score_contract"] = {
        "semantic_field": "scores",
        "tracking_field": "foreground_scores",
        "prototype_field": "prototype_id",
    }
    result["skipped_existing_videos"] = len(existing_video_ids)
    audit_path = args.output_root / "audit" / f"anonymous_inference_{args.mode}{suffix}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
