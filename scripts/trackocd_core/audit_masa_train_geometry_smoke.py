#!/usr/bin/env python3
"""Run a bounded MASA Train geometry smoke with a label-free predictor.

The predictor receives only four already-registered Train image paths and
image IDs.  Its sealed prediction payload is evaluated by the parent process
against the existing category-free physical geometry index.  This is an
engineering diagnostic for frontend transfer; it is not a qualification,
model-selection result, or OCD/MOT score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file  # noqa: E402

FRAMES = Path("/data3/liuyeqiang/TAO-Amodal/frames")
MASA_ENV = Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")
SELECTION = ROOT / "outputs/trackocd_core/pilots/gt_train_known/selection_plan.json"
GEOMETRY_GT = ROOT / "outputs/trackocd_cvpr/cvpr2/joint_known_protocol_v10_physical_GT/private_exact_joint_proof_or_assignment.json"
OUTPUT = ROOT / "outputs/trackocd_core/audit/masa_train_geometry_smoke_20261026.json"
RUN_ROOT = ROOT / "outputs/trackocd_core/audit/masa-train-geometry-smoke-20261026"
COVTRACK_STREAM = ROOT / (
    "outputs/trackocd_v2_runtime/manifests/frontend_streams/covtrack_native/"
    "covtrack_train_dense_shards_v1/shard_2_part_01_tccoff13_20261022/physical_tracks.jsonl"
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _validate_input(input_data: dict[str, Any]) -> list[Path]:
    paths = input_data.get("image_paths")
    if not isinstance(paths, list) or len(paths) != 4:
        raise ValueError("exactly four registered Train images are required")
    absolute: list[Path] = []
    for raw in paths:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "train":
            raise ValueError("worker input is not Train-only")
        path = FRAMES / relative
        if not path.resolve().is_relative_to((FRAMES / "train").resolve()) or not path.is_file():
            raise ValueError("worker input escapes the local Train tree or is missing")
        absolute.append(path)
    if len(set(absolute)) != 4 or len({path.parent for path in absolute}) != 1:
        raise ValueError("four distinct images from one video are required")
    return absolute


def _deny_network_and_annotations() -> None:
    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("MASA geometry worker forbids network and child processes: " + event)
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            forbidden = (
                "/TAO-Amodal/annotations/",
                "/recovered_splits/",
                "/frames/test/",
                "evaluator_only_geometry_join.json",
                "/roles.json",
            )
            if any(token in path for token in forbidden):
                raise PermissionError("MASA geometry worker cannot open GT, roles, or Test input")

    sys.addaudithook(audit)


def _snapshot(instances: Any, tracks: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    boxes = instances.bboxes.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1, 4)
    scores = instances.scores.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1)
    labels = instances.labels.detach().cpu().numpy().astype(np.int64, copy=True).reshape(-1)
    ids = None
    if tracks:
        ids = instances.instances_id.detach().cpu().numpy().astype(np.int64, copy=True).reshape(-1)
        if len(ids) != len(boxes) or len(np.unique(ids)) != len(ids):
            raise ValueError("invalid native track IDs")
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("native output field lengths differ")
    if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2])
            or np.any((scores < 0) | (scores > 1))
            or np.any(labels != 0)):
        raise ValueError("non-finite, degenerate, or non-anonymous MASA output")
    return boxes, scores, labels, ids


def worker(run: Path) -> int:
    started = time.monotonic()
    status = {
        "schema_version": "trackocd.core.masa_train_geometry_prediction.v1",
        "status": "RUNNING",
        "started_utc": _now(),
        "real_image_forwards_started": 0,
        "real_image_forwards_completed": 0,
        "boundary": {
            "gt_or_category_model_input": False,
            "training": False,
            "future_model_input": False,
            "val_or_test_access": False,
            "offline_filtering": False,
            "extra_weight_download": False,
            "foreign_process_interference": False,
        },
    }
    atomic_json(run / "worker_status.json", status)
    try:
        input_data = json.loads((run / "worker_input.json").read_text())
        images = _validate_input(input_data)
        _deny_network_and_annotations()
        import torch
        from PIL import Image
        from src.trackocd_core.masa_native import build_frozen_model, infer_current_frame

        torch.set_num_threads(1)
        torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model, proof = build_frozen_model()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("frozen MASA state contains a non-finite tensor")
        status["frozen_model"] = proof
        status["runtime_versions"] = {"torch": torch.__version__}
        model.to("cuda:0")
        model.eval()
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, (8 * 1024**3) / torch.cuda.get_device_properties(0).total_memory), device=0
        )
        torch.cuda.reset_peak_memory_stats()

        image_ids = [int(value) for value in input_data["image_ids"]]
        if len(image_ids) != len(images):
            raise ValueError("image ID and image path counts differ")
        arrays: dict[str, list[Any]] = {
            "image_id": [], "det_offsets": [0], "det_boxes": [], "det_scores": [],
            "track_offsets": [0], "track_ids": [], "track_boxes": [], "track_scores": [],
        }
        rows = []
        model.tracker.reset()
        for ordinal, (image_id, path) in enumerate(zip(image_ids, images)):
            if time.monotonic() - started > 600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3:
                raise RuntimeError("registered MASA geometry smoke resource ceiling")
            with Image.open(path) as raw:
                pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
            status["real_image_forwards_started"] += 1
            atomic_json(run / "worker_status.json", status)
            detections, tracks = infer_current_frame(model, pixels, ordinal, "cuda:0")
            torch.cuda.synchronize()
            det_boxes, det_scores, _, _ = _snapshot(detections, False)
            track_boxes, track_scores, _, track_ids = _snapshot(tracks, True)
            assert track_ids is not None
            arrays["image_id"].append(image_id)
            arrays["det_boxes"].extend(det_boxes.tolist())
            arrays["det_scores"].extend(det_scores.tolist())
            arrays["det_offsets"].append(len(arrays["det_scores"]))
            arrays["track_boxes"].extend(track_boxes.tolist())
            arrays["track_scores"].extend(track_scores.tolist())
            arrays["track_ids"].extend(track_ids.tolist())
            arrays["track_offsets"].append(len(arrays["track_ids"]))
            rows.append({
                "ordinal": ordinal,
                "image_id": image_id,
                "image_sha256": sha256_file(path),
                "detections": int(len(det_scores)),
                "tracks": int(len(track_ids)),
            })
            status["real_image_forwards_completed"] += 1
            status["frames"] = rows
            atomic_json(run / "worker_status.json", status)
            if torch.cuda.max_memory_reserved() > 8 * 1024**3:
                raise RuntimeError("registered MASA GPU memory ceiling")

        prediction_path = run / "sealed_prediction.npz"
        np.savez_compressed(
            prediction_path,
            image_id=np.asarray(arrays["image_id"], dtype=np.int64),
            det_offsets=np.asarray(arrays["det_offsets"], dtype=np.int64),
            det_boxes=np.asarray(arrays["det_boxes"], dtype=np.float32).reshape(-1, 4),
            det_scores=np.asarray(arrays["det_scores"], dtype=np.float32),
            track_offsets=np.asarray(arrays["track_offsets"], dtype=np.int64),
            track_ids=np.asarray(arrays["track_ids"], dtype=np.int64),
            track_boxes=np.asarray(arrays["track_boxes"], dtype=np.float32).reshape(-1, 4),
            track_scores=np.asarray(arrays["track_scores"], dtype=np.float32),
        )
        if prediction_path.stat().st_size > 10 * 1024**2:
            raise RuntimeError("registered prediction output ceiling")
        status.update(
            status="SEALED_CATEGORY_FREE_TRAIN_PREDICTIONS",
            frames=rows,
            prediction_filename=prediction_path.name,
            prediction_sha256=sha256_file(prediction_path),
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    except Exception as exc:  # keep an auditable failure receipt
        status.update(status="BLOCKED_CATEGORY_FREE_TRAIN_PREDICTION", error=type(exc).__name__ + ": " + str(exc))
        import traceback
        traceback.print_exc()
    status.update(
        elapsed_seconds=time.monotonic() - started,
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        completed_utc=_now(),
    )
    atomic_json(run / "worker_status.json", status)
    return 0 if status["status"].startswith("SEALED") else 1


def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float32)
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_left = np.maximum(0, left[:, 2] - left[:, 0]) * np.maximum(0, left[:, 3] - left[:, 1])
    area_right = np.maximum(0, right[:, 2] - right[:, 0]) * np.maximum(0, right[:, 3] - right[:, 1])
    return inter / np.maximum(area_left[:, None] + area_right[None, :] - inter, 1e-12)


def _max_cardinality_pairs(gt_boxes: np.ndarray, pred_boxes: np.ndarray, threshold: float = 0.5) -> list[tuple[int, int]]:
    """Return thresholded maximum-cardinality pairs without persisting them."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    ious = _iou_matrix(gt_boxes, pred_boxes)
    costs = np.where(ious >= threshold, 1.0 - ious, 1000.0)
    rows, cols = linear_sum_assignment(costs)
    return [(int(row), int(col)) for row, col in zip(rows, cols) if ious[row, col] >= threshold]


def _load_geometry(geometry_path: Path, image_ids: set[int]) -> dict[int, list[tuple[str, list[float]]]]:
    data = json.loads(geometry_path.read_text())
    if data.get("semantic_category_labels_retained") is not False or data.get("not_model_facing") is not True:
        raise ValueError("geometry index is not the sealed category-free evaluator input")
    result: dict[int, list[tuple[str, list[float]]]] = {}
    for track in data["tracks"]:
        track_id = str(track["opaque_track_id"])
        for observation in track["observations"]:
            image_id = int(observation[0])
            if image_id not in image_ids:
                continue
            x, y, width, height = map(float, observation[2:6])
            box = [x, y, x + width, y + height]
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("invalid category-free GT geometry")
            result.setdefault(image_id, []).append((track_id, box))
    if set(result) != image_ids:
        raise ValueError("geometry index does not cover the four registered images")
    return result


def evaluate(run: Path, geometry_path: Path, input_data: dict[str, Any]) -> dict[str, Any]:
    worker_status = json.loads((run / "worker_status.json").read_text())
    if worker_status.get("status") != "SEALED_CATEGORY_FREE_TRAIN_PREDICTIONS":
        raise ValueError("complete sealed prediction is required before geometry access")
    prediction_path = run / worker_status["prediction_filename"]
    if sha256_file(prediction_path) != worker_status["prediction_sha256"]:
        raise ValueError("sealed prediction changed before evaluation")
    image_ids = [int(value) for value in input_data["image_ids"]]
    gt = _load_geometry(geometry_path, set(image_ids))
    matched_by_image: dict[int, dict[str, int]] = {}
    counts = {
        "gt_observations": 0,
        "predicted_track_observations": 0,
        "matched_track_observations": 0,
        "predicted_detection_observations": 0,
        "matched_detection_observations": 0,
    }
    with np.load(prediction_path, allow_pickle=False) as arrays:
        pred_image_ids = [int(value) for value in arrays["image_id"]]
        det_offsets = arrays["det_offsets"].astype(np.int64)
        det_boxes = arrays["det_boxes"].astype(np.float32)
        track_offsets = arrays["track_offsets"].astype(np.int64)
        track_boxes = arrays["track_boxes"].astype(np.float32)
        track_ids = arrays["track_ids"].astype(np.int64)
        if pred_image_ids != image_ids:
            raise ValueError("prediction image order differs from registered input")
        for index, image_id in enumerate(image_ids):
            gt_rows = gt[image_id]
            gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=np.float32).reshape(-1, 4)
            db, de = int(det_offsets[index]), int(det_offsets[index + 1])
            tb, te = int(track_offsets[index]), int(track_offsets[index + 1])
            detections = det_boxes[db:de]
            tracks = track_boxes[tb:te]
            ids = track_ids[tb:te]
            detection_pairs = _max_cardinality_pairs(gt_boxes, detections)
            track_pairs = _max_cardinality_pairs(gt_boxes, tracks)
            counts["gt_observations"] += len(gt_boxes)
            counts["predicted_detection_observations"] += len(detections)
            counts["matched_detection_observations"] += len(detection_pairs)
            counts["predicted_track_observations"] += len(tracks)
            counts["matched_track_observations"] += len(track_pairs)
            # Keep only this small in-memory posthoc join long enough to count
            # continuity; no prediction-to-GT mapping is written to disk.
            matched_by_image[image_id] = {gt_rows[g][0]: int(ids[p]) for g, p in track_pairs}

    gt_track_observation_counts: dict[str, int] = {}
    matched_track_counts: dict[str, int] = {}
    for image_id in image_ids:
        for track_id, _ in gt[image_id]:
            gt_track_observation_counts[track_id] = gt_track_observation_counts.get(track_id, 0) + 1
        for track_id in matched_by_image[image_id]:
            matched_track_counts[track_id] = matched_track_counts.get(track_id, 0) + 1
    adjacent_pairs = both_pairs = same_id_pairs = switched_id_pairs = 0
    for previous, current in zip(image_ids, image_ids[1:]):
        previous_ids = matched_by_image[previous]
        current_ids = matched_by_image[current]
        for gt_id in set(previous_ids) & set(current_ids):
            both_pairs += 1
            if previous_ids[gt_id] == current_ids[gt_id]:
                same_id_pairs += 1
            else:
                switched_id_pairs += 1
        adjacent_pairs += len(set(track_id for track_id, _ in gt[previous]) & set(track_id for track_id, _ in gt[current]))
    return {
        "status": "COMPLETE_CATEGORY_FREE_TRAIN_MASA_GEOMETRY_SMOKE_NOT_QUALIFICATION",
        "scope": {
            "video_count": 1,
            "image_count": len(image_ids),
            "image_sampling": "four chronological registered Train images, 30-frame source spacing, ordinal native replay",
            "iou_threshold": 0.5,
            "official_ocd_or_trackeval_metrics": False,
            "frontend_qualification": False,
            "model_selection": False,
        },
        "counts": counts,
        "geometry": {
            "track_observation_precision": counts["matched_track_observations"] / max(counts["predicted_track_observations"], 1),
            "track_observation_recall": counts["matched_track_observations"] / max(counts["gt_observations"], 1),
            "detection_observation_precision": counts["matched_detection_observations"] / max(counts["predicted_detection_observations"], 1),
            "detection_observation_recall": counts["matched_detection_observations"] / max(counts["gt_observations"], 1),
            "gt_tracks_with_any_track_match": sum(value > 0 for value in matched_track_counts.values()),
            "gt_tracks": len(gt_track_observation_counts),
        },
        "temporal": {
            "annotated_adjacent_gt_pairs": adjacent_pairs,
            "pairs_with_both_track_matches": both_pairs,
            "same_id_pairs": same_id_pairs,
            "switched_id_pairs": switched_id_pairs,
            "conditional_same_id_rate": same_id_pairs / max(both_pairs, 1),
            "conditional_switch_rate": switched_id_pairs / max(both_pairs, 1),
        },
        "inputs": {
            "prediction_run": str(run.relative_to(ROOT)),
            "prediction_sha256": worker_status["prediction_sha256"],
            "geometry_only_gt_index": str(geometry_path.resolve()),
            "geometry_only_gt_index_sha256": sha256_file(geometry_path),
            "image_ids": image_ids,
        },
        "boundary": {
            "predictor_received_only_current_images_and_image_ids": True,
            "gt_category_fields_read": False,
            "gt_role_or_vocabulary_read": False,
            "gt_opened_only_after_sealed_prediction": True,
            "val_labels_read": False,
            "test_accessed": False,
            "training_started": False,
            "prediction_to_gt_track_mapping_persisted": False,
        },
        "interpretation": (
            "Bounded engineering smoke on a selection inherited from a long Known-GT pilot; "
            "the selection is intentionally not a coverage estimate and cannot qualify MASA "
            "or establish an OCD/MOT result."
        ),
    }


def _stream_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_covtrack_same_clip(stream_path: Path, geometry_path: Path, input_data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the existing anonymous COVTrack stream on the same four images."""
    physical_fields = {
        "sample_key", "source_sample_id", "video_id", "physical_track_id", "frame_ids",
        "boxes_xyxy", "image_paths", "quality", "stream_order", "source_split",
    }
    image_ids = [int(value) for value in input_data["image_ids"]]
    paths = [str(value) for value in input_data["image_paths"]]
    gt = _load_geometry(geometry_path, set(image_ids))
    by_path: dict[str, list[tuple[str, list[float]]]] = {path: [] for path in paths}
    rows_scanned = 0
    target_rows = 0
    with stream_path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rows_scanned += 1
            row = json.loads(raw)
            if set(row) != physical_fields:
                raise ValueError("COVTrack comparator is not a physical-only stream")
            if row.get("source_split") != "train_predicted":
                raise ValueError("COVTrack comparator contains a non-Train stream")
            hits = 0
            for path, box in zip(row["image_paths"], row["boxes_xyxy"]):
                if path in by_path:
                    if not isinstance(box, list) or len(box) != 4:
                        raise ValueError("invalid COVTrack physical box")
                    by_path[path].append((str(row["physical_track_id"]), [float(value) for value in box]))
                    hits += 1
            target_rows += bool(hits)
    if any(not by_path[path] for path in paths):
        raise ValueError("COVTrack comparator does not cover all registered images")
    matched_by_image: dict[int, dict[str, str]] = {}
    gt_observations = sum(len(gt[image_id]) for image_id in image_ids)
    predicted_observations = 0
    matched_observations = 0
    for image_id, path in zip(image_ids, paths):
        gt_rows = gt[image_id]
        predictions = by_path[path]
        pairs = _max_cardinality_pairs(
            np.asarray([row[1] for row in gt_rows], dtype=np.float32).reshape(-1, 4),
            np.asarray([row[1] for row in predictions], dtype=np.float32).reshape(-1, 4),
        )
        matched_by_image[image_id] = {gt_rows[g][0]: predictions[p][0] for g, p in pairs}
        predicted_observations += len(predictions)
        matched_observations += len(pairs)
    adjacent_pairs = both_pairs = same_id_pairs = switched_id_pairs = 0
    for previous, current in zip(image_ids, image_ids[1:]):
        ids = set(row[0] for row in gt[previous]) & set(row[0] for row in gt[current])
        adjacent_pairs += len(ids)
        for gt_id in ids:
            if gt_id not in matched_by_image[previous] or gt_id not in matched_by_image[current]:
                continue
            both_pairs += 1
            if matched_by_image[previous][gt_id] == matched_by_image[current][gt_id]:
                same_id_pairs += 1
            else:
                switched_id_pairs += 1
    return {
        "stream": str(stream_path.resolve()),
        "stream_bytes": stream_path.stat().st_size,
        "stream_sha256": _stream_digest(stream_path),
        "stream_rows_scanned": rows_scanned,
        "rows_with_registered_images": target_rows,
        "gt_observations": gt_observations,
        "predicted_track_observations": predicted_observations,
        "matched_track_observations": matched_observations,
        "track_observation_precision": matched_observations / max(predicted_observations, 1),
        "track_observation_recall": matched_observations / max(gt_observations, 1),
        "annotated_adjacent_gt_pairs": adjacent_pairs,
        "pairs_with_both_track_matches": both_pairs,
        "same_id_pairs": same_id_pairs,
        "switched_id_pairs": switched_id_pairs,
        "conditional_same_id_rate": same_id_pairs / max(both_pairs, 1),
        "conditional_switch_rate": switched_id_pairs / max(both_pairs, 1),
    }


def augment_existing() -> int:
    result = json.loads(OUTPUT.read_text())
    input_data = json.loads((RUN_ROOT / "worker_input.json").read_text())
    comparison = evaluate_covtrack_same_clip(COVTRACK_STREAM, GEOMETRY_GT, input_data)
    masa = result["evaluation"]["geometry"]
    result["paired_same_clip_comparison"] = {
        "covtrack_native": comparison,
        "masa": {
            "track_observation_precision": masa["track_observation_precision"],
            "track_observation_recall": masa["track_observation_recall"],
            "conditional_same_id_rate": result["evaluation"]["temporal"]["conditional_same_id_rate"],
            "conditional_switch_rate": result["evaluation"]["temporal"]["conditional_switch_rate"],
        },
        "conclusion": (
            "MASA is not superior to the existing COVTrack-native stream on this same-video, "
            "same-image engineering smoke; this does not qualify either frontend or generalize "
            "beyond the preregistered biased clip."
        ),
        "boundary": {
            "same_category_free_geometry_join": True,
            "model_selection": False,
            "official_ocd_or_trackeval_metrics": False,
            "test_accessed": False,
            "mapping_persisted": False,
        },
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "paired_same_clip_comparison": result["paired_same_clip_comparison"]}, sort_keys=True))
    return 0


def _make_input() -> dict[str, Any]:
    selection = json.loads(SELECTION.read_text())
    row = selection["rows"][0]
    observations = row["observations"][:4]
    paths = [str(item["image_path"]) for item in observations]
    image_ids = [int(item["image_id"]) for item in observations]
    if len(set(image_ids)) != 4:
        raise ValueError("registered selection contains duplicate image IDs")
    return {
        "image_paths": paths,
        "image_ids": image_ids,
        "selection_plan_sha256": sha256_file(SELECTION),
        "selection_bias": "Existing long Known-GT pilot selection; engineering only, not coverage estimation",
    }


def _select_gpu() -> tuple[str, str]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    occupied = set(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    ).split())
    for line in query.splitlines():
        uuid, used, free, utilization = [value.strip() for value in line.split(",")]
        if uuid not in occupied and int(used) < 100 and int(free) >= 9216 and int(utilization) == 0:
            return uuid, query
    raise RuntimeError("no freshly idle GPU with registered headroom")


def parent() -> int:
    if RUN_ROOT.exists() or OUTPUT.exists():
        raise ValueError("preserve an existing MASA geometry smoke; no automatic rerun")
    input_data = _make_input()
    images = _validate_input(input_data)
    RUN_ROOT.mkdir(parents=True)
    atomic_json(RUN_ROOT / "worker_input.json", {"image_paths": input_data["image_paths"], "image_ids": input_data["image_ids"]})
    gpu_uuid, gpu_query = _select_gpu()
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=gpu_uuid,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLCONFIGDIR=str(RUN_ROOT / "matplotlib"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    process = subprocess.Popen(
        [str(MASA_ENV / "bin/python"), str(Path(__file__).resolve()), "--worker", str(RUN_ROOT)],
        cwd=ROOT,
        env=env,
        stdout=(RUN_ROOT / "worker.log").open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    error = None
    started = time.monotonic()
    try:
        while process.poll() is None:
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(memory["MemAvailable"].split()[0]) * 1024
            total = int(memory["MemTotal"].split()[0]) * 1024
            if time.monotonic() - started > 600 or available < total * 0.25 or RUN_ROOT.stat().st_size > 64 * 1024**2:
                raise RuntimeError("registered MASA geometry smoke resource guard")
            time.sleep(1)
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
    status_path = RUN_ROOT / "worker_status.json"
    worker_status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "NO_WORKER_RECEIPT"}
    result: dict[str, Any] = {
        "schema_version": "trackocd.core.masa_train_geometry_smoke.v1",
        "generated_utc": _now(),
        "worker_returncode": process.returncode,
        "worker_error": error,
        "gpu_uuid": gpu_uuid,
        "fresh_gpu_query": gpu_query,
        "selection_plan_sha256": input_data["selection_plan_sha256"],
        "input_image_sha256": [sha256_file(path) for path in images],
        "worker": worker_status,
    }
    if error is None and worker_status.get("status") == "SEALED_CATEGORY_FREE_TRAIN_PREDICTIONS":
        result["evaluation"] = evaluate(RUN_ROOT, GEOMETRY_GT, input_data)
        result["status"] = result["evaluation"]["status"]
    else:
        result["status"] = "BLOCKED_CATEGORY_FREE_TRAIN_MASA_GEOMETRY_SMOKE"
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "worker_returncode": process.returncode,
                      "evaluation": result.get("evaluation", {})}, sort_keys=True))
    return 0 if result["status"].startswith("COMPLETE") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--augment-existing", action="store_true")
    args = parser.parse_args()
    if args.augment_existing:
        raise SystemExit(augment_existing())
    if args.worker is not None:
        raise SystemExit(worker(args.worker))
    raise SystemExit(parent())
