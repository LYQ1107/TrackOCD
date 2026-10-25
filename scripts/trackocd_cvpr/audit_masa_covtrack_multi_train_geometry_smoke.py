#!/usr/bin/env python3
"""Compare frozen MASA and existing COVTrack on fixed anonymous Train clips.

The parent selects eight videos and four image-metadata timestamps per video
using a frozen structural rule.  The worker sees only current Train pixels and
image IDs.  Geometry evaluation happens only after the prediction payload is
sealed and uses the already sealed category-free physical GT index.  No
category/role/vocabulary field, Val/Test asset, or prediction-to-GT mapping is
written to the result.
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json  # noqa: E402
from scripts.trackocd_core.audit_masa_train_geometry_smoke import (  # noqa: E402
    FRAMES,
    MASA_ENV,
    _deny_network_and_annotations,
    _load_geometry,
    _max_cardinality_pairs,
    _snapshot,
)

CONFIG = ROOT / "configs/trackocd_cvpr/masa_covtrack_multi_train_geometry_smoke_v1.json"
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_covtrack_multi_train_geometry_smoke_v1/result.json"
RUN_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_covtrack_multi_train_geometry_smoke_v1/run"
PHYSICAL_FIELDS = {
    "sample_key", "source_sample_id", "video_id", "physical_track_id", "frame_ids",
    "boxes_xyxy", "image_paths", "quality", "stream_order", "source_split",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != "trackocd.cvpr.masa-covtrack-multi-train-geometry-smoke.v1":
        raise ValueError("unexpected MASA/COVTrack smoke config schema")
    if cfg.get("status") != "FROZEN_MULTI_VIDEO_MASA_COVTRACK_GEOMETRY_SMOKE_NOT_QUALIFICATION":
        raise ValueError("MASA/COVTrack smoke config is not frozen")
    if cfg["fixed_selection"]["image_plan_sha256"] != _sha256(ROOT / cfg["fixed_selection"]["image_plan"]):
        raise ValueError("image plan hash mismatch")
    if cfg["inputs"]["geometry_only_gt_index_sha256"] != _sha256(ROOT / cfg["inputs"]["geometry_only_gt_index"]):
        raise ValueError("geometry index hash mismatch")
    if cfg["boundaries"]["test_accessed"] is not False or cfg["boundaries"]["training_started"] is not False:
        raise ValueError("MASA/COVTrack smoke boundary is open")
    return cfg


def _build_fixed_input(cfg: dict[str, Any]) -> dict[str, Any]:
    image_plan_path = ROOT / cfg["fixed_selection"]["image_plan"]
    payload = json.loads(image_plan_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "trackocd.cvpr.private-selected-images.v1":
        raise ValueError("unexpected category-free image plan schema")
    if payload.get("not_model_facing") is not True or payload.get("not_a_frozen_protocol") is not True:
        raise ValueError("image plan is not evaluator-only")
    if payload.get("semantic_labels_or_GT_boxes_in_image_plan") is not False:
        raise ValueError("image plan retains semantic labels or boxes")
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    allowed = set(int(value) for value in cfg["fixed_selection"]["video_ids"])
    for row in payload["image_metadata"]:
        if set(row) != {"file_name", "frame_index", "height", "id", "video_id", "width"}:
            raise ValueError("image plan contains fields outside anonymous image metadata")
        video_id = int(row["video_id"])
        if video_id in allowed:
            rel = str(row["file_name"])
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or not rel.startswith("train/"):
                raise ValueError("fixed input is not Train-only")
            if not (FRAMES / rel).resolve().is_relative_to((FRAMES / "train").resolve()):
                raise ValueError("fixed input escapes Train frame root")
            by_video[video_id].append({
                "video_id": video_id,
                "image_id": int(row["id"]),
                "frame_index": int(row["frame_index"]),
                "image_path": rel,
            })
    selected: list[dict[str, Any]] = []
    for video_id in cfg["fixed_selection"]["video_ids"]:
        rows = sorted(by_video[int(video_id)], key=lambda row: (row["frame_index"], row["image_id"]))
        if len(rows) < int(cfg["fixed_selection"]["images_per_video"]):
            raise ValueError(f"fixed video {video_id} has too few image metadata rows")
        chosen = rows[: int(cfg["fixed_selection"]["images_per_video"])]
        if len({row["image_id"] for row in chosen}) != len(chosen):
            raise ValueError("duplicate selected image ID")
        selected.extend(chosen)
    if len(selected) != len(cfg["fixed_selection"]["video_ids"]) * int(cfg["fixed_selection"]["images_per_video"]):
        raise ValueError("fixed image selection count mismatch")
    for row in selected:
        if not (FRAMES / row["image_path"]).is_file():
            raise FileNotFoundError(FRAMES / row["image_path"])
    return {
        "videos": [int(value) for value in cfg["fixed_selection"]["video_ids"]],
        "images_per_video": int(cfg["fixed_selection"]["images_per_video"]),
        "images": selected,
        "image_plan_sha256": _sha256(image_plan_path),
    }


def _deny_child_and_network() -> None:
    _deny_network_and_annotations()


def _worker(run: Path, cfg: dict[str, Any]) -> int:
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": "trackocd.cvpr.masa-covtrack-multi-worker.v1",
        "status": "RUNNING",
        "started_utc": _now(),
        "completed_images": 0,
        "boundary": cfg["boundaries"],
    }
    atomic_json(run / "worker_status.json", status)
    try:
        input_data = json.loads((run / "worker_input.json").read_text(encoding="utf-8"))
        images = input_data["images"]
        _deny_child_and_network()
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
        model.to("cuda:0").eval()
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, float(cfg["resources"]["gpu_peak_bytes_ceiling"]) / torch.cuda.get_device_properties(0).total_memory),
            device=0,
        )
        torch.cuda.reset_peak_memory_stats()
        arrays: dict[str, list[Any]] = {
            "image_id": [], "video_id": [], "det_offsets": [0], "det_boxes": [], "det_scores": [],
            "track_offsets": [0], "track_ids": [], "track_boxes": [], "track_scores": [],
        }
        rows: list[dict[str, Any]] = []
        current_video = None
        for ordinal, row in enumerate(images):
            if current_video != int(row["video_id"]):
                model.tracker.reset()
                current_video = int(row["video_id"])
                video_ordinal = 0
            else:
                video_ordinal += 1
            if time.monotonic() - started > float(cfg["resources"]["wall_seconds_ceiling"]):
                raise RuntimeError("MASA multi-clip wall ceiling")
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > int(cfg["resources"]["host_rss_bytes_ceiling"]):
                raise MemoryError("MASA multi-clip host RSS ceiling")
            path = (FRAMES / str(row["image_path"])).resolve()
            with Image.open(path) as raw:
                pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
            detections, tracks = infer_current_frame(model, pixels, video_ordinal, "cuda:0")
            torch.cuda.synchronize()
            det_boxes, det_scores, _, _ = _snapshot(detections, False)
            track_boxes, track_scores, _, track_ids = _snapshot(tracks, True)
            assert track_ids is not None
            arrays["image_id"].append(int(row["image_id"]))
            arrays["video_id"].append(int(row["video_id"]))
            arrays["det_boxes"].extend(det_boxes.tolist())
            arrays["det_scores"].extend(det_scores.tolist())
            arrays["det_offsets"].append(len(arrays["det_scores"]))
            arrays["track_boxes"].extend(track_boxes.tolist())
            arrays["track_scores"].extend(track_scores.tolist())
            arrays["track_ids"].extend(track_ids.tolist())
            arrays["track_offsets"].append(len(arrays["track_ids"]))
            rows.append({
                "video_id": int(row["video_id"]),
                "image_id": int(row["image_id"]),
                "image_sha256": _sha256(path),
                "detections": int(len(det_scores)),
                "tracks": int(len(track_ids)),
            })
            status["completed_images"] = ordinal + 1
            status["frames"] = rows
            atomic_json(run / "worker_status.json", status)
            if int(torch.cuda.max_memory_reserved()) > int(cfg["resources"]["gpu_peak_bytes_ceiling"]):
                raise MemoryError("MASA multi-clip GPU ceiling")
        prediction = run / "sealed_prediction.npz"
        np.savez_compressed(
            prediction,
            image_id=np.asarray(arrays["image_id"], dtype=np.int64),
            video_id=np.asarray(arrays["video_id"], dtype=np.int64),
            det_offsets=np.asarray(arrays["det_offsets"], dtype=np.int64),
            det_boxes=np.asarray(arrays["det_boxes"], dtype=np.float32).reshape(-1, 4),
            det_scores=np.asarray(arrays["det_scores"], dtype=np.float32),
            track_offsets=np.asarray(arrays["track_offsets"], dtype=np.int64),
            track_ids=np.asarray(arrays["track_ids"], dtype=np.int64),
            track_boxes=np.asarray(arrays["track_boxes"], dtype=np.float32).reshape(-1, 4),
            track_scores=np.asarray(arrays["track_scores"], dtype=np.float32),
        )
        if prediction.stat().st_size > int(cfg["resources"]["output_bytes_ceiling"]):
            raise RuntimeError("MASA multi-clip output ceiling")
        status.update(
            status="SEALED_CATEGORY_FREE_MULTI_TRAIN_PREDICTIONS",
            frames=rows,
            prediction_filename=prediction.name,
            prediction_sha256=_sha256(prediction),
            frozen_model=proof,
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    except Exception as exc:
        status.update(status="BLOCKED_CATEGORY_FREE_MULTI_TRAIN_PREDICTION", error=type(exc).__name__ + ": " + str(exc))
        import traceback
        traceback.print_exc()
    status.update(
        elapsed_seconds=time.monotonic() - started,
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        completed_utc=_now(),
    )
    atomic_json(run / "worker_status.json", status)
    return 0 if status["status"].startswith("SEALED") else 1


def _evaluate_masa(run: Path, input_data: dict[str, Any], geometry_path: Path) -> dict[str, Any]:
    worker = json.loads((run / "worker_status.json").read_text(encoding="utf-8"))
    if worker.get("status") != "SEALED_CATEGORY_FREE_MULTI_TRAIN_PREDICTIONS":
        raise ValueError("sealed MASA prediction required")
    prediction = run / worker["prediction_filename"]
    if _sha256(prediction) != worker["prediction_sha256"]:
        raise ValueError("sealed MASA prediction changed")
    image_ids = {int(row["image_id"]) for row in input_data["images"]}
    gt = _load_geometry(geometry_path, image_ids)
    by_video = defaultdict(list)
    for row in input_data["images"]:
        by_video[int(row["video_id"])].append(int(row["image_id"]))
    matched_by_image: dict[int, dict[str, int]] = {}
    counts = {"gt_observations": 0, "predicted_track_observations": 0, "matched_track_observations": 0,
              "predicted_detection_observations": 0, "matched_detection_observations": 0}
    with np.load(prediction, allow_pickle=False) as arrays:
        pred_ids = [int(value) for value in arrays["image_id"]]
        expected_ids = [int(row["image_id"]) for row in input_data["images"]]
        if pred_ids != expected_ids:
            raise ValueError("MASA prediction image order mismatch")
        for index, image_id in enumerate(expected_ids):
            gt_rows = gt[image_id]
            gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=np.float32).reshape(-1, 4)
            db, de = int(arrays["det_offsets"][index]), int(arrays["det_offsets"][index + 1])
            tb, te = int(arrays["track_offsets"][index]), int(arrays["track_offsets"][index + 1])
            detections = arrays["det_boxes"][db:de]
            tracks = arrays["track_boxes"][tb:te]
            ids = arrays["track_ids"][tb:te]
            detection_pairs = _max_cardinality_pairs(gt_boxes, detections)
            track_pairs = _max_cardinality_pairs(gt_boxes, tracks)
            counts["gt_observations"] += len(gt_boxes)
            counts["predicted_detection_observations"] += len(detections)
            counts["matched_detection_observations"] += len(detection_pairs)
            counts["predicted_track_observations"] += len(tracks)
            counts["matched_track_observations"] += len(track_pairs)
            matched_by_image[image_id] = {gt_rows[g][0]: int(ids[p]) for g, p in track_pairs}
    temporal = {"annotated_adjacent_gt_pairs": 0, "pairs_with_both_track_matches": 0,
                "same_id_pairs": 0, "switched_id_pairs": 0}
    per_video = {}
    for video_id, ids in by_video.items():
        local = {key: 0 for key in temporal}
        for previous, current in zip(ids, ids[1:]):
            previous_gt = {track_id for track_id, _box in gt[previous]}
            current_gt = {track_id for track_id, _box in gt[current]}
            shared = previous_gt & current_gt
            local["annotated_adjacent_gt_pairs"] += len(shared)
            temporal["annotated_adjacent_gt_pairs"] += len(shared)
            for track_id in shared:
                if track_id not in matched_by_image[previous] or track_id not in matched_by_image[current]:
                    continue
                local["pairs_with_both_track_matches"] += 1
                temporal["pairs_with_both_track_matches"] += 1
                if matched_by_image[previous][track_id] == matched_by_image[current][track_id]:
                    local["same_id_pairs"] += 1
                    temporal["same_id_pairs"] += 1
                else:
                    local["switched_id_pairs"] += 1
                    temporal["switched_id_pairs"] += 1
        local["conditional_switch_rate"] = local["switched_id_pairs"] / max(local["pairs_with_both_track_matches"], 1)
        per_video[str(video_id)] = local
    temporal["conditional_switch_rate"] = temporal["switched_id_pairs"] / max(temporal["pairs_with_both_track_matches"], 1)
    return {
        "counts": counts,
        "geometry": {
            "track_observation_precision": counts["matched_track_observations"] / max(counts["predicted_track_observations"], 1),
            "track_observation_recall": counts["matched_track_observations"] / max(counts["gt_observations"], 1),
            "detection_observation_precision": counts["matched_detection_observations"] / max(counts["predicted_detection_observations"], 1),
            "detection_observation_recall": counts["matched_detection_observations"] / max(counts["gt_observations"], 1),
        },
        "temporal": temporal,
        "per_video": per_video,
    }


def _stream_digest(path: Path) -> str:
    return _sha256(path)


def _evaluate_covtrack(input_data: dict[str, Any], stream_root: Path, geometry_path: Path) -> dict[str, Any]:
    target_paths = {str(row["image_path"]): int(row["image_id"]) for row in input_data["images"]}
    by_path: dict[str, list[tuple[str, list[float]]]] = {path: [] for path in target_paths}
    stream_records = []
    rows_scanned = 0
    for stream_path in sorted(stream_root.glob("*/physical_tracks.jsonl")):
        digest = hashlib.sha256()
        local_rows = 0
        with stream_path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                rows_scanned += 1
                local_rows += 1
                if set(row) != PHYSICAL_FIELDS or row.get("source_split") != "train_predicted":
                    raise ValueError("COVTrack stream is not anonymous physical Train data")
                for path, box in zip(row["image_paths"], row["boxes_xyxy"]):
                    if path in by_path:
                        by_path[path].append((str(row["physical_track_id"]), [float(value) for value in box]))
        stream_records.append({"path": str(stream_path.resolve()), "bytes": stream_path.stat().st_size,
                               "sha256": digest.hexdigest(), "rows": local_rows})
    if any(not values for values in by_path.values()):
        missing = [path for path, values in by_path.items() if not values]
        raise ValueError("COVTrack stream misses fixed image paths: " + repr(missing[:3]))
    gt = _load_geometry(geometry_path, set(target_paths.values()))
    matched_by_image: dict[int, dict[str, str]] = {}
    counts = {"gt_observations": 0, "predicted_track_observations": 0, "matched_track_observations": 0}
    for path, image_id in target_paths.items():
        gt_rows = gt[image_id]
        predictions = by_path[path]
        pairs = _max_cardinality_pairs(
            np.asarray([row[1] for row in gt_rows], dtype=np.float32).reshape(-1, 4),
            np.asarray([row[1] for row in predictions], dtype=np.float32).reshape(-1, 4),
        )
        matched_by_image[image_id] = {gt_rows[g][0]: predictions[p][0] for g, p in pairs}
        counts["gt_observations"] += len(gt_rows)
        counts["predicted_track_observations"] += len(predictions)
        counts["matched_track_observations"] += len(pairs)
    temporal = {"annotated_adjacent_gt_pairs": 0, "pairs_with_both_track_matches": 0,
                "same_id_pairs": 0, "switched_id_pairs": 0}
    per_video = {}
    for video_id in input_data["videos"]:
        ids = [int(row["image_id"]) for row in input_data["images"] if int(row["video_id"]) == int(video_id)]
        local = {key: 0 for key in temporal}
        for previous, current in zip(ids, ids[1:]):
            shared = {track_id for track_id, _box in gt[previous]} & {track_id for track_id, _box in gt[current]}
            local["annotated_adjacent_gt_pairs"] += len(shared)
            temporal["annotated_adjacent_gt_pairs"] += len(shared)
            for track_id in shared:
                if track_id not in matched_by_image[previous] or track_id not in matched_by_image[current]:
                    continue
                local["pairs_with_both_track_matches"] += 1
                temporal["pairs_with_both_track_matches"] += 1
                if matched_by_image[previous][track_id] == matched_by_image[current][track_id]:
                    local["same_id_pairs"] += 1
                    temporal["same_id_pairs"] += 1
                else:
                    local["switched_id_pairs"] += 1
                    temporal["switched_id_pairs"] += 1
        local["conditional_switch_rate"] = local["switched_id_pairs"] / max(local["pairs_with_both_track_matches"], 1)
        per_video[str(video_id)] = local
    temporal["conditional_switch_rate"] = temporal["switched_id_pairs"] / max(temporal["pairs_with_both_track_matches"], 1)
    return {
        "counts": counts,
        "geometry": {
            "track_observation_precision": counts["matched_track_observations"] / max(counts["predicted_track_observations"], 1),
            "track_observation_recall": counts["matched_track_observations"] / max(counts["gt_observations"], 1),
        },
        "temporal": temporal,
        "per_video": per_video,
        "stream_rows_scanned": rows_scanned,
        "streams": stream_records,
    }


def _select_gpu() -> tuple[str, str]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    )
    occupied = set(subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    ).split())
    for line in query.splitlines():
        uuid, used, free, utilization = [value.strip() for value in line.split(",")]
        if uuid not in occupied and int(used) < 100 and int(free) >= 9216 and int(utilization) == 0:
            return uuid, query
    raise RuntimeError("no fresh idle GPU with registered headroom")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    cfg = _load_config()
    output = OUTPUT
    run = RUN_ROOT
    geometry_path = ROOT / cfg["inputs"]["geometry_only_gt_index"]
    stream_root = ROOT / cfg["inputs"]["covtrack_stream_root"]
    if args.worker is not None:
        return _worker(args.worker, cfg)
    if output.exists() or run.exists():
        raise RuntimeError("preserve existing multi-clip smoke; refusing overwrite")
    input_data = _build_fixed_input(cfg)
    gpu_uuid, gpu_query = _select_gpu()
    run.mkdir(parents=True)
    atomic_json(run / "worker_input.json", input_data)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(run / "matplotlib"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    log = (run / "worker.log").open("wb")
    process = subprocess.Popen(
        [str(MASA_ENV / "bin/python"), str(Path(__file__).resolve()), "--worker", str(run)],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    guard_error = None
    started = time.monotonic()
    try:
        while process.poll() is None:
            memory = {line.split(":", 1)[0]: line.split(":", 1)[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
            available = int(memory["MemAvailable"].split()[0]) * 1024
            total = int(memory["MemTotal"].split()[0]) * 1024
            if time.monotonic() - started > float(cfg["resources"]["wall_seconds_ceiling"]):
                guard_error = "parent_wall_ceiling"
                break
            if available < total * 0.25:
                guard_error = "system_RAM_reserve"
                break
            if run.exists() and sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > int(cfg["resources"]["output_bytes_ceiling"]):
                guard_error = "parent_output_ceiling"
                break
            time.sleep(1)
    finally:
        if guard_error and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
        log.close()
    worker_status = json.loads((run / "worker_status.json").read_text()) if (run / "worker_status.json").exists() else {"status": "NO_WORKER_RECEIPT"}
    result: dict[str, Any] = {
        "schema_version": "trackocd.cvpr.masa-covtrack-multi-train-geometry-smoke-result.v1",
        "generated_utc": _now(),
        "status": "BLOCKED_CATEGORY_FREE_MULTI_TRAIN_PREDICTION",
        "config": {"path": str(CONFIG.resolve()), "sha256": _sha256(CONFIG)},
        "selection": input_data,
        "gpu_uuid": gpu_uuid,
        "fresh_gpu_query": gpu_query,
        "worker_returncode": process.returncode,
        "worker_guard_error": guard_error,
        "worker": worker_status,
        "boundary": cfg["boundaries"],
    }
    if worker_status.get("status") == "SEALED_CATEGORY_FREE_MULTI_TRAIN_PREDICTIONS" and guard_error is None:
        masa = _evaluate_masa(run, input_data, geometry_path)
        covtrack = _evaluate_covtrack(input_data, stream_root, geometry_path)
        result["status"] = "COMPLETE_CATEGORY_FREE_MULTI_FRONTEND_TRAIN_GEOMETRY_SMOKE_NOT_QUALIFICATION"
        result["masa"] = masa
        result["covtrack_native"] = covtrack
        result["interpretation"] = (
            "Fixed same-image Train geometry/continuity smoke only. It does not qualify a frontend, "
            "select a model, create an OCD/MOT metric, or accept Goal A/B/C."
        )
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "worker_returncode": process.returncode,
                      "worker_guard_error": guard_error, "masa": result.get("masa", {}),
                      "covtrack_native": result.get("covtrack_native", {})}, sort_keys=True))
    return 0 if result["status"].startswith("COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
