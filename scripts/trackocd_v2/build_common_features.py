#!/usr/bin/env python3
"""Build the v2 causal common visual feature cache.

The cache is deliberately separate from the historical v1 cache.  Each
track keeps one DINOv2 descriptor per public observation, plus quality-
weighted causal aggregates for ``full`` and prefixes 1/2/4/8/16.  No
category, split role, or semantic label is read by the encoder.

This command is a bounded supervisor in its own right.  It selects only
GPUs with no compute applications, starts at most four one-GPU workers, and
leaves completion evidence per track so an interrupted run can resume.
Exit code 2 means that the requested work is valid but resources are not
available yet; callers should record a wait and invoke this command again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MAX_WORKERS = 4
WORKER_RAM_GIB = 8.0
DEFAULT_BATCH = 16
DEFAULT_HUBS = (
    Path("/home/user/.cache/torch/hub/facebookresearch_dinov2_main"),
    Path("/home/lwr/.cache/torch/hub/facebookresearch_dinov2_main"),
)
FRAMES_ROOT = (ROOT / "data/raw/tao/frames").resolve()
MANIFESTS = OUTPUT_TARGET / "manifests"
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"
PRED_FEATURE_ROOT = OUTPUT_TARGET / "features/pred_tracks"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_meminfo() -> Dict[str, int]:
    values: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if ":" in line:
            key, rest = line.split(":", 1)
            fields = rest.split()
            if fields:
                values[key] = int(fields[0])
    return values


def _csv_lines(command: Sequence[str]) -> List[str]:
    try:
        output = __import__("subprocess").check_output(command, text=True, stderr=__import__("subprocess").STDOUT)
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def resource_snapshot() -> Dict[str, Any]:
    """Return a read-only resource snapshot suitable for audit evidence."""

    mem = read_meminfo()
    process_count = sum(1 for _ in Path("/proc").glob("[0-9]*"))
    gpu_rows = _csv_lines([
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    app_rows = _csv_lines([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ])
    apps_by_uuid: Dict[str, List[str]] = {}
    for row in app_rows:
        uuid = row.split(",", 1)[0].strip()
        apps_by_uuid.setdefault(uuid, []).append(row)
    parsed_gpus: List[Dict[str, Any]] = []
    for row in gpu_rows:
        fields = [x.strip() for x in row.split(",")]
        if len(fields) < 6:
            continue
        index, uuid = int(fields[0]), fields[1]
        parsed_gpus.append({
            "index": index,
            "uuid": uuid,
            "memory_used_mib": int(float(fields[2])),
            "memory_free_mib": int(float(fields[3])),
            "memory_total_mib": int(float(fields[4])),
            "utilization_gpu_percent": int(float(fields[5])),
            "compute_apps": apps_by_uuid.get(uuid, []),
        })
    return {
        "time": utc_now(),
        "mem_total_kib": mem.get("MemTotal"),
        "mem_available_kib": mem.get("MemAvailable"),
        "process_count": process_count,
        "gpu_rows": parsed_gpus,
        "compute_app_rows": app_rows,
    }


def eligible_gpus(snapshot: Dict[str, Any], requested: int) -> List[int]:
    """Select idle GPUs, preserving physical indices and a hard worker cap."""

    candidates = []
    for gpu in snapshot.get("gpu_rows", []):
        if gpu.get("compute_apps"):
            continue
        if int(gpu.get("memory_free_mib", 0)) < 8192:
            continue
        candidates.append(int(gpu["index"]))
    total_kib = snapshot.get("mem_total_kib") or 0
    available_kib = snapshot.get("mem_available_kib") or 0
    floor_kib = int(total_kib * 0.25)
    max_by_ram = max(0, int((available_kib - floor_kib) / (WORKER_RAM_GIB * 1024 * 1024)))
    return candidates[: min(MAX_WORKERS, max(1, requested), max_by_ram)] if max_by_ram else []


def load_rows(split: str) -> List[Tuple[str, Dict[str, Any], str]]:
    names = {
        "train": "tao_train_gt_tracks.jsonl",
        "val": "tao_val_gt_tracks.jsonl",
        "pred": "tao_val_predicted_tracks.jsonl",
    }
    selected = ("train", "val") if split == "both" else (split,)
    rows: List[Tuple[str, Dict[str, Any], str]] = []
    for source_split in selected:
        path = MANIFESTS / names[source_split]
        if not path.exists():
            raise FileNotFoundError(path)
        manifest_sha = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append((source_split, json.loads(line), manifest_sha))
    return rows


def cache_path(source_split: str, sample_key: str) -> Path:
    root = PRED_FEATURE_ROOT if source_split == "pred" else FEATURE_ROOT
    return root / source_split / (str(sample_key) + ".json")


def marker_path(output: Path, suffix: str) -> Path:
    return output.with_name(output.name + suffix)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def reclaim_stale_markers(rows: Iterable[Tuple[str, Dict[str, Any], str]]) -> Dict[str, int]:
    """Reclaim only markers whose recorded worker PID is no longer alive."""

    reclaimed = active = 0
    for source_split, row, _ in rows:
        output = cache_path(source_split, row["sample_key"])
        launch = marker_path(output, ".launched")
        done = marker_path(output, ".done")
        if done.exists():
            continue
        if not launch.exists():
            continue
        try:
            payload = json.loads(launch.read_text())
            pid = int(payload["pid"])
            os.kill(pid, 0)
            active += 1
        except (FileNotFoundError, ProcessLookupError, ValueError, KeyError, PermissionError):
            _unlink(launch)
            reclaimed += 1
    return {"reclaimed_stale": reclaimed, "active_launches": active}


def pending_rows(rows: Iterable[Tuple[str, Dict[str, Any], str]]) -> List[Tuple[str, Dict[str, Any], str]]:
    pending: List[Tuple[str, Dict[str, Any], str]] = []
    for item in rows:
        source_split, row, _ = item
        output = cache_path(source_split, row["sample_key"])
        if marker_path(output, ".done").exists() and output.exists():
            continue
        if marker_path(output, ".launched").exists():
            continue
        pending.append(item)
    return pending


def crop_box(image: Any, box: Sequence[float], context: float = 0.10) -> Any:
    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    xa = max(0.0, cx - bw * (1.0 + 2.0 * context) * 0.5)
    ya = max(0.0, cy - bh * (1.0 + 2.0 * context) * 0.5)
    xb = min(float(width), cx + bw * (1.0 + 2.0 * context) * 0.5)
    yb = min(float(height), cy + bh * (1.0 + 2.0 * context) * 0.5)
    if xb - xa < 2 or yb - ya < 2:
        xa, ya, xb, yb = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    left, top, right, bottom = int(xa), int(ya), int(xb), int(yb)
    # A valid object can touch an image boundary and be only a few pixels
    # wide.  Keep that observation instead of dropping the track: expand only
    # the integer crop window to the minimum usable raster size.
    if right - left < 4:
        if left + 4 <= width:
            right = left + 4
        else:
            left, right = max(0, width - 4), width
    if bottom - top < 4:
        if top + 4 <= height:
            bottom = top + 4
        else:
            top, bottom = max(0, height - 4), height
    return image.crop((left, top, right, bottom))


def normalized_mean(values: np.ndarray, quality: Sequence[float], end: int) -> np.ndarray:
    values = np.asarray(values[:end], dtype=np.float32)
    weights = np.asarray(list(quality)[:end], dtype=np.float32)
    if weights.size == 0 or float(weights.sum()) <= 1e-12:
        weights = np.ones(end, dtype=np.float32)
    result = (values * weights[:, None]).sum(axis=0) / float(weights.sum())
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("zero or non-finite aggregate feature")
    return (result / norm).astype(np.float16)


def claim(output: Path, worker_pid: int) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if marker_path(output, ".done").exists() and output.exists():
        return False
    marker = marker_path(output, ".launched")
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump({"pid": worker_pid, "parent_pid": os.getppid(), "started_utc": utc_now()}, handle)
    except FileExistsError:
        return False
    return True


def extract_one(row: Dict[str, Any], source_split: str, manifest_sha: str, output: Path, device_index: int, batch_size: int, model: Any, transform: Any, torch: Any, image_cls: Any, model_repo: Path) -> None:
    image_paths = list(row["image_paths"])
    boxes = list(row["boxes_xyxy"])
    if len(image_paths) != len(boxes) or not image_paths:
        raise ValueError("image/box lineage is empty or misaligned")
    tensors: List[Any] = []
    embeddings: List[np.ndarray] = []

    def flush() -> None:
        nonlocal tensors
        if not tensors:
            return
        batch = torch.cat(tensors, dim=0).to("cuda:0")
        with torch.no_grad():
            output_features = model.forward_features(batch)
            values = torch.nn.functional.normalize(output_features["x_norm_clstoken"], dim=-1)
        embeddings.extend(values.cpu().numpy().astype(np.float32))
        tensors = []

    for image_path, box in zip(image_paths, boxes):
        path = FRAMES_ROOT / str(image_path)
        with image_cls.open(path) as raw:
            crop = crop_box(raw.convert("RGB"), box)
        if min(crop.size) < 4:
            raise ValueError("degenerate crop")
        tensors.append(transform(crop).unsqueeze(0))
        if len(tensors) >= batch_size:
            flush()
    flush()
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (len(image_paths), 768):
        raise ValueError("unexpected DINOv2 feature shape: %s" % (values.shape,))
    if not np.isfinite(values).all():
        raise ValueError("non-finite DINOv2 feature")
    quality = list(row.get("quality", [1.0] * len(values)))
    prefix_features = {str(prefix): normalized_mean(values, quality, min(prefix, len(values))).tolist() for prefix in PREFIXES}
    result = {
        "schema_version": "trackocd.v2.common_feature.v1",
        "sample_key": str(row["sample_key"]),
        "source_split": source_split,
        "video_id": int(row["video_id"]),
        "physical_track_id": str(row["physical_track_id"]),
        "frame_ids": [int(value) for value in row["frame_ids"]],
        "image_paths": image_paths,
        "boxes_xyxy": row["boxes_xyxy"],
        "quality": quality,
        "frame_embeddings": values.astype(np.float16).tolist(),
        "prefix_features": prefix_features,
        "full_feature": normalized_mean(values, quality, len(values)).tolist(),
        "prefix_observations_used": {str(prefix): min(prefix, len(values)) for prefix in PREFIXES},
        "feature": {
            "encoder": "DINOv2 ViT-B/14",
            "dimension": 768,
            "aggregation": "quality-weighted mean then L2 normalization",
            "text_or_category_logits": False,
            "category_id_feature": False,
            "physical_id_feature": False,
            "future_observations_used": False,
        },
        "lineage": {
            "manifest_sha256": manifest_sha,
            "frames_root": str(FRAMES_ROOT),
            "model_repo": str(model_repo.resolve()),
            "device_index": int(device_index),
        },
    }
    atomic_json(output, result)
    atomic_json(marker_path(output, ".done"), {"sample_key": str(row["sample_key"]), "finished_utc": utc_now(), "pid": os.getpid()})


def worker(worker_index: int, gpu_index: int, items: Sequence[Tuple[str, Dict[str, Any], str]], batch_size: int, hub_text: str, queue: Any) -> None:
    hub = Path(hub_text)
    completed = failed = skipped = 0
    errors: List[Dict[str, str]] = []
    # Imports and model construction happen once per bounded worker.  A
    # resource-only preflight therefore remains usable with system Python,
    # while actual extraction uses the CUDA environment supplied by the
    # caller.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    try:
        import torch
        from PIL import Image
        from torchvision import transforms

        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside feature worker")
        model = torch.hub.load(str(hub), "dinov2_vitb14", source="local").eval().to("cuda:0")
        transform = transforms.Compose([
            transforms.Resize((518, 518), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    except Exception as exc:
        queue.put({"worker": worker_index, "gpu": gpu_index, "completed": 0, "failed": 1, "skipped": 0, "errors": [{"sample_key": "__worker__", "error": repr(exc)}]})
        return
    for source_split, row, manifest_sha in items:
        output = cache_path(source_split, row["sample_key"])
        if not claim(output, os.getpid()):
            skipped += 1
            continue
        try:
            extract_one(row, source_split, manifest_sha, output, gpu_index, batch_size, model, transform, torch, Image, hub)
            completed += 1
        except Exception as exc:
            failed += 1
            errors.append({"sample_key": str(row["sample_key"]), "error": repr(exc)})
            atomic_json(marker_path(output, ".failed"), {"sample_key": str(row["sample_key"]), "error": repr(exc), "time": utc_now(), "worker_pid": os.getpid()})
            _unlink(marker_path(output, ".launched"))
            break
    queue.put({"worker": worker_index, "gpu": gpu_index, "completed": completed, "failed": failed, "skipped": skipped, "errors": errors})


def preflight(rows: Sequence[Tuple[str, Dict[str, Any], str]], requested_workers: int) -> Tuple[Dict[str, Any], List[int]]:
    snapshot = resource_snapshot()
    stale = reclaim_stale_markers(rows)
    snapshot["marker_recovery"] = stale
    snapshot["pending_units"] = len(pending_rows(rows))
    snapshot["requested_workers"] = requested_workers
    selected = eligible_gpus(snapshot, requested_workers)
    snapshot["selected_gpu_indices"] = selected
    atomic_json(OUTPUT_TARGET / "audit/feature_build_preflight.json", snapshot)
    return snapshot, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "pred", "both"), default="both")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--model-repo", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise SystemExit("--workers must be between 1 and 4")
    if args.batch < 1 or args.batch > 64:
        raise SystemExit("--batch must be between 1 and 64")
    ensure_output_layout()
    rows = load_rows(args.split)
    hub = args.model_repo or next((path for path in DEFAULT_HUBS if path.is_dir()), DEFAULT_HUBS[0])
    if not hub.is_dir():
        raise SystemExit("DINOv2 local hub repository is missing: %s" % hub)
    snapshot, gpus = preflight(rows, args.workers)
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0 if snapshot["pending_units"] == 0 or gpus else 2
    pending = pending_rows(rows)
    if not pending:
        atomic_json(OUTPUT_TARGET / "audit/common_feature_cache_manifest.json", {
            "schema_version": "trackocd.v2.common_feature_cache.v1",
            "status": "COMPLETE",
            "generated_utc": utc_now(),
            "split": args.split,
            "prefixes": list(PREFIXES),
            "pending_units": 0,
            "feature_root": str(FEATURE_ROOT),
        })
        return 0
    if not gpus:
        print("FEATURE_BUILD_WAITING_RESOURCE: no idle GPU with the RAM safety margin", file=sys.stderr)
        return 2
    worker_count = min(len(gpus), args.workers)
    assignments = [pending[index::worker_count] for index in range(worker_count)]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = []
    for index, (gpu, items) in enumerate(zip(gpus[:worker_count], assignments)):
        process = context.Process(target=worker, args=(index, gpu, items, args.batch, str(hub), queue), daemon=False)
        process.start()
        processes.append(process)
    results = []
    for process in processes:
        process.join()
    for _ in processes:
        results.append(queue.get())
    failed = [result for result in results if result.get("failed")]
    process_failed = any(process.exitcode != 0 for process in processes)
    atomic_json(OUTPUT_TARGET / "audit/common_feature_cache_manifest.json", {
        "schema_version": "trackocd.v2.common_feature_cache.v1",
        "status": "FAILED" if failed or process_failed else "COMPLETE",
        "generated_utc": utc_now(),
        "split": args.split,
        "prefixes": list(PREFIXES),
        "feature_root": str(FEATURE_ROOT),
        "workers": worker_count,
        "gpu_indices": gpus[:worker_count],
        "results": results,
        "process_exitcodes": [process.exitcode for process in processes],
    })
    return 1 if failed or process_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
