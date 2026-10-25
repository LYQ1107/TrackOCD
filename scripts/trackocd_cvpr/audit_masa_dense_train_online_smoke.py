#!/usr/bin/env python3
"""Run a bounded frozen MASA smoke on adjacent Train images only.

This intentionally exercises the current-image causal path on consecutive
frames.  It never opens annotations, split metadata, Val/Test frames, or
semantic fields, and writes only compact frame/lifecycle statistics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FRAMES = Path("/data3/liuyeqiang/TAO-Amodal/frames")
MASA_ENV = Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")
CONFIG = ROOT / "configs/trackocd_cvpr/masa_dense_train_online_smoke_v1.json"
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_smoke_v1/result.json"
RUN_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/masa_dense_train_online_smoke_v1/run"

import sys

sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file  # noqa: E402


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    if value.get("schema_version") != "trackocd.cvpr.masa-dense-train-online-smoke.v1":
        raise ValueError("unexpected dense MASA smoke configuration")
    paths = value.get("image_paths")
    if not isinstance(paths, list) or len(paths) != 8 or len(set(paths)) != 8:
        raise ValueError("exactly eight distinct adjacent-frame paths are required")
    if paths != sorted(paths):
        raise ValueError("registered image order must be deterministic")
    return value


def _image_paths(config: dict[str, Any]) -> list[Path]:
    result: list[Path] = []
    train_root = (FRAMES / "train").resolve()
    for raw in config["image_paths"]:
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "train":
            raise ValueError("registered input is not Train-only")
        path = FRAMES / relative
        if not path.resolve().is_relative_to(train_root) or not path.is_file():
            raise ValueError("registered Train image is missing or escapes the Train tree")
        result.append(path)
    if len({path.parent for path in result}) != 1:
        raise ValueError("registered frames must come from one video directory")
    numbers = [int(path.stem) for path in result]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("registered frames are not consecutive")
    return result


def _deny_worker_inputs() -> None:
    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("dense MASA worker forbids network and child processes: " + event)
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            forbidden = (
                "/TAO-Amodal/annotations/",
                "/recovered_splits/",
                "/frames/val/",
                "/frames/test/",
                "evaluator_only_geometry_join.json",
                "/roles.json",
                "known_ids.json",
                "unknown_ids_val.json",
            )
            if any(token in path for token in forbidden):
                raise PermissionError("dense MASA worker cannot open GT, split, or Val/Test input")

    sys.addaudithook(audit)


def _snapshot(instances: Any, *, tracks: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
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
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or np.any(boxes[:, 2:] <= boxes[:, :2])
        or np.any((scores < 0) | (scores > 1))
        or np.any(labels != 0)
    ):
        raise ValueError("non-finite, degenerate, or non-anonymous MASA output")
    return boxes, scores, labels, ids


def worker(run: Path) -> int:
    started = time.monotonic()
    status: dict[str, Any] = {
        "schema_version": "trackocd.cvpr.masa-dense-train-online-worker.v1",
        "status": "RUNNING",
        "started_utc": _now(),
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
        config = _config()
        images = _image_paths(config)
        _deny_worker_inputs()
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
        model.tracker.reset()
        frame_rows: list[dict[str, Any]] = []
        previous_ids: set[int] | None = None
        for ordinal, path in enumerate(images):
            if time.monotonic() - started > 600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3:
                raise RuntimeError("registered dense MASA smoke resource ceiling")
            with Image.open(path) as raw:
                pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
            detections, tracks = infer_current_frame(model, pixels, ordinal, "cuda:0")
            torch.cuda.synchronize()
            det_boxes, det_scores, _, _ = _snapshot(detections, tracks=False)
            track_boxes, track_scores, _, track_ids = _snapshot(tracks, tracks=True)
            assert track_ids is not None
            current_ids = {int(value) for value in track_ids}
            retained = len(previous_ids & current_ids) if previous_ids is not None else 0
            frame_rows.append(
                {
                    "ordinal": ordinal,
                    "image_filename": path.name,
                    "image_sha256": sha256_file(path),
                    "detections": int(len(det_scores)),
                    "tracks": int(len(track_ids)),
                    "retained_track_ids_from_previous_frame": retained,
                    "finite_boxes": bool(np.isfinite(det_boxes).all() and np.isfinite(track_boxes).all()),
                    "foreground_label_only": bool(len(track_ids) == 0 or np.all(np.asarray(tracks.labels.detach().cpu()) == 0)),
                }
            )
            previous_ids = current_ids
            status["frames_completed"] = ordinal + 1
            atomic_json(run / "worker_status.json", status)
            if torch.cuda.max_memory_reserved() > 8 * 1024**3:
                raise RuntimeError("registered dense MASA GPU memory ceiling")
        status.update(
            status="SEALED_ADJACENT_TRAIN_IMAGE_PREDICTIONS",
            frames=frame_rows,
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    except Exception as exc:
        status.update(status="BLOCKED_ADJACENT_TRAIN_IMAGE_PREDICTIONS", error=type(exc).__name__ + ": " + str(exc))
        import traceback

        traceback.print_exc()
    status.update(
        elapsed_seconds=time.monotonic() - started,
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        completed_utc=_now(),
    )
    atomic_json(run / "worker_status.json", status)
    return 0 if status["status"].startswith("SEALED") else 1


def _select_gpu() -> tuple[str, str]:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    occupied = set(
        subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).split()
    )
    for line in query.splitlines():
        uuid, used, free, utilization = [value.strip() for value in line.split(",")]
        if uuid not in occupied and int(used) < 100 and int(free) >= 9216 and int(utilization) == 0:
            return uuid, query
    raise RuntimeError("no freshly idle GPU with registered headroom")


def parent() -> int:
    if RUN_ROOT.exists() or OUTPUT.exists():
        raise ValueError("preserve an existing dense MASA smoke; no automatic rerun")
    config = _config()
    images = _image_paths(config)
    RUN_ROOT.mkdir(parents=True)
    atomic_json(
        RUN_ROOT / "worker_input.json",
        {"config_sha256": sha256_file(CONFIG), "image_paths": config["image_paths"], "frame_count": len(images)},
    )
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
    guard_error = None
    started = time.monotonic()
    try:
        while process.poll() is None:
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(memory["MemAvailable"].split()[0]) * 1024
            total = int(memory["MemTotal"].split()[0]) * 1024
            if time.monotonic() - started > 600 or available < total * 0.25 or sum(path.stat().st_size for path in RUN_ROOT.rglob("*") if path.is_file()) > 10 * 1024**2:
                raise RuntimeError("registered dense MASA smoke resource guard")
            time.sleep(1)
    except BaseException as exc:
        guard_error = type(exc).__name__ + ": " + str(exc)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
    status_path = RUN_ROOT / "worker_status.json"
    worker_status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "NO_WORKER_RECEIPT"}
    result = {
        "schema_version": "trackocd.cvpr.masa-dense-train-online-smoke-result.v1",
        "status": "PASS_BOUNDED_MASA_DENSE_FRAME_ONLINE_TRAIN_SMOKE_NO_GT" if worker_status.get("status", "").startswith("SEALED") and guard_error is None else "BLOCKED_BOUNDED_MASA_DENSE_FRAME_ONLINE_TRAIN_SMOKE",
        "generated_utc": _now(),
        "config": str(CONFIG.resolve()),
        "config_sha256": sha256_file(CONFIG),
        "run_root": str(RUN_ROOT.resolve()),
        "gpu_uuid": gpu_uuid,
        "fresh_gpu_query": gpu_query,
        "worker_returncode": process.returncode,
        "worker_guard_error": guard_error,
        "worker": worker_status,
        "boundaries": {
            "train_images_only": True,
            "gt_or_category_fields_read": False,
            "val_or_test_accessed": False,
            "training_started": False,
            "formal_metrics_created": False,
            "model_selection": False,
            "external_processes_touched": False,
            "raw_predictions_persisted": False,
        },
        "interpretation": config["interpretation"],
    }
    atomic_json(OUTPUT, result)
    print(json.dumps({"status": result["status"], "worker_returncode": process.returncode}, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker) if args.worker is not None else parent())
