#!/usr/bin/env python3
"""Extract a bounded, category-free DINO pilot from one frozen Train stream video.

This is input preparation only.  It intentionally does not train, read GT,
read roles/vocabulary, access Val/Test, or invoke the formal v2 cache builder.
The large feature payload stays local; only the compact receipt is suitable
for version control.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import atomic_json, sha256_file  # noqa: E402
from scripts.trackocd_v2.build_common_features import crop_box  # noqa: E402


_CONFIG_ENV = os.environ.get(
    "TRACKOCD_FEATURE_PILOT_CONFIG",
    str(ROOT / "configs/trackocd_cvpr/predicted_train_feature_pilot_v1.json"),
)
CONFIG = Path(_CONFIG_ENV)
if not CONFIG.is_absolute():
    CONFIG = ROOT / CONFIG
PREFIXES = (1, 2, 4, 8, 16)
FORBIDDEN_TOP_LEVEL_FIELDS = {
    "category_id",
    "category_ids",
    "class_id",
    "class_name",
    "label",
    "labels",
    "role",
    "vocabulary",
}


def now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def sha256_and_size(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def install_input_barrier() -> None:
    """Reject network, child processes, and prohibited dataset paths."""

    def audit(event: str, args: Tuple[Any, ...]) -> None:
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.sendto",
            "subprocess.Popen",
            "os.system",
        }:
            raise PermissionError(f"pilot forbids external activity: {event}")
        if event == "open" and args and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            forbidden = (
                "/TAO-Amodal/annotations/",
                "/frames/val/",
                "/frames/test/",
                "/recovered_splits/",
                "/roles.json",
                "/train_labels.parquet",
                "/evaluator_only_geometry_join.json",
            )
            if any(token in path for token in forbidden):
                raise PermissionError(f"pilot forbids prohibited input: {path}")

    sys.addaudithook(audit)


def read_vendor_commit(repo: Path) -> str:
    git_dir = repo / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[6:]
        ref_path = git_dir / ref
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and " " in line:
                sha, name = line.split(" ", 1)
                if name == ref:
                    return sha
        raise RuntimeError(f"unresolved vendor ref: {ref}")
    return head


def validate_box(box: Sequence[Any]) -> List[float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not np.isfinite(values).all():
        raise ValueError("non-finite or malformed box")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("degenerate physical box")
    return values


def scan_source(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = ROOT / cfg["source_stream"]
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = cfg["fixed_selection"]
    source_hash = hashlib.sha256()
    source_bytes = 0
    source_rows = 0
    source_videos = set()
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    selected_obs = 0
    with path.open("rb") as handle:
        for raw in handle:
            source_hash.update(raw)
            source_bytes += len(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            source_rows += 1
            video_id = int(row["video_id"])
            source_videos.add(video_id)
            if FORBIDDEN_TOP_LEVEL_FIELDS.intersection(row):
                raise ValueError("semantic field found in physical stream row")
            if video_id != int(expected["video_id"]):
                continue
            source_split = str(row.get("source_split", ""))
            if source_split != "train_predicted":
                raise ValueError(f"unexpected source_split: {source_split}")
            frame_ids = [int(value) for value in row["frame_ids"]]
            image_paths = [str(value) for value in row["image_paths"]]
            boxes = [validate_box(box) for box in row["boxes_xyxy"]]
            quality = [float(value) for value in row["quality"]]
            if not frame_ids or len(frame_ids) != len(image_paths) or len(frame_ids) != len(boxes) or len(frame_ids) != len(quality):
                raise ValueError("physical row arrays are not aligned")
            if frame_ids != sorted(frame_ids) or len(set(frame_ids)) != len(frame_ids):
                raise ValueError("frame ids are not strictly ordered within track")
            if not np.isfinite(quality).all() or min(quality) < 0:
                raise ValueError("invalid physical quality values")
            for rel in image_paths:
                rel_path = Path(rel)
                if rel_path.is_absolute() or ".." in rel_path.parts or not rel.startswith("train/"):
                    raise ValueError(f"non-Train or unsafe image path: {rel}")
            identity = (video_id, str(row["physical_track_id"]))
            if identity in selected_ids:
                raise ValueError(f"duplicate physical track: {identity}")
            selected_ids.add(identity)
            selected.append({
                "sample_key": str(row["sample_key"]),
                "source_sample_id": str(row.get("source_sample_id", row["sample_key"])),
                "video_id": video_id,
                "physical_track_id": str(row["physical_track_id"]),
                "frame_ids": frame_ids,
                "image_paths": image_paths,
                "boxes_xyxy": boxes,
                "quality": quality,
                "source_split": source_split,
            })
            selected_obs += len(frame_ids)
    actual_hash = source_hash.hexdigest()
    if source_bytes != int(cfg["source_size_bytes"]) or actual_hash != cfg["source_sha256"]:
        raise ValueError(f"source receipt mismatch: {source_bytes} {actual_hash}")
    if len(selected) != int(expected["expected_tracks"]) or selected_obs != int(expected["expected_observations"]):
        raise ValueError(f"fixed selection mismatch: tracks={len(selected)} observations={selected_obs}")
    return selected, {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": source_bytes,
        "sha256": actual_hash,
        "source_rows": source_rows,
        "source_videos": len(source_videos),
        "selected_tracks": len(selected),
        "selected_observations": selected_obs,
    }


def normalized_weighted(values: np.ndarray, quality: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(list(quality), dtype=np.float32)
    if len(values) == 0 or len(values) != len(weights):
        raise ValueError("empty or misaligned prefix")
    if float(weights.sum()) <= 1e-12:
        weights = np.ones(len(values), dtype=np.float32)
    result = (values * weights[:, None]).sum(axis=0) / float(weights.sum())
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid prefix feature")
    return (result / norm).astype(np.float16)


def resolve_image(rel: str, frames_root: Path) -> Path:
    path = (frames_root / rel).resolve()
    if os.path.commonpath((str(path), str(frames_root))) != str(frames_root):
        raise ValueError(f"image escapes frames root: {rel}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_model(cfg: Dict[str, Any], torch: Any) -> Any:
    encoder = cfg["encoder"]
    repo = ROOT / encoder["local_repository"]
    checkpoint = ROOT / encoder["checkpoint"]
    if read_vendor_commit(repo) != encoder["repository_commit"]:
        raise ValueError("DINO repository commit mismatch")
    if sha256_file(checkpoint) != encoder["checkpoint_sha256"]:
        raise ValueError("DINO checkpoint hash mismatch")
    if sha256_file(ROOT / encoder["common_config"]) != encoder["common_config_sha256"]:
        raise ValueError("common encoder config hash mismatch")
    sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitb14

    model = dinov2_vitb14(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("encoder is not frozen")
    return model


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["output_directory"]
    result_path = ROOT / cfg["result_path"]
    if output.exists() or result_path.exists():
        raise RuntimeError("pilot output/result already exists; refusing overwrite")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices or "," in visible_devices:
        raise RuntimeError("launch this pilot with exactly one freshly selected CUDA_VISIBLE_DEVICES value")
    if shutil.disk_usage(ROOT).free < int(cfg["resources"]["disk_free_reserve_bytes"]) + int(cfg["resources"]["output_bytes_ceiling"]):
        raise RuntimeError("insufficient disk reserve for bounded pilot")

    selected, source = scan_source(cfg)
    output.mkdir(parents=True, exist_ok=False)
    install_input_barrier()

    import torch
    from PIL import Image
    from torchvision import transforms

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("pilot requires exactly one visible CUDA device")
    torch.set_num_threads(1)
    torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    model = load_model(cfg, torch)
    torch.cuda.set_per_process_memory_fraction(0.75, device=0)
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    encoder = cfg["encoder"]
    transform = transforms.Compose([
        transforms.Resize(tuple(encoder["resize"]), interpolation=Image.Resampling.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(encoder["normalization_mean"], encoder["normalization_std"]),
    ])
    frames_root = (ROOT / "data/raw/tao/frames").resolve()
    total = int(source["selected_observations"])
    visual = np.empty((total, 768), dtype=np.float16)
    geometry = np.empty((total, 4), dtype=np.float32)
    quality = np.empty(total, dtype=np.float32)
    frame_ids = np.empty(total, dtype=np.int64)
    offsets = np.empty(len(selected) + 1, dtype=np.int64)
    batch_tensors: List[Any] = []
    batch_offsets: List[int] = []
    cursor = 0
    forwards = 0
    started = time.monotonic()
    batch_size = int(cfg["resources"]["batch_size"])

    def check_limits() -> None:
        if peak_rss_bytes() > int(cfg["resources"]["rss_bytes_ceiling"]):
            raise MemoryError("pilot RSS ceiling exceeded")
        reserved = int(torch.cuda.max_memory_reserved(device))
        free, _ = torch.cuda.mem_get_info(device)
        if reserved > int(cfg["resources"]["gpu_peak_bytes_ceiling"]):
            raise MemoryError("pilot GPU peak ceiling exceeded")
        if free < int(cfg["resources"]["gpu_free_reserve_bytes"]):
            raise MemoryError("pilot GPU free reserve violated")

    def flush() -> None:
        nonlocal forwards, batch_tensors, batch_offsets
        if not batch_tensors:
            return
        with torch.inference_mode():
            batch = torch.stack(batch_tensors, dim=0).to(device)
            values = model.forward_features(batch)["x_norm_clstoken"]
            values = torch.nn.functional.normalize(values, dim=-1).cpu().numpy().astype(np.float16)
        if values.shape != (len(batch_offsets), 768) or not np.isfinite(values).all():
            raise ValueError("invalid DINO feature batch")
        visual[np.asarray(batch_offsets, dtype=np.int64)] = values
        forwards += 1
        batch_tensors = []
        batch_offsets = []
        check_limits()
        atomic_json(output / "progress.json", {
            "status": "RUNNING_FIXED_VIDEO_DINO_FEATURE_PILOT",
            "completed_observations": int(np.count_nonzero(np.isfinite(visual[:, 0]))),
            "total_observations": total,
            "forward_batches": forwards,
            "elapsed_seconds": time.monotonic() - started,
            "pid": os.getpid(),
        })

    for track_index, row in enumerate(selected):
        offsets[track_index] = cursor
        for frame_id, rel, box, score in zip(row["frame_ids"], row["image_paths"], row["boxes_xyxy"], row["quality"]):
            image_path = resolve_image(rel, frames_root)
            with Image.open(image_path) as raw:
                image = raw.convert("RGB")
            width, height = image.size
            crop = crop_box(image, box, float(encoder["crop_context_fraction_per_side"]))
            if min(crop.size) < 4:
                raise ValueError("degenerate crop")
            offset = cursor
            geometry[offset] = np.asarray(box, dtype=np.float32) / np.asarray([width, height, width, height], dtype=np.float32)
            quality[offset] = float(score)
            frame_ids[offset] = int(frame_id)
            batch_tensors.append(transform(crop))
            batch_offsets.append(offset)
            cursor += 1
            if len(batch_tensors) >= batch_size:
                flush()
        offsets[track_index + 1] = cursor
    flush()
    if cursor != total or not np.isfinite(visual).all():
        raise RuntimeError(f"incomplete feature extraction: {cursor}/{total}")
    if not np.isfinite(geometry).all() or not np.isfinite(quality).all():
        raise RuntimeError("non-finite non-visual arrays")
    norms = np.linalg.norm(visual.astype(np.float32), axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 3e-3:
        raise RuntimeError("DINO output is not normalized")

    prefix_features = np.empty((len(selected), len(PREFIXES), 768), dtype=np.float16)
    for track_index, row in enumerate(selected):
        begin, end = int(offsets[track_index]), int(offsets[track_index + 1])
        for prefix_index, prefix in enumerate(PREFIXES):
            prefix_end = min(begin + int(prefix), end)
            prefix_features[track_index, prefix_index] = normalized_weighted(
                visual[begin:prefix_end], quality[begin:prefix_end]
            )

    video_id = int(cfg["fixed_selection"]["video_id"])
    payload = output / f"video_{video_id:04d}_prefix_features.npz"
    temporary_payload = payload.with_name(f".{payload.name}.tmp.{os.getpid()}")
    with temporary_payload.open("wb") as handle:
        np.savez_compressed(
            handle,
            visual=visual,
            geometry_normalized_xyxy=geometry,
            quality=quality,
            frame_ids=frame_ids,
            track_offsets=offsets,
            prefix_features=prefix_features,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_payload, payload)

    index_path = output / "track_index.jsonl"
    temporary_index = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
    with temporary_index.open("w", encoding="utf-8") as handle:
        for track_index, row in enumerate(selected):
            begin, end = int(offsets[track_index]), int(offsets[track_index + 1])
            handle.write(json.dumps({
                "video_id": row["video_id"],
                "physical_track_id": row["physical_track_id"],
                "sample_key": row["sample_key"],
                "source_sample_id": row["source_sample_id"],
                "source_split": row["source_split"],
                "frame_ids": row["frame_ids"],
                "offset": begin,
                "count": end - begin,
            }, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_index, index_path)
    check_limits()
    torch.cuda.synchronize(device)

    payload_files = [payload, index_path]
    payload_bytes = sum(path.stat().st_size for path in payload_files)
    if payload_bytes > int(cfg["resources"]["output_bytes_ceiling"]):
        raise RuntimeError("pilot output ceiling exceeded")
    if shutil.disk_usage(ROOT).free < int(cfg["resources"]["disk_free_reserve_bytes"]):
        raise RuntimeError("post-pilot disk reserve violated")
    result = {
        "schema_version": "trackocd.cvpr.predicted-train-feature-pilot-result.v1",
        "status": "PASS_FIXED_SINGLE_VIDEO_DINO_FEATURE_PILOT_NO_TRAINING",
        "created_utc": now(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source": source,
        "fixed_selection": cfg["fixed_selection"],
        "encoder": {
            **cfg["encoder"],
            "repository_commit_verified": True,
            "checkpoint_hash_verified": True,
            "frozen": True,
            "training_mode": False,
        },
        "features": {
            "tracks": len(selected),
            "observations": total,
            "prefixes": list(PREFIXES),
            "arrays": [
                "visual",
                "geometry_normalized_xyxy",
                "quality",
                "frame_ids",
                "track_offsets",
                "prefix_features",
            ],
            "payload_local_path": str(payload.relative_to(ROOT)),
            "payload_bytes": payload.stat().st_size,
            "payload_sha256": sha256_file(payload),
            "index_local_path": str(index_path.relative_to(ROOT)),
            "index_bytes": index_path.stat().st_size,
            "index_sha256": sha256_file(index_path),
            "payload_uploaded_to_github": False,
        },
        "resources": {
            "device_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "batch_size": batch_size,
            "forward_batches": forwards,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "disk_free_bytes_after": shutil.disk_usage(ROOT).free,
            "training_gpu_hours": 0.0,
        },
        "boundaries": cfg["boundaries"],
        "interpretation": "Engineering/input-interface pilot only; no formal v2 cache, no MOT/OCD metric, no model-selection result, and no claim about Train scientific performance.",
    }
    atomic_json(result_path, result)
    print(json.dumps({"status": result["status"], "tracks": len(selected), "observations": total, "wall_seconds": result["resources"]["wall_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
