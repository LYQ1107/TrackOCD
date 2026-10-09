#!/usr/bin/env python3
"""Build the post-selection TrackOCD v2 feature cache.

This is the only formal predicted-feature route.  It is deliberately gated by
``FINAL_PHYSICAL_FRONTEND`` and writes bounded shards instead of one JSON file
per track:

* ``observations.npy``: float16 DINOv2 descriptors, one row per observation;
* ``prefix_features.npy``: float16 causal aggregates with shape
  ``[tracks, 5, 768]`` for prefixes 1/2/4/8/16;
* ``full_features.npy``: float16 full-track aggregates;
* ``index.parquet``: lineage and offsets for the arrays.

Each shard is committed atomically and has ``.launched``/``.done`` evidence,
so an interrupted run can resume at the shard boundary.  No category, text,
GT label, or physical identity is consumed by the encoder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402
from scripts.trackocd_v2.build_common_features import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_HUBS,
    FRAMES_ROOT,
    PREFIXES,
    WORKER_RAM_GIB,
    crop_box,
    eligible_gpus,
    normalized_mean,
    resource_snapshot,
)


FRONTENDS = {
    "simowt": "SimOWT/Q0",
    "ovtr": "OVTR-native",
    "covtrack_native": "COVTrack-native",
    "covtrack_nosem": "COVTrack-NoSemantic",
}
STAGE_FILES = {
    "simowt": "frontend_simowt.json",
    "ovtr": "frontend_ovtr.json",
    "covtrack_native": "frontend_covtrack_native.json",
    "covtrack_nosem": "frontend_covtrack_nosem.json",
}
FORBIDDEN_FIELDS = {
    "category_id",
    "category_ids",
    "category_name",
    "text",
    "gt_category_id",
    "gt_split",
    "gt_match_id",
}
DEFAULT_SHARD_TRACKS = 512
FORMAL_ROOT = OUTPUT_TARGET / "features/formal"
FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("physical stream line %d is not an object" % line_number)
            yield value


def _stage_path(slug: str) -> Path:
    return OUTPUT_TARGET / "audit" / STAGE_FILES[slug]


def _authorize(slug: str, split: str = "val") -> Dict[str, Any]:
    """Require the explicit final selection before formal predicted caching."""

    if split not in {"val", "test"}:
        raise ValueError("formal cache split must be val or test")
    if split == "test":
        # Test feature extraction is still category-free, but the route itself
        # is only legal after the immutable Val freeze.  The guard is checked
        # before any Test-side artifact is opened.
        assert_test_semantic_access_allowed(FINAL_FREEZE, "build frozen TAO Test feature cache")

    selection_path = OUTPUT_TARGET / "audit/frontend_selection.json"
    representation_path = OUTPUT_TARGET / "audit/representation_decision.json"
    stage_path = (
        OUTPUT_TARGET / "audit" / f"frontend_{slug}_test.json"
        if split == "test"
        else _stage_path(slug)
    )
    for path in (selection_path, representation_path, stage_path):
        if not path.is_file():
            raise RuntimeError("formal cache authorization artifact is missing: %s" % path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    frontend = FRONTENDS[slug]
    if selection.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
        raise RuntimeError(
            "FORMAL_CACHE_NOT_AUTHORIZED: frontend selection is %s"
            % selection.get("status")
        )
    if selection.get("selected_frontend") != frontend:
        raise RuntimeError("formal cache frontend does not match selected frontend")
    if representation.get("status") != "FINAL_REPRESENTATION_SELECTED":
        raise RuntimeError("formal cache representation decision is not final")
    if representation.get("formal_common_feature_cache_authorized") is not True:
        raise RuntimeError("formal cache authorization flag is not true")
    required_stage_status = "FINAL_TEST_NATIVE_STREAM_NORMALIZED" if split == "test" else "BAKEOFF_METRICS_COMPLETE"
    if stage.get("status") != required_stage_status:
        raise RuntimeError("selected frontend has not passed the required frontend gate")
    required_fields = ["native_run_complete", "physical_stream_contract_complete"]
    if split == "val":
        required_fields.extend([
            "same_v2_physical_metric_protocol",
            "same_v2_evaluator_contract",
            "requested_metrics_complete",
        ])
    for field in required_fields:
        if stage.get(field) is not True:
            raise RuntimeError("selected frontend stage does not satisfy %s" % field)
    if stage.get("test_semantic_accessed") is not False:
        raise RuntimeError("formal cache stage has Test semantic access")
    normalized = stage.get("normalized_outputs") or {}
    physical = normalized.get("physical_stream")
    if not physical:
        raise RuntimeError("selected frontend stage has no normalized physical stream")
    return {
        "frontend": frontend,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "representation_path": str(representation_path.resolve()),
        "representation_sha256": sha256_file(representation_path),
        "stage_path": str(stage_path.resolve()),
        "stage_sha256": sha256_file(stage_path),
        "default_input": str(Path(str(physical)).resolve()),
        "split": split,
    }


def _validate_track(row: Dict[str, Any], row_number: int) -> Tuple[List[str], List[List[float]], List[float]]:
    if FORBIDDEN_FIELDS & set(row):
        raise ValueError("formal physical row %d contains forbidden semantic fields" % row_number)
    for field in ("sample_key", "video_id", "physical_track_id", "frame_ids", "boxes_xyxy", "image_paths"):
        if field not in row:
            raise ValueError("formal physical row %d misses %s" % (row_number, field))
    frame_ids = [int(value) for value in row["frame_ids"]]
    image_paths = [str(value) for value in row["image_paths"]]
    boxes = [[float(value) for value in box] for box in row["boxes_xyxy"]]
    if not frame_ids or len(frame_ids) != len(image_paths) or len(frame_ids) != len(boxes):
        raise ValueError("formal physical row %d has mismatched lineage arrays" % row_number)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("formal physical row %d has duplicate frame IDs" % row_number)
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("formal physical row %d has non-finite/malformed box" % row_number)
    quality = [float(value) for value in row.get("quality", [1.0] * len(frame_ids))]
    if len(quality) != len(frame_ids) or not all(math.isfinite(value) for value in quality):
        raise ValueError("formal physical row %d has invalid quality" % row_number)
    return image_paths, boxes, quality


def _count_tracks(path: Path) -> int:
    count = 0
    for row_number, row in enumerate(_jsonl(path), 1):
        _validate_track(row, row_number)
        count += 1
    if count == 0:
        raise ValueError("formal physical stream is empty: %s" % path)
    return count


def _shard_dir(cache_root: Path, shard_index: int) -> Path:
    return cache_root / "shards" / ("shard-%06d" % shard_index)


def _marker(shard_dir: Path, name: str) -> Path:
    return shard_dir / name


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _recover_or_reject_markers(cache_root: Path) -> None:
    shards = cache_root / "shards"
    if not shards.exists():
        return
    for shard_dir in sorted(shards.glob("shard-*")):
        launched = _marker(shard_dir, ".launched")
        done = _marker(shard_dir, ".done")
        if done.exists():
            # A process can be interrupted after the atomic completion marker
            # is committed but before the launch marker is removed.  The done
            # marker is authoritative, so clean only that stale marker.
            if launched.exists():
                launched.unlink()
            continue
        if not launched.exists():
            continue
        try:
            payload = json.loads(launched.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid active shard marker: %s" % launched) from exc
        if _pid_alive(pid):
            raise RuntimeError("shard is already owned by live PID %d: %s" % (pid, shard_dir))
        launched.unlink()


def _done_payload(shard_dir: Path, source_sha256: str) -> Optional[Dict[str, Any]]:
    done = _marker(shard_dir, ".done")
    if not done.is_file():
        return None
    try:
        payload = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = ("source_sha256", "track_start", "track_count", "observation_count")
    if payload.get("source_sha256") != source_sha256 or not all(key in payload for key in required):
        return None
    for name in ("observations.npy", "prefix_features.npy", "full_features.npy", "index.parquet"):
        if not (shard_dir / name).is_file():
            return None
    return payload


def _claim_shard(shard_dir: Path, payload: Dict[str, Any]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    marker = _marker(shard_dir, ".launched")
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
    except FileExistsError as exc:
        raise RuntimeError("shard marker already exists: %s" % marker) from exc


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, records: List[Dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the formal sharded index") from exc
    temporary = path.with_name(".%s.tmp.%d" % (path.name, os.getpid()))
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(temporary), compression="zstd")
    os.replace(temporary, path)


def _extract_track(
    row: Dict[str, Any],
    model: Any,
    transform: Any,
    torch: Any,
    image_cls: Any,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_paths, boxes, quality = _validate_track(row, -1)
    tensors: List[Any] = []
    embeddings: List[np.ndarray] = []

    def flush() -> None:
        if not tensors:
            return
        batch = torch.cat(tensors, dim=0).to("cuda:0")
        with torch.no_grad():
            output_features = model.forward_features(batch)
            values = torch.nn.functional.normalize(output_features["x_norm_clstoken"], dim=-1)
        embeddings.extend(values.cpu().numpy().astype(np.float32))
        tensors[:] = []

    for image_path, box in zip(image_paths, boxes):
        path = FRAMES_ROOT / image_path
        with image_cls.open(path) as raw:
            crop = crop_box(raw.convert("RGB"), box)
        if min(crop.size) < 4:
            raise ValueError("degenerate crop for %s" % row.get("sample_key"))
        tensors.append(transform(crop).unsqueeze(0))
        if len(tensors) >= batch_size:
            flush()
    flush()
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (len(image_paths), 768) or not np.isfinite(values).all():
        raise ValueError("unexpected/non-finite DINOv2 feature shape for %s" % row.get("sample_key"))
    prefix = np.stack([
        np.asarray(normalized_mean(values, quality, min(size, len(values))), dtype=np.float32)
        for size in PREFIXES
    ], axis=0)
    full = np.asarray(normalized_mean(values, quality, len(values)), dtype=np.float32)
    return values.astype(np.float16), prefix.astype(np.float16), full.astype(np.float16)


def _extract_tracks_batched(
    rows: Sequence[Dict[str, Any]],
    model: Any,
    transform: Any,
    torch: Any,
    image_cls: Any,
    batch_size: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract a shard while filling batches across track boundaries.

    The old implementation flushed inside each track.  Because the selected
    frontend has only a few observations per track, that left most GPU
    batches nearly empty.  This keeps per-track causal aggregation intact but
    shares each inference batch across adjacent tracks in the same shard.
    """

    validated: List[Tuple[List[str], List[List[float]], List[float]]] = []
    track_embeddings: List[List[Optional[np.ndarray]]] = []
    pending_tensors: List[Any] = []
    pending_refs: List[Tuple[int, int]] = []

    def flush() -> None:
        if not pending_tensors:
            return
        batch = torch.cat(pending_tensors, dim=0).to("cuda:0")
        with torch.no_grad():
            output_features = model.forward_features(batch)
            values = torch.nn.functional.normalize(output_features["x_norm_clstoken"], dim=-1)
        cpu_values = values.cpu().numpy().astype(np.float32)
        for value, (track_index, observation_index) in zip(cpu_values, pending_refs):
            track_embeddings[track_index][observation_index] = value
        pending_tensors.clear()
        pending_refs.clear()

    for track_index, row in enumerate(rows):
        image_paths, boxes, quality = _validate_track(row, -1)
        validated.append((image_paths, boxes, quality))
        track_embeddings.append([None] * len(image_paths))
        for observation_index, (image_path, box) in enumerate(zip(image_paths, boxes)):
            path = FRAMES_ROOT / image_path
            with image_cls.open(path) as raw:
                crop = crop_box(raw.convert("RGB"), box)
            if min(crop.size) < 4:
                raise ValueError("degenerate crop for %s" % row.get("sample_key"))
            pending_tensors.append(transform(crop).unsqueeze(0))
            pending_refs.append((track_index, observation_index))
            if len(pending_tensors) >= batch_size:
                flush()
    flush()

    outputs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for track_index, (_image_paths, _boxes, quality) in enumerate(validated):
        values_list = track_embeddings[track_index]
        if any(value is None for value in values_list):
            raise RuntimeError("batched feature extraction left an observation unfilled")
        values = np.asarray(values_list, dtype=np.float32)
        if values.shape != (len(values_list), 768) or not np.isfinite(values).all():
            raise ValueError("unexpected/non-finite batched DINOv2 feature")
        prefix = np.stack([
            np.asarray(normalized_mean(values, quality, min(size, len(values))), dtype=np.float32)
            for size in PREFIXES
        ], axis=0)
        full = np.asarray(normalized_mean(values, quality, len(values)), dtype=np.float32)
        outputs.append((values.astype(np.float16), prefix.astype(np.float16), full.astype(np.float16)))
    return outputs


def _write_shard(
    cache_root: Path,
    shard_index: int,
    source_sha256: str,
    track_start: int,
    records: List[Dict[str, Any]],
    observation_features: List[np.ndarray],
    prefix_features: List[np.ndarray],
    full_features: List[np.ndarray],
) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot write an empty shard")
    shard_dir = _shard_dir(cache_root, shard_index)
    _claim_shard(shard_dir, {
        "pid": os.getpid(),
        "started_utc": utc_now(),
        "source_sha256": source_sha256,
        "track_start": track_start,
        "track_count": len(records),
    })
    try:
        observations = np.concatenate(observation_features, axis=0).astype(np.float16)
        prefixes = np.stack(prefix_features, axis=0).astype(np.float16)
        full = np.stack(full_features, axis=0).astype(np.float16)
        if prefixes.shape != (len(records), len(PREFIXES), 768) or full.shape != (len(records), 768):
            raise ValueError("invalid shard aggregate shape")
        _atomic_npy(shard_dir / "observations.npy", observations)
        _atomic_npy(shard_dir / "prefix_features.npy", prefixes)
        _atomic_npy(shard_dir / "full_features.npy", full)
        _atomic_parquet(shard_dir / "index.parquet", records)
        payload = {
            "schema_version": "trackocd.v2.formal_feature_shard.v1",
            "source_sha256": source_sha256,
            "track_start": int(track_start),
            "track_count": len(records),
            "observation_count": int(observations.shape[0]),
            "prefixes": list(PREFIXES),
            "format": "npy_plus_parquet_index",
            "finished_utc": utc_now(),
        }
        atomic_json(_marker(shard_dir, ".done"), payload)
        _marker(shard_dir, ".launched").unlink()
        return payload
    except Exception:
        # Temporary files remain recoverable evidence.  The next invocation
        # can reclaim this shard after the current PID is gone.
        raise


def _manifest_payload(
    *,
    cache_root: Path,
    auth: Dict[str, Any],
    input_path: Path,
    input_sha256: str,
    shard_tracks: int,
    status: str,
    total_tracks: int,
    completed_shards: List[Dict[str, Any]],
    selected_gpu: Optional[int],
    split: str,
) -> Dict[str, Any]:
    completed_shards = sorted(completed_shards, key=lambda item: int(item["track_start"]))
    return {
        "schema_version": "trackocd.v2.formal_feature_cache.v1",
        "status": status,
        "generated_utc": utc_now(),
        "split": split,
        "frontend": auth["frontend"],
        "input": {"path": str(input_path.resolve()), "sha256": input_sha256, "format": "grouped_jsonl"},
        "cache_root": str(cache_root.resolve()),
        "format": "sharded_numpy_with_parquet_index",
        "prefixes": list(PREFIXES),
        "dimension": 768,
        "shard_tracks": int(shard_tracks),
        "total_tracks": int(total_tracks),
        "total_observations": int(sum(int(item["observation_count"]) for item in completed_shards)),
        "completed_shards": completed_shards,
        "shard_count": len(completed_shards),
        "no_per_track_json": True,
        "encoder": {
            "name": "DINOv2 ViT-B/14",
            "aggregation": "quality-weighted causal mean then L2 normalization",
            "text_or_category_logits": False,
            "category_id_feature": False,
            "physical_id_feature": False,
            "future_observations_used": False,
        },
        "authorization": auth,
        "selected_gpu_index": selected_gpu,
        "test_semantic_accessed": False,
        "test_evaluation_unlocked": split == "test",
    }


def build(
    *,
    slug: str,
    input_path: Path,
    cache_root: Path,
    model_repo: Path,
    batch_size: int,
    shard_tracks: int,
    gpu_index: Optional[int],
    preflight_only: bool,
    split: str = "val",
) -> Dict[str, Any]:
    auth = _authorize(slug, split=split)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() != Path(auth["default_input"]).resolve():
        raise RuntimeError("formal input must be the selected frontend's normalized physical stream")
    if not model_repo.is_dir():
        raise FileNotFoundError(model_repo)
    input_sha256 = sha256_file(input_path)
    total_tracks = _count_tracks(input_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "shards").mkdir(parents=True, exist_ok=True)
    _recover_or_reject_markers(cache_root)
    completed: List[Dict[str, Any]] = []
    completed_starts: Set[int] = set()
    shard_count = (total_tracks + shard_tracks - 1) // shard_tracks
    for shard_index in range(shard_count):
        done = _done_payload(_shard_dir(cache_root, shard_index), input_sha256)
        expected_start = shard_index * shard_tracks
        expected_count = min(shard_tracks, total_tracks - expected_start)
        if done is not None and int(done.get("track_start", -1)) == expected_start and int(done.get("track_count", -1)) == expected_count:
            completed.append(done)
            completed_starts.add(int(done["track_start"]))

    if len(completed) == shard_count:
        payload = _manifest_payload(
            cache_root=cache_root,
            auth=auth,
            input_path=input_path,
            input_sha256=input_sha256,
            shard_tracks=shard_tracks,
            status="COMPLETE",
            total_tracks=total_tracks,
            completed_shards=completed,
            selected_gpu=gpu_index,
            split=split,
        )
        atomic_json(cache_root / "cache_manifest.json", payload)
        return payload

    snapshot = resource_snapshot()
    eligible = eligible_gpus(snapshot, 1)
    if gpu_index is not None:
        if gpu_index not in eligible:
            raise RuntimeError("requested GPU %d is not idle within the RAM safety margin" % gpu_index)
        selected_gpu = gpu_index
    elif not eligible:
        wait = _manifest_payload(
            cache_root=cache_root,
            auth=auth,
            input_path=input_path,
            input_sha256=input_sha256,
            shard_tracks=shard_tracks,
            status="WAITING_RESOURCE",
            total_tracks=total_tracks,
            completed_shards=completed,
            selected_gpu=None,
            split=split,
        )
        wait["resource_snapshot"] = snapshot
        wait["reason"] = "no idle GPU with >=25% RAM floor and per-worker margin"
        atomic_json(cache_root / "cache_manifest.json", wait)
        if preflight_only:
            return wait
        raise RuntimeError("FORMAL_CACHE_WAITING_RESOURCE: no eligible GPU")
    else:
        selected_gpu = int(eligible[0])

    preflight = _manifest_payload(
        cache_root=cache_root,
        auth=auth,
        input_path=input_path,
        input_sha256=input_sha256,
        shard_tracks=shard_tracks,
        status="RUNNING",
        total_tracks=total_tracks,
        completed_shards=completed,
        selected_gpu=selected_gpu,
        split=split,
    )
    preflight["resource_snapshot"] = snapshot
    preflight["worker_ram_gib_estimate"] = WORKER_RAM_GIB
    atomic_json(cache_root / "cache_manifest.json", preflight)
    if preflight_only:
        return preflight

    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("formal DINOv2 runtime import failed") from exc
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in formal feature worker")
    model = torch.hub.load(str(model_repo), "dinov2_vitb14", source="local").eval().to("cuda:0")
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def commit_rows(shard_index: int, items: List[Tuple[int, Dict[str, Any]]]) -> None:
        if not items:
            return
        rows = [row for _track_index, row in items]
        extracted = _extract_tracks_batched(rows, model, transform, torch, Image, batch_size)
        records: List[Dict[str, Any]] = []
        observation_features: List[np.ndarray] = []
        prefix_features: List[np.ndarray] = []
        full_features: List[np.ndarray] = []
        observation_offset = 0
        for (track_index, row), (frame_embeddings, track_prefix, track_full) in zip(items, extracted):
            frame_ids = [int(value) for value in row["frame_ids"]]
            image_paths = [str(value) for value in row["image_paths"]]
            boxes = [[float(value) for value in box] for box in row["boxes_xyxy"]]
            quality = [float(value) for value in row.get("quality", [1.0] * len(frame_ids))]
            records.append({
                "sample_key": str(row["sample_key"]),
                "source_split": str(row.get("source_split", "val_predicted")),
                "video_id": int(row["video_id"]),
                "physical_track_id": str(row["physical_track_id"]),
                "stream_order": int(row.get("stream_order", track_index)),
                "global_track_index": int(track_index),
                "observation_offset": int(observation_offset),
                "observation_count": len(frame_ids),
                "frame_ids": frame_ids,
                "image_paths": image_paths,
                "boxes_xyxy": boxes,
                "quality": quality,
                "prefix_observations_used": [min(prefix, len(frame_ids)) for prefix in PREFIXES],
            })
            observation_features.append(frame_embeddings)
            prefix_features.append(track_prefix)
            full_features.append(track_full)
            observation_offset += int(frame_embeddings.shape[0])
        done = _write_shard(
            cache_root,
            shard_index,
            input_sha256,
            shard_index * shard_tracks,
            records,
            observation_features,
            prefix_features,
            full_features,
        )
        completed.append(done)
        atomic_json(cache_root / "cache_manifest.json", _manifest_payload(
            cache_root=cache_root,
            auth=auth,
            input_path=input_path,
            input_sha256=input_sha256,
            shard_tracks=shard_tracks,
            status="RUNNING",
            total_tracks=total_tracks,
            completed_shards=completed,
            selected_gpu=selected_gpu,
            split=split,
        ))

    current_shard = -1
    current_items: List[Tuple[int, Dict[str, Any]]] = []
    for track_index, row in enumerate(_jsonl(input_path)):
        shard_index = track_index // shard_tracks
        track_start = shard_index * shard_tracks
        if track_start in completed_starts:
            continue
        if current_shard < 0:
            current_shard = shard_index
        if shard_index != current_shard:
            commit_rows(current_shard, current_items)
            current_items = []
            current_shard = shard_index
        current_items.append((track_index, row))
    if current_items:
        commit_rows(current_shard, current_items)
    payload = _manifest_payload(
        cache_root=cache_root,
        auth=auth,
        input_path=input_path,
        input_sha256=input_sha256,
        shard_tracks=shard_tracks,
        status="COMPLETE",
        total_tracks=total_tracks,
            completed_shards=completed,
            selected_gpu=selected_gpu,
            split=split,
        )
    atomic_json(cache_root / "cache_manifest.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(FRONTENDS), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--model-repo", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--shard-tracks", type=int, default=DEFAULT_SHARD_TRACKS)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or args.batch > 64:
        raise SystemExit("--batch must be between 1 and 64")
    if args.shard_tracks < 1:
        raise SystemExit("--shard-tracks must be positive")
    ensure_output_layout()
    try:
        auth = _authorize(args.frontend, split=args.split)
    except Exception as exc:
        print(
            json.dumps({
                "schema_version": "trackocd.v2.formal_feature_cache.v1",
                "status": "FORMAL_CACHE_NOT_AUTHORIZED",
                "frontend": FRONTENDS[args.frontend],
                "split": args.split,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "test_semantic_accessed": False,
            }, indent=2, sort_keys=True),
        )
        return 3
    input_path = (args.input or Path(auth["default_input"])).resolve()
    model_repo = (args.model_repo or next((path for path in DEFAULT_HUBS if path.is_dir()), DEFAULT_HUBS[0])).resolve()
    default_cache_root = FORMAL_ROOT / args.frontend / ("test" if args.split == "test" else "")
    cache_root = (args.output_root or default_cache_root).resolve()
    try:
        result = build(
            slug=args.frontend,
            input_path=input_path,
            cache_root=cache_root,
            model_repo=model_repo,
            batch_size=args.batch,
            shard_tracks=args.shard_tracks,
            gpu_index=args.gpu_index,
            preflight_only=args.preflight_only,
            split=args.split,
        )
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.formal_feature_cache.v1",
            "status": "FAILED_FORMAL_FEATURE_CACHE",
            "generated_utc": utc_now(),
            "frontend": FRONTENDS[args.frontend],
            "split": args.split,
            "input": str(input_path),
            "error": "%s: %s" % (type(exc).__name__, exc),
            "test_semantic_accessed": False,
        }
        atomic_json(cache_root / "cache_failure.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2 if "WAITING_RESOURCE" in str(exc) else 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("status") == "WAITING_RESOURCE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
