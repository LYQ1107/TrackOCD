#!/usr/bin/env python3
"""Small frozen-encoder Train Known GT feasibility smoke, NOT training/main MOT.

At most four GT tracks / 64 observations; one worker, one freshly idle GPU.
Reuse the existing checkpoint; no hub downloads, optimizer, detector, Val,
Test, per-track JSON, crop files or full-track mean are created.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.audit_assets import DATASET, file_record, read_annotation
from scripts.trackocd_v2.build_common_features import crop_box
from src.trackocd_core.features import PREFIXES, prefix_view
from src.trackocd_v2.io import atomic_json, sha256_file


def select_train_known_tracks(annotation: dict, known_ids: set[int]) -> list[dict]:
    images = {int(im["id"]): im for im in annotation["images"]}
    grouped = {}
    for ann in annotation["annotations"]:
        category = int(ann["category_id"])
        if category not in known_ids:
            continue
        key = int(ann["video_id"]), int(ann["track_id"])
        row = grouped.setdefault(key, {"video_id": key[0], "gt_track_id": key[1],
                                      "category": category, "observations": []})
        if row["category"] != category:
            raise ValueError("Inconsistent Train GT track label")
        row["observations"].append({"annotation": ann, "image": images[int(ann["image_id"])]})
    by_category = defaultdict(lambda: defaultdict(list))
    for key, row in sorted(grouped.items()):
        if len(row["observations"]) >= 16:
            row["observations"].sort(key=lambda r: (int(r["image"]["frame_index"]), int(r["image"]["id"])))
            by_category[row["category"]][row["video_id"]].append(row)
    categories = [c for c in sorted(by_category) if len(by_category[c]) >= 2][:2]
    if len(categories) != 2:
        raise RuntimeError("Train Known cross-video smoke support is insufficient")
    result = []
    for category in categories:
        for video in sorted(by_category[category])[:2]:
            row = by_category[category][video][0]
            result.append(dict(row, observations=row["observations"][:16]))
    return result


def choose_idle_gpu() -> str:
    def query(option):
        return subprocess.check_output(["nvidia-smi", option, "--format=csv,noheader,nounits"], text=True, timeout=10)
    occupied = {line.strip() for line in query("--query-compute-apps=gpu_uuid").splitlines() if line.strip()}
    for line in query("--query-gpu=uuid,memory.used,memory.free,utilization.gpu").splitlines():
        uuid, used, free, utilization = [part.strip() for part in line.split(",")]
        if uuid not in occupied and int(used) < 100 and int(free) >= 4096 and int(utilization) == 0:
            return uuid
    raise RuntimeError("Resource wait: no freshly idle GPU; no foreign process touched")


def main() -> int:
    started = time.monotonic()
    config_path = ROOT / "configs/trackocd_core/common_features.json"
    config = json.loads(config_path.read_text())
    cache = ROOT / "outputs/trackocd_core/features/gt_train_known_smoke"
    if cache.exists():
        raise RuntimeError("Smoke cache already exists: preserve and validate it, do not overwrite")
    mem = {k: int(v.split()[0]) for k, v in (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}
    if mem["MemAvailable"] - 4 * 2**20 < mem["MemTotal"] * .25:
        raise RuntimeError("Resource wait: 4 GiB RAM plan would cross 25% headroom")
    repo = ROOT / config["local_repository"]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["upstream_commit"]:
        raise ValueError("Wrong pinned DINOv2 source commit")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pinned DINOv2 source is dirty")
    checkpoint = ROOT / config["checkpoint"]
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("DINOv2 checkpoint identity mismatch")
    roles_path = ROOT / "configs/trackocd_core/roles.json"
    known = set(json.loads(roles_path.read_text())["known_ids"])
    annotation_path = DATASET / "annotations/train.json"
    selected = select_train_known_tracks(read_annotation(annotation_path, "train"), known)
    uuid = choose_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from torchvision import transforms

    sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitb14

    torch.set_num_threads(1)
    torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = dinov2_vitb14(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to("cuda:0")
    del state
    transform = transforms.Compose([
        transforms.Resize(tuple(config["resize"]), interpolation=Image.Resampling.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(config["normalization_mean"], config["normalization_std"]),
    ])
    tensors, geometry, rows, labels = [], [], [], []
    for row in selected:
        start = len(tensors)
        frame_ids, image_ids, image_paths, boxes = [], [], [], []
        for ob in row["observations"]:
            ann, image = ob["annotation"], ob["image"]
            x, y, w, h = map(float, ann["bbox"])
            box = [x, y, x + w, y + h]
            path = DATASET / "frames" / image["file_name"]
            with Image.open(path) as raw:
                width, height = raw.size
                crop = crop_box(raw.convert("RGB"), box, config["crop_context_fraction_per_side"])
                tensors.append(transform(crop))
            geometry.append(np.clip(np.asarray(box) / [width, height, width, height], 0, 1))
            boxes.append(box)
            frame_ids.append(int(image["frame_index"]))
            image_ids.append(int(image["id"]))
            image_paths.append(image["file_name"])
        key = f"train_gt_{row['video_id']}_{row['gt_track_id']}"
        rows.append({"key": key, "video_id": row["video_id"], "physical_track_id": row["gt_track_id"],
                     "source_split": "train_gt_feasibility", "observation_offset": start,
                     "observation_count": 16, "frame_ids": frame_ids, "image_ids": image_ids,
                     "image_paths": image_paths, "boxes_xyxy": boxes, "quality": [1.] * 16})
        labels.append({"key": key, "category_id": row["category"], "role": "Train Known only"})
    features = []
    with torch.inference_mode():
        for start in range(0, len(tensors), config["gt_smoke_batch_size"]):
            batch = torch.stack(tensors[start:start + config["gt_smoke_batch_size"]]).to("cuda:0")
            output = model.forward_features(batch)["x_norm_clstoken"]
            features.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy())
        # Same first crop with and without later crops in its batch.
        first = model.forward_features(tensors[0][None].to("cuda:0"))["x_norm_clstoken"]
        first = torch.nn.functional.normalize(first, dim=-1).cpu().numpy()[0]
    visual = np.concatenate(features).astype(np.float16)
    if visual.shape != (64, 768) or not np.isfinite(visual).all():
        raise ValueError("Unexpected DINOv2 smoke descriptors")
    singleton_delta = float(np.max(np.abs(first - features[0][0])))
    if singleton_delta > 1e-5:
        raise ValueError("Single-image feature changed with unrelated batch crops")
    prefixes = []
    geometry = np.asarray(geometry, dtype=np.float32)
    for row in rows:
        begin = row["observation_offset"]
        observations = visual[begin:begin + 16]
        boxes = geometry[begin:begin + 16]
        prefixes.append([prefix_view(observations, boxes, row["quality"], row["frame_ids"], p).weighted_mean()
                         for p in PREFIXES])
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".gt-smoke-", dir=cache.parent))
    np.save(partial / "observations.npy", visual, allow_pickle=False)
    np.save(partial / "prefix_features.npy", np.asarray(prefixes, dtype=np.float16), allow_pickle=False)
    np.save(partial / "geometry.npy", geometry, allow_pickle=False)
    pq.write_table(pa.Table.from_pylist(rows), partial / "index.parquet")
    pq.write_table(pa.Table.from_pylist(labels), partial / "train_labels.parquet")
    payloads = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in partial.iterdir()}
    manifest = {
        "schema_version": "trackocd.core.gt-feature-smoke.v1", "status": "PASS_ENGINEERING_GT_ONLY",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_role": "Train Known GT feasibility, not predicted frontend/main metrics",
        "config": file_record(config_path), "upstream_commit": commit,
        "upstream_source_bytes": sum(p.stat().st_size for p in repo.rglob('*') if p.is_file()),
        "checkpoint": file_record(checkpoint), "annotation": file_record(annotation_path), "roles": file_record(roles_path),
        "tracks": len(rows), "observations": len(visual), "feature_dimension": 768,
        "prefixes": list(PREFIXES), "prefix_features_shape": [4, 5, 768],
        "train_known_categories": sorted({r["category_id"] for r in labels}),
        "all_selected_labels_in_inherited_known": all(r["category_id"] in known for r in labels),
        "all_visual_encoder_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
        "visual_model_training_mode": model.training,
        "strict_state_load_pass": True, "parameters": sum(p.numel() for p in model.parameters()),
        "singleton_vs_batch_max_abs_delta_fp32": singleton_delta,
        "stored_descriptor_max_unit_norm_error": float(np.max(np.abs(np.linalg.norm(visual.astype(np.float32), axis=1) - 1))),
        "payloads": payloads, "new_feature_payload_bytes": sum(p["bytes"] for p in payloads.values()),
        "labels_separated_from_model_view": True, "full_track_mean_written": False,
        "causality_claim": config["causality_claim"], "frame_online_decisions_tested": False,
        "old_nas_feature_byte_compatibility_verified": False,
        "training_started": False, "optimizer_used": False, "formal_cache_started": False,
        "val_data_accessed": False, "novel_labels_for_training": False, "test_data_accessed": False,
        "external_process_interference": False, "duplicate_checkpoint_downloaded": False,
        "resources": {"worker_count": 1, "gpu_uuid": uuid, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "initial_mem_available_kib": mem["MemAvailable"], "ram_plan_bytes": 4 * 2**30,
                      "wall_seconds": time.monotonic() - started},
    }
    atomic_json(partial / "manifest.json", manifest)
    atomic_json(partial / ".done", {"status": "COMPLETE_SMALL_GT_SMOKE", "config_sha256": manifest["config"]["sha256"]})
    partial.rename(cache)
    atomic_json(ROOT / "outputs/trackocd_core/audit/gt_feature_smoke.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "tracks", "observations", "new_feature_payload_bytes", "resources")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
