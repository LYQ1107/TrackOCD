#!/usr/bin/env python3
"""At most 64 preregistered Train Known GT tracks; frozen descriptor inference.

One worker, a freshly idle GPU, batch four, streaming crops, bounded wall time.
Reuse exact previous smoke observations where identities/protocols match.
No optimizer, new weight, raw crop file, Val/Test or predicted/frontend claim.
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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.audit_assets import DATASET, file_record
from scripts.trackocd_core.smoke_gt_features import choose_idle_gpu
from scripts.trackocd_v2.build_common_features import crop_box
from src.trackocd_core.features import CompactGTFeasibilityCache, PREFIXES, prefix_view
from src.trackocd_v2.io import atomic_json, sha256_file


def main() -> int:
    started = time.monotonic()
    pilot_path = ROOT / "configs/trackocd_core/gt_feasibility_pilot.json"
    pilot = json.loads(pilot_path.read_text())
    plan_path = ROOT / "outputs/trackocd_core/pilots/gt_train_known/selection_plan.json"
    prereg = json.loads((ROOT / "outputs/trackocd_core/audit/gt_pilot_preregistration.json").read_text())
    if sha256_file(plan_path) != prereg["private_plan"]["sha256"]:
        raise ValueError("Preregistered selection plan changed")
    plan = json.loads(plan_path.read_text())
    for name in ("config", "roles", "annotation", "visual_config"):
        if sha256_file(Path(plan[name]["path"])) != plan[name]["sha256"]:
            raise ValueError("Registered pilot input changed: " + name)
    known = set(json.loads(Path(plan["roles"]["path"]).read_text())["known_ids"])
    selected = plan["rows"]
    if (len(selected) > pilot["maximum_tracks"] or any(r["category"] not in known for r in selected)
            or sum(len(r["observations"]) for r in selected) > pilot["maximum_observations"]):
        raise ValueError("Pilot size/Train Known allowlist violated")
    config_path = Path(plan["visual_config"]["path"])
    config = json.loads(config_path.read_text())
    cache = ROOT / pilot["cache_directory"]
    if cache.exists():
        raise RuntimeError("Pilot cache exists; validate/reuse rather than overwrite")

    def headroom():
        mem = {k: int(v.split()[0]) for k, v in (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}
        if mem["MemAvailable"] * 1024 - pilot["ram_plan_bytes"] < mem["MemTotal"] * 1024 * pilot["minimum_system_ram_headroom"]:
            raise RuntimeError("Resource wait: registered RAM plan crosses 25% headroom")
        if time.monotonic() - started > pilot["maximum_extraction_wall_seconds"]:
            raise RuntimeError("Bounded pilot exceeded registered wall-time budget")
        return mem

    initial_mem = headroom()
    if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < pilot["maximum_new_payload_bytes"] + 2**30:
        raise RuntimeError("Resource wait: insufficient disk headroom")
    repo, checkpoint = ROOT / config["local_repository"], ROOT / config["checkpoint"]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["upstream_commit"] or subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Pinned DINOv2 source identity differs")
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("Frozen DINO checkpoint differs")
    uuid = choose_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from torchvision import transforms

    torch.set_num_threads(1)
    torch.manual_seed(1027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitb14
    model = dinov2_vitb14(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to("cuda:0")
    del state
    transform = transforms.Compose([transforms.Resize(tuple(config["resize"]), interpolation=Image.Resampling.BILINEAR),
                                    transforms.ToTensor(), transforms.Normalize(config["normalization_mean"], config["normalization_std"])])
    smoke_root = ROOT / "outputs/trackocd_core/features/gt_train_known_smoke"
    old = CompactGTFeasibilityCache(smoke_root)
    if old.manifest["config"]["sha256"] != plan["visual_config"]["sha256"] or old.manifest["upstream_commit"] != commit:
        raise ValueError("Previous smoke feature lineage is incompatible")
    old_rows = {r["key"]: r for r in pq.read_table(smoke_root / "index.parquet").to_pylist()}
    old_visual = np.load(smoke_root / "observations.npy", mmap_mode="r", allow_pickle=False)
    old_geometry = np.load(smoke_root / "geometry.npy", mmap_mode="r", allow_pickle=False)
    features, geometry, rows, labels = [], [], [], []
    inferred = reused = 0
    for row in selected:
        headroom()
        observations = row["observations"]
        key = f"train_gt_{row['video_id']}_{row['gt_track_id']}"
        offset = len(features)
        previous = old_rows.get(key)
        identities = {"frame_ids": [ob["frame_id"] for ob in observations], "image_ids": [ob["image_id"] for ob in observations],
                      "image_paths": [ob["image_path"] for ob in observations], "boxes_xyxy": [ob["bbox_xyxy"] for ob in observations]}
        can_reuse = previous is not None and previous["observation_count"] == len(observations) and all(previous[k] == v for k, v in identities.items())
        if can_reuse:
            begin = previous["observation_offset"]
            features.extend(np.array(old_visual[begin:begin + len(observations)], copy=True))
            geometry.extend(np.array(old_geometry[begin:begin + len(observations)], copy=True))
            reused += len(observations)
        else:
            for start in range(0, len(observations), pilot["batch_size"]):
                headroom()
                tensors = []
                for ob in observations[start:start + pilot["batch_size"]]:
                    with Image.open(DATASET / "frames" / ob["image_path"]) as raw:
                        width, height = raw.size
                        crop = crop_box(raw.convert("RGB"), ob["bbox_xyxy"], config["crop_context_fraction_per_side"])
                        tensors.append(transform(crop))
                    geometry.append(np.clip(np.asarray(ob["bbox_xyxy"]) / [width, height, width, height], 0, 1))
                with torch.inference_mode():
                    output = model.forward_features(torch.stack(tensors).to("cuda:0"))["x_norm_clstoken"]
                    features.extend(torch.nn.functional.normalize(output, dim=-1).cpu().numpy().astype(np.float16))
                inferred += len(tensors)
        rows.append({"key": key, "video_id": row["video_id"], "physical_track_id": row["gt_track_id"],
                     "source_split": "train_gt_feasibility", "observation_offset": offset,
                     "observation_count": len(observations), **identities, "quality": [1.] * len(observations)})
        labels.append({"key": key, "category_id": row["category"], "role": "Train Known only",
                       "partition": row["partition"], "purpose": row["purpose"], "simulation_role": row["simulation_role"]})
        print(json.dumps({"completed_tracks": len(rows), "maximum_tracks": len(selected),
                          "inferred_observations": inferred, "reused_observations": reused}), flush=True)
    visual, geometry = np.asarray(features, dtype=np.float16), np.asarray(geometry, dtype=np.float32)
    if visual.shape != (plan["summary"]["observations"], 768) or not np.isfinite(visual).all():
        raise ValueError("Pilot descriptor shape/value mismatch")
    prefixes = []
    for row in rows:
        begin, count = row["observation_offset"], row["observation_count"]
        prefixes.append([prefix_view(visual[begin:begin + count], geometry[begin:begin + count], row["quality"], row["frame_ids"], p).weighted_mean() for p in PREFIXES])
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".gt-pilot-", dir=cache.parent))
    np.save(partial / "observations.npy", visual, allow_pickle=False)
    np.save(partial / "geometry.npy", geometry, allow_pickle=False)
    np.save(partial / "prefix_features.npy", np.asarray(prefixes, dtype=np.float16), allow_pickle=False)
    pq.write_table(pa.Table.from_pylist(rows), partial / "index.parquet")
    pq.write_table(pa.Table.from_pylist(labels), partial / "train_labels.parquet")
    payloads = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in partial.iterdir()}
    payload_bytes = sum(p["bytes"] for p in payloads.values())
    if payload_bytes > pilot["maximum_new_payload_bytes"]:
        raise ValueError("Compact pilot payload exceeded registered budget")
    manifest = {"schema_version": "trackocd.core.gt-feature-pilot.v1", "status": "PASS_SMALL_GT_FEASIBILITY_INPUT_ONLY",
                "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_role": old.manifest["source_role"],
                "config": file_record(config_path), "pilot_config": file_record(pilot_path), "selection_plan_sha256": sha256_file(plan_path),
                "preregistration": file_record(ROOT / "outputs/trackocd_core/audit/gt_pilot_preregistration.json"),
                "upstream_commit": commit, "checkpoint": file_record(checkpoint), "annotation": plan["annotation"], "roles": plan["roles"],
                "selection": plan["summary"], "tracks": len(rows), "observations": len(visual), "feature_dimension": 768,
                "prefixes": list(PREFIXES), "prefix_features_shape": [len(rows), 5, 768],
                "payloads": payloads, "new_feature_payload_bytes": payload_bytes,
                "new_inferred_observations": inferred, "reused_exact_smoke_observations": reused,
                "source_sha256": {name: sha256_file(ROOT / name) for name in
                                  ("scripts/trackocd_core/extract_gt_pilot.py", "src/trackocd_core/features.py")},
                "all_visual_encoder_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
                "labels_separated_from_model_view": True, "full_track_mean_written": False,
                "GT_physical_coverage_is_not_predicted_coverage": True, "unit_quality_not_detector_confidence": True,
                "causality_claim": config["causality_claim"], "training_started": False, "optimizer_used": False,
                "val_or_test_accessed": False, "formal_predicted_result": False, "external_process_interference": False,
                "resources": {"wall_seconds": time.monotonic() - started, "worker_count": 1, "gpu_uuid": uuid,
                              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              "initial_mem_available_kib": initial_mem["MemAvailable"], "ram_plan_bytes": pilot["ram_plan_bytes"]}}
    atomic_json(partial / "manifest.json", manifest)
    atomic_json(partial / ".done", {"status": "COMPLETE_SMALL_GT_PILOT", "config_sha256": manifest["config"]["sha256"]})
    partial.rename(cache)
    # Independently check completed payload lineage before publishing the receipt.
    verified = CompactGTFeasibilityCache(cache)
    for key in verified.keys():
        for p in PREFIXES:
            if not np.isfinite(verified.get_prefix(key, p).weighted_mean()).all():
                raise ValueError("Invalid completed pilot prefix")
    atomic_json(ROOT / "outputs/trackocd_core/audit/gt_pilot_features.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "tracks", "observations", "new_feature_payload_bytes", "new_inferred_observations", "reused_exact_smoke_observations", "resources")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
