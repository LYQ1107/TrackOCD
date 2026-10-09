#!/usr/bin/env python3
"""At most eight real predicted-box DINO crops, not training/formal M2 cache."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import completed_video, val_image
from src.trackocd_core.features import prefix_view
from scripts.trackocd_v2.build_common_features import crop_box

PUBLIC = ROOT / "configs/trackocd_core/masa_predicted_feature_smoke.json"
RUN = ROOT / "outputs/trackocd_core/features/masa_predicted_interface_smoke"
SUMMARY = ROOT / "outputs/trackocd_core/audit/masa_predicted_feature_smoke.json"


def select_observed_prefixes(video, arrays, tracks=4, frames=8, observations=2):
    """Select at first frame, not by later length, label, score or coverage."""
    if not 1 <= tracks <= 4 or not 1 <= frames <= 8 or not 1 <= observations <= 2:
        raise ValueError("Outside tiny registered smoke bounds")
    ids = sorted(int(i) for i in arrays["track_id"][:int(arrays["frame_offsets"][1])])[:tracks]
    if not ids:
        raise ValueError("Empty first image; no replacement frame/video")
    grouped, scanned = {i: [] for i in ids}, 0
    for index, image in enumerate(video["images"][:frames]):
        begin, end = map(int, arrays["frame_offsets"][index:index + 2])
        for offset in range(begin, end):
            identity = int(arrays["track_id"][offset])
            if identity in grouped and len(grouped[identity]) < observations:
                grouped[identity].append({"image": image, "box": arrays["boxes"][offset].tolist(),
                                          "quality": float(arrays["score"][offset])})
        scanned += 1
        if all(len(rows) == observations for rows in grouped.values()):
            break
    return [{"track_id": i, "observations": rows} for i, rows in grouped.items()], scanned


def verify_prefixes(visual, geometry, quality, frames, counts):
    """Exercise available prefixes, exposing no routing metadata to models."""
    means, tested, start = [], [], 0
    for count in counts:
        arrays = [a[start:start + count] for a in (visual, geometry, quality, frames)]
        for prefix in (1, 2):
            if prefix > count:
                continue  # Report short tracks; no fabricated prefix2.
            view = prefix_view(*arrays, prefix)
            changed = [a.copy() for a in arrays]
            for a in changed[:3]:
                a[prefix:] = np.nan
            changed[3][prefix:] = -1
            np.testing.assert_array_equal(view.weighted_mean(), prefix_view(*changed, prefix).weighted_mean())
            means.append(view.weighted_mean()); tested.append(prefix)
        start += count
    if start != len(visual):
        raise ValueError("Track offsets do not cover actual observations")
    return np.asarray(means, dtype=np.float32), tested


def memory():
    return {k: int(v.split()[0]) for k, v in
            (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}


def install_input_barrier():
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("Frozen feature smoke prohibits network/child execution")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ("/TAO-Amodal/annotations/" in path or "/recovered_splits/" in path
                    or path.endswith(("/roles.json", "/train_labels.parquet")) or "/frames/test/" in path):
                raise PermissionError("Predicted feature smoke prohibits GT/role/Test input")
    sys.addaudithook(audit)


def preflight(commit):
    if RUN.exists() or SUMMARY.exists():
        raise ValueError("Preserve existing smoke/attempt; no automatic overwrite")
    delivery = json.loads((ROOT / "outputs/trackocd_core/audit/masa_predicted_feature_preregistration_delivery.json").read_text())
    if delivery["commit"] != commit or delivery["remote_verified"] is not True:
        raise ValueError("Missing exact remote-verified preregistration")
    for name in (str(PUBLIC.relative_to(ROOT)), "scripts/trackocd_core/smoke_predicted_features.py"):
        if subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError("Changed smoke source/config since preregistration")
    cfg = json.loads(PUBLIC.read_text())
    if delivery["config_sha256"] != sha256_file(PUBLIC):
        raise ValueError("Preregistration config receipt mismatch")
    for name, expected in cfg["unchanged_source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Inherited encoder/crop interface changed")
    if sha256_file(ROOT / cfg["common_config"]) != cfg["common_config_sha256"]:
        raise ValueError("Common feature parameters changed")
    common = json.loads((ROOT / cfg["common_config"]).read_text())
    repo = ROOT / common["local_repository"]
    if (subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != common["upstream_commit"]
            or subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
            or sha256_file(ROOT / common["checkpoint"]) != common["checkpoint_sha256"]):
        raise ValueError("Pinned DINOv2 source/checkpoint changed")
    limits, mem = cfg["limits"], memory()
    if (mem["MemAvailable"] * 1024 - limits["host_rss_bytes"] < mem["MemTotal"] * 1024 * limits["minimum_ram_headroom_fraction"]
            or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]):
        raise RuntimeError("Resource wait: preserve RAM/disk headroom")
    roots = [ROOT, Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")]
    allocated = sum(int(s.split()[0]) for s in subprocess.check_output(["du", "-s", "-B1", *map(str, roots)], text=True).splitlines())
    if allocated + limits["output_allocated_bytes"] > limits["overall_soft_bytes"]:
        raise RuntimeError("Conservative repository+owned-env exceeds soft storage budget")
    def query(option):
        return subprocess.check_output(["nvidia-smi", option, "--format=csv,noheader,nounits"], text=True, timeout=10)
    occupied = set(query("--query-compute-apps=gpu_uuid").split())
    snapshot = query("--query-gpu=uuid,memory.used,memory.free,utilization.gpu")
    selected = None
    for line in snapshot.splitlines():
        gpu, used, free, utilization = [s.strip() for s in line.split(",")]
        if gpu not in occupied and int(used) < 100 and int(free) >= 5120 and int(utilization) == 0:
            selected = gpu; break
    if not selected:
        raise RuntimeError("Resource wait: no freshly idle GPU; no foreign process touched")
    return cfg, common, selected, {"mem_available_kib": mem["MemAvailable"], "gpu_snapshot": snapshot,
                                  "conservative_repo_and_owned_env_allocated_bytes": allocated}


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def main(commit):
    started = time.monotonic()
    cfg, common, gpu, initial_resources = preflight(commit)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    install_input_barrier()
    RUN.mkdir(parents=True, exist_ok=False)
    receipt = {"schema_version": "trackocd.core.masa_predicted_feature_smoke_result.v1", "status": "RUNNING",
               "preregistration_commit": commit, "preregistration_config_sha256": sha256_file(PUBLIC),
               "primary_freeze_permitted": False, "scientific_pass_permitted": False,
               "training": False, "optimizer_used": False, "formal_cache_started": False,
               "gt_or_role_annotation_access": False, "test_access": False,
               "external_process_interference": False, "new_image_or_checkpoint_download": False,
               "source_role": "Tiny real MASA predicted-box interface smoke, never main/predicted feature cache",
               "scope": cfg["scope"], "resources": initial_resources}
    atomic_json(RUN / "receipt.json", receipt)
    try:
        private, run = ROOT / cfg["private_plan"], ROOT / cfg["prediction_run"]
        if (sha256_file(private) != cfg["private_plan_sha256"]
                or sha256_file(run / "prediction_manifest.json") != cfg["prediction_manifest_sha256"]):
            raise ValueError("Changed sealed input stream or metadata plan")
        video = json.loads(private.read_text())["videos"][0]
        record = completed_video(run, video, cfg["prediction_config_sha256"])
        if record is None:
            raise ValueError("First video not complete; no replacement")
        with np.load(run / "shards" / record["npz_filename"], allow_pickle=False) as arrays:
            rows, scanned = select_observed_prefixes(video, arrays, cfg["maximum_tracks"], cfg["maximum_scanned_images"], cfg["maximum_observations_per_track"])
        import torch
        from PIL import Image
        from torchvision import transforms
        sys.path.insert(0, str(ROOT / common["local_repository"]))
        from dinov2.hub.backbones import dinov2_vitb14
        torch.set_num_threads(1); torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model = dinov2_vitb14(pretrained=False)
        state = torch.load(ROOT / common["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state, strict=True); del state
        model.eval().requires_grad_(False).to("cuda:0")
        initial_state = state_digest(model)
        def guard():
            mem = memory()
            if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > cfg["limits"]["host_rss_bytes"]
                    or torch.cuda.max_memory_reserved() > cfg["limits"]["gpu_reserved_bytes"]
                    or mem["MemAvailable"] < mem["MemTotal"] * cfg["limits"]["minimum_ram_headroom_fraction"]):
                raise RuntimeError("Smoke resource guard; stop only this owned process")
        guard()
        transform = transforms.Compose([transforms.Resize(tuple(common["resize"]), interpolation=Image.Resampling.BILINEAR),
                                        transforms.ToTensor(), transforms.Normalize(common["normalization_mean"], common["normalization_std"])])
        tensors, geometry, quality, frames, counts = [], [], [], [], []
        for row in rows:
            counts.append(len(row["observations"]))
            for ob in row["observations"]:
                image = ob["image"]
                path = val_image(Path(cfg["frames_root"]), image["image_path"])
                if path.stat().st_size != image["image_bytes"] or sha256_file(path) != image["image_sha256"]:
                    raise ValueError("Registered current image changed")
                with Image.open(path) as raw:
                    width, height = raw.size
                    tensors.append(transform(crop_box(raw.convert("RGB"), ob["box"], common["crop_context_fraction_per_side"])))
                geometry.append(np.clip(np.asarray(ob["box"]) / [width, height, width, height], 0, 1))
                quality.append(ob["quality"]); frames.append(image["frame_index"])
        if not 1 <= len(tensors) <= cfg["maximum_observations"]:
            raise ValueError("Outside actual smoke observation bound")
        features = []
        with torch.inference_mode():
            for start in range(0, len(tensors), cfg["batch_size"]):
                output = model.forward_features(torch.stack(tensors[start:start + cfg["batch_size"]]).to("cuda:0"))["x_norm_clstoken"]
                features.append(torch.nn.functional.normalize(output, dim=-1).cpu().numpy()); guard()
            first = model.forward_features(tensors[0][None].to("cuda:0"))["x_norm_clstoken"]
            first = torch.nn.functional.normalize(first, dim=-1).cpu().numpy()[0]
        visual = np.concatenate(features).astype(np.float16)
        delta = float(np.max(np.abs(first - features[0][0])))
        if visual.shape != (len(tensors), 768) or not np.isfinite(visual).all() or delta > 1e-5:
            raise ValueError("Wrong feature shape/finiteness/batch independence")
        geometry, quality, frames = np.asarray(geometry, dtype=np.float32), np.asarray(quality, dtype=np.float32), np.asarray(frames, dtype=np.int64)
        means, prefixes = verify_prefixes(visual, geometry, quality, frames, counts)
        final_state = state_digest(model); guard()
        if final_state != initial_state or model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Encoder not immutable/frozen")
        stored_error = float(np.max(np.abs(np.linalg.norm(visual.astype(np.float32), axis=1) - 1)))
        if stored_error > 1e-3:
            raise ValueError("Invalid stored unit descriptors")
        payload = RUN / "engineering_only.npz"
        with payload.open("xb") as writer:
            np.savez_compressed(writer, visual=visual, geometry=geometry, quality=quality, frame_index=frames,
                                observation_counts=np.asarray(counts), prefix_means=means, prefixes=np.asarray(prefixes))
        receipt.update(status="PASS_TINY_REAL_PREDICTED_INTERFACE_NOT_PRIMARY_QUALIFICATION", video_id=video["video_id"],
                       scanned_images=scanned, tracks=len(rows), observations=len(visual), feature_dimension=768,
                       observation_counts=counts, tested_prefixes=prefixes, input_shard_sha256=record["npz_sha256"],
                       input_metadata_plan_sha256=cfg["private_plan_sha256"], checkpoint_sha256=common["checkpoint_sha256"],
                       encoder_source_commit=common["upstream_commit"], strict_state_load=True, all_encoder_parameters_frozen=True,
                       model_initial_state_sha256=initial_state, model_final_state_sha256=final_state,
                       singleton_vs_batch_max_abs_delta_fp32=delta, stored_descriptor_max_unit_norm_error=stored_error,
                       poisoned_future_prefix_invariance=True, frame_online_decisions_tested=False,
                       model_view_fields=cfg["model_view_fields"], private_payload={"bytes": payload.stat().st_size, "sha256": sha256_file(payload)})
        receipt["resources"].update(gpu_uuid=gpu, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(), worker_count=1)
    except Exception as exc:
        receipt.update(status="FAILED_PRESERVED_TINY_SMOKE_ATTEMPT", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["resources"].update(wall_seconds=time.monotonic() - started,
                                    peak_cpu_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        receipt["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(RUN / "receipt.json", receipt)
    allocated = sum(p.stat().st_blocks * 512 for p in RUN.rglob("*") if p.is_file())
    if allocated > cfg["limits"]["output_allocated_bytes"]:
        receipt.update(status="FAILED_PRESERVED_TINY_SMOKE_ATTEMPT", error="Output allocation exceeds preregistered limit")
        atomic_json(RUN / "receipt.json", receipt)
        raise RuntimeError("Tiny output exceeds limit; preserve, do not publish PASS")
    receipt["resources"]["private_output_allocated_bytes_before_marker"] = allocated
    atomic_json(RUN / "receipt.json", receipt)
    atomic_json(RUN / ".done", {"status": receipt["status"], "preregistration_config_sha256": receipt["preregistration_config_sha256"],
                               "payload_sha256": receipt["private_payload"]["sha256"]})
    atomic_json(SUMMARY, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "tracks", "observations", "resources")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration-commit", required=True)
    main(parser.parse_args().preregistration_commit)
