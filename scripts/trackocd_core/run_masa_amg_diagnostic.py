#!/usr/bin/env python3
"""Independent <=64-image frozen SAM-grid Val diagnostic, not qualification."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import val_image, json_digest
from src.trackocd_core.amg_physical_diagnostic import completed_video, seal_video
from scripts.trackocd_core.smoke_masa_native import ISOLATED, owned_process_tree_rss, snapshot_instances
from scripts.trackocd_core.smoke_masa_amg import state_digest, memory

PUBLIC = ROOT / "configs/trackocd_core/masa_amg_physical_diagnostic.json"
RUN = ROOT / "outputs/trackocd_core/physical/masa_amg_val4_first16"
FRAMES = Path("/data3/liuyeqiang/TAO-Amodal/frames")
DELIVERY = ROOT / "outputs/trackocd_core/audit/masa_amg_diagnostic_preregistration_delivery.json"


def load_plan():
    from src.trackocd_core.masa_amg import require_defaults, verified_sources
    config = json.loads(PUBLIC.read_text())
    private = ROOT / config["private_plan"]
    if sha256_file(private) != config["private_plan_sha256"]:
        raise ValueError("Original metadata-only fixed image plan changed")
    plan = json.loads(private.read_text())
    if ([v["video_id"] for v in plan["videos"]] != config["video_ids"]
            or [len(v["images"]) for v in plan["videos"]] != config["images_per_video"]
            or [[i["frame_index"] for i in v["images"]] for v in plan["videos"]] != config["frame_indices"]
            or sum(len(v["images"]) for v in plan["videos"]) != 64):
        raise ValueError("No video/frame substitution or resampling")
    for relative, digest in config["unchanged_sources_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError("Inherited pinned source changed: " + relative)
    inherited = ROOT / config["inherited_amg_config"]
    if sha256_file(inherited) != config["inherited_amg_config_sha256"]:
        raise ValueError("No AMG interface/default changes")
    original = json.loads(inherited.read_text())
    for relative, digest in original["unchanged_sources_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError("Inherited frozen model/runtime helper changed")
    require_defaults(original["amg_defaults"]); verified_sources()
    smoke = ROOT / config["source_train_smoke"]
    if sha256_file(smoke) != config["source_train_smoke_sha256"] or not json.loads(smoke.read_text())["status"].startswith("PASS"):
        raise ValueError("Required bounded Train interface result changed/missing")
    return config, plan, sha256_file(PUBLIC)


def input_barrier():
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("Frozen diagnostic prohibits network/external children")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ("/TAO-Amodal/annotations/" in path or "/recovered_splits/" in path
                    or path.endswith(("/roles.json", "/train_labels.parquet")) or "/frames/test/" in path):
                raise PermissionError("Physical predictor cannot open GT/role/Test input")
    sys.addaudithook(audit)


def worker():
    started = time.monotonic()
    config, plan, digest = load_plan()
    input_barrier()
    result = {"schema_version": "trackocd.core.masa_amg_diagnostic_prediction.v1", "status": "RUNNING",
              "config_sha256": digest, "private_plan_sha256": config["private_plan_sha256"], "videos": [],
              "real_image_forwards_started": 0, "real_image_forwards_completed": 0,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "boundary": {"gt_or_category_model_input": False, "training": False, "future_model_input": False,
                           "test_access": False, "offline_filtering": False, "extra_weight_download": False,
                           "foreign_process_interference": False}, "primary_freeze_permitted": False}
    try:
        import numpy as np
        import torch
        from PIL import Image
        from src.trackocd_core.masa_native import build_frozen_model
        from src.trackocd_core.masa_amg import infer_current_frame
        torch.set_num_threads(1); torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        model, proof = build_frozen_model()
        initial = state_digest(model)
        if (proof["checkpoint_sha256"] != config["checkpoint_sha256"]
                or json_digest(proof["native_model_config"]) != config["native_model_config_sha256"]
                or initial != config["model_state_sha256"]
                or any(not torch.isfinite(t).all() for t in model.state_dict().values())):
            raise ValueError("Frozen checkpoint/config/state/finiteness changed")
        result.update(frozen_state_initial_sha256=initial, strict_state_tensor_keys=proof["strict_state_tensor_keys"])
        torch.cuda.set_per_process_memory_fraction(min(1., config["limits"]["max_gpu_bytes"] / torch.cuda.get_device_properties(0).total_memory), 0)
        model.to("cuda:0"); torch.cuda.reset_peak_memory_stats()
        def guard():
            limits, mem = config["limits"], memory()
            if (time.monotonic() - started > limits["max_seconds"]
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["max_host_rss_bytes"]
                    or torch.cuda.max_memory_reserved() > limits["max_gpu_bytes"]
                    or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]):
                raise RuntimeError("Registered physical prediction guard; stop this owned worker only")
        for video in plan["videos"]:
            model.tracker.reset()
            arrays = {"image_id": [], "frame_index": [], "frame_offsets": [0], "track_id": [], "boxes": [],
                      "score": [], "det_offsets": [0], "det_boxes": [], "det_score": []}
            stats = {"frames": [], "empty_detection_frames": 0, "empty_tracking_frames": 0}
            for ordinal, image in enumerate(video["images"]):
                guard()
                path = val_image(FRAMES, image["image_path"])
                if sha256_file(path) != image["image_sha256"] or result["real_image_forwards_started"] >= 64:
                    raise ValueError("Changed registered image or forward ceiling")
                with Image.open(path) as raw:
                    pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
                result["real_image_forwards_started"] += 1
                result.update(current_video_id=video["video_id"], current_ordinal=ordinal)
                atomic_json(RUN / "prediction_manifest.json", result)
                detections, tracks, masks = infer_current_frame(model, pixels, ordinal, "cuda:0", guard)
                torch.cuda.synchronize(); guard()
                snapshot_instances(detections, False); snapshot_instances(tracks, True)
                stats["frames"].append({"ordinal": ordinal, "detections": len(detections), "tracks": len(tracks), "masks": masks})
                stats["empty_detection_frames"] += len(detections) == 0; stats["empty_tracking_frames"] += len(tracks) == 0
                arrays["image_id"].append(image["image_id"]); arrays["frame_index"].append(image["frame_index"])
                arrays["track_id"].extend(tracks.instances_id.cpu().tolist())
                arrays["boxes"].extend(tracks.bboxes.cpu().tolist()); arrays["score"].extend(tracks.scores.cpu().tolist())
                arrays["frame_offsets"].append(len(arrays["track_id"]))
                arrays["det_boxes"].extend(detections.bboxes.cpu().tolist()); arrays["det_score"].extend(detections.scores.cpu().tolist())
                arrays["det_offsets"].append(len(arrays["det_score"]))
                result["real_image_forwards_completed"] += 1
            if state_digest(model) != initial or model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError("Frozen state changed; no video seal")
            stats["frozen_state_unchanged"] = True
            arrays = {k: np.asarray(v, dtype=np.float32 if k in {"boxes", "score", "det_boxes", "det_score"} else np.int64) for k, v in arrays.items()}
            arrays["boxes"] = arrays["boxes"].reshape(-1, 4); arrays["det_boxes"] = arrays["det_boxes"].reshape(-1, 4)
            result["videos"].append(seal_video(RUN, video, arrays, digest, stats))
            atomic_json(RUN / "prediction_manifest.json", result)
            print(json.dumps({"completed_videos": len(result["videos"]), "completed_forwards": result["real_image_forwards_completed"]}), flush=True)
        result.update(status="SEALED_BOUNDED_AMG_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION", frozen_state_final_sha256=state_digest(model),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as exc:
        result.update(status="BLOCKED_AMG_BOUNDED_PHYSICAL_PREDICTION", error=type(exc).__name__ + ": " + str(exc))
        import traceback
        traceback.print_exc()
    result.update(elapsed_seconds=time.monotonic() - started, peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(RUN / "prediction_manifest.json", result)
    return 0 if result["status"].startswith("SEALED") else 1


def parent(commit):
    from scripts.trackocd_core.install_masa_runtime import allocated_bytes, distribution_receipt, matches_install_provenance
    if RUN.exists():
        raise ValueError("Preserve any completed/partial diagnostic; no automatic rerun")
    config, plan, digest = load_plan()
    delivery = json.loads(DELIVERY.read_text())
    if delivery["commit"] != commit or not delivery["remote_verified"] or delivery["config_sha256"] != digest:
        raise ValueError("Exact remote-verified preregistration required")
    for relative in (str(PUBLIC.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
                     "src/trackocd_core/amg_physical_diagnostic.py", "scripts/trackocd_core/evaluate_masa_amg_diagnostic.py"):
        if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != (ROOT / relative).read_bytes():
            raise ValueError("Diagnostic source/config changed after preregistration")
    install = json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_install.json").read_text())
    if (install["status"] != "PASS_HASH_LOCKED_ISOLATED_INSTALL_NOT_MODEL_COMPATIBILITY" or len(install["completed_packages"]) != 48
            or install["dependency_lock_sha256"] != sha256_file(ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution/pylock.toml")):
        raise ValueError("Existing isolated install identity changed")
    installed = distribution_receipt(ISOLATED)
    for row in json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_wheel_inventory.json").read_text())["wheels"]:
        name = row["name"].lower().replace("_", "-")
        if not matches_install_provenance(installed.get(name, {}), row, install["verified_install_operations"].get(name)):
            raise ValueError("Installed distribution changed: " + name)
    mem, limits = memory(), config["limits"]
    if (mem["MemAvailable"] * 1024 - limits["max_host_rss_bytes"] < mem["MemTotal"] * 1024 * limits["minimum_ram_headroom_fraction"]
            or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]
            or allocated_bytes([ROOT, ISOLATED.parent.parent]) + limits["max_output_bytes"] > limits["overall_soft_bytes"]):
        raise RuntimeError("Resource wait: preserve RAM/disk/goal storage headroom")
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=10).split())
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10)
    selected = next((row.split(",")[0].strip() for row in snapshot.splitlines() if row.split(",")[0].strip() not in occupied
                     and int(row.split(",")[1]) < 100 and int(row.split(",")[2]) >= 10240 and int(row.split(",")[3]) == 0), None)
    if not selected:
        raise RuntimeError("Resource wait: no freshly idle GPU; no external intervention")
    (RUN / "shards").mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=selected, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(RUN / "matplotlib"),
        HF_HOME=str(RUN / "hf"), TORCH_HOME=str(RUN / "torch"))
    started, peak_rss, error = time.monotonic(), 0, None
    with (RUN / "worker.log").open("xb") as log:
        process = subprocess.Popen([str(ISOLATED), str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"status": "RUNNING_AMG_VAL4_FIRST16", "owned_pid": process.pid, "gpu_uuid": selected}), flush=True)
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, owned_process_tree_rss(process.pid)); mem = memory()
                if (time.monotonic() - started > limits["max_seconds"] or peak_rss > limits["max_host_rss_bytes"]
                        or allocated_bytes([RUN]) > limits["max_output_bytes"]
                        or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]):
                    raise RuntimeError("Registered owned prediction supervisor resource stop")
                time.sleep(1)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=30)  # This newly spawned owned session only.
    result = {"schema_version": "trackocd.core.masa_amg_diagnostic_supervisor.v1", "preregistration_commit": commit,
              "config_sha256": digest, "remote_preregistration_verified": True, "worker_returncode": process.returncode,
              "error": error, "gpu_uuid": selected, "fresh_gpu_query": snapshot, "elapsed_seconds": time.monotonic() - started,
              "peak_owned_tree_rss_bytes": peak_rss, "private_allocated_bytes": allocated_bytes([RUN])}
    if process.returncode == 0 and error is None:
        prediction = json.loads((RUN / "prediction_manifest.json").read_text())
        sealed = [completed_video(RUN, video, digest) for video in plan["videos"]]
        if (any(s is None for s in sealed) or sealed != prediction["videos"] or prediction["real_image_forwards_completed"] != 64
                or prediction["status"] != "SEALED_BOUNDED_AMG_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION"):
            raise ValueError("Entire diagnostic must be sealed; no GT evaluation")
        result["full_prediction_seal_sha256"] = sha256_file(RUN / "prediction_manifest.json")
    atomic_json(RUN / "supervisor.json", result)
    print(json.dumps(result), flush=True)
    return 0 if process.returncode == 0 and error is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true"); parser.add_argument("--preregistration-commit")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    if not args.preregistration_commit:
        parser.error("--preregistration-commit required")
    raise SystemExit(parent(args.preregistration_commit))
