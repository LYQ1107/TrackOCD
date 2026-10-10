#!/usr/bin/env python3
"""Preregistered one-CPU raw-RPN/ByteTrack replay; no pixels/GT or new weights."""
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
from src.trackocd_core.rpn_bytetrack import (PARAMETERS, read_detector_video, detector_digest, new_tracker,
                                          step, causal_probe, seal_video, completed_video)
from scripts.trackocd_core.smoke_masa_native import owned_process_tree_rss
from scripts.trackocd_core.smoke_masa_amg import memory

PUBLIC = ROOT / "configs/trackocd_core/rpn_bytetrack_diagnostic.json"
RUN = ROOT / "outputs/trackocd_core/physical/rpn_bytetrack_val4_first16"
DELIVERY = ROOT / "outputs/trackocd_core/audit/rpn_bytetrack_preregistration_delivery.json"


def load_plan():
    config = json.loads(PUBLIC.read_text())
    for relative, expected in config["unchanged_sources_sha256"].items():
        if sha256_file(ROOT / relative) != expected: raise ValueError("Inherited pinned source changed: " + relative)
    inherited = ROOT / config["inherited_clip_config"]
    if sha256_file(inherited) != config["inherited_clip_config_sha256"]: raise ValueError("Existing clip protocol changed")
    private = ROOT / config["private_plan"]
    if sha256_file(private) != config["private_plan_sha256"]: raise ValueError("No metadata resampling")
    plan = json.loads(private.read_text())
    if ([v["video_id"] for v in plan["videos"]] != config["video_ids"] or [len(v["images"]) for v in plan["videos"]] != [16] * 4
            or [[i["frame_index"] for i in v["images"]] for v in plan["videos"]] != config["frame_indices"]):
        raise ValueError("Fixed4x16 original image universe required")
    preflight = ROOT / config["preflight_receipt"]
    if (sha256_file(preflight) != config["preflight_receipt_sha256"] or config["tracker_parameters"] != PARAMETERS
            or json.loads(preflight.read_text())["frozen_reference_tracker_parameters"] != PARAMETERS):
        raise ValueError("Exact prior tracker parameters; no score calibration or threshold search")
    source = ROOT / config["input_source"]; manifest_path = source / "prediction_manifest.json"
    if sha256_file(manifest_path) != config["input_manifest_sha256"]: raise ValueError("Frozen raw detector manifest changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "SEALED_BOUNDED_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION" or manifest["real_image_forwards_started"] != 64:
        raise ValueError("Entire original detector source must already be sealed")
    records = {v["video_id"]: v for v in manifest["videos"]}
    files = {}
    for video in plan["videos"]:
        row = records[video["video_id"]]; filename = Path(row["npz_filename"]); path = source / filename
        if filename.name != str(filename) or path.is_symlink() or path.stat().st_size != row["npz_bytes"] or sha256_file(path) != row["npz_sha256"]:
            raise ValueError("Source NPZ identity changed; no detector repeat")
        files[video["video_id"]] = path
    return config, plan, files, sha256_file(PUBLIC)


def input_barrier():
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("Classical worker prohibits network/external children")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ("/TAO-Amodal/" in path or "/recovered_splits/" in path or "/checkpoints/" in path
                    or path.endswith(("/roles.json", "/train_labels.parquet"))):
                raise PermissionError("Classical predictor prohibits pixels/GT/splits/weights/Test")
    sys.addaudithook(audit)


def worker():
    import numpy as np
    started = time.monotonic()
    config, plan, files, digest = load_plan(); input_barrier()
    result = {"schema_version": "trackocd.core.rpn_bytetrack_diagnostic_prediction.v1", "status": "RUNNING",
              "config_sha256": digest, "tracker_source_sha256": config["tracker_source_sha256"], "videos": [],
              "production_frame_updates": 0, "causal_probe_frame_updates": 0, "total_frame_updates": 0,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "primary_freeze_permitted": False,
              "boundary": {"pixel_inference": False, "new_weights": False, "training": False, "gpu": False,
                           "gt_or_semantic_or_native_track_input": False, "test_access": False, "score_repair": False,
                           "parameter_search": False, "foreign_process_interference": False}}
    try:
        limits = config["limits"]
        def guard():
            mem = memory()
            if (time.monotonic() - started > limits["max_seconds"]
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["max_host_rss_bytes"]
                    or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]
                    or result["total_frame_updates"] > limits["total_real_frame_updates_with_causal_probe"]):
                raise RuntimeError("Registered one-CPU classical guard")
        guard()
        for video_index, video in enumerate(plan["videos"]):
            guard(); source = files[video["video_id"]]; source_before = sha256_file(source); data = read_detector_video(source, video)
            before = detector_digest(data)
            if video_index == 0:
                result["causality_probe"] = causal_probe(data, config["tracker_parameters"])
                result["causal_probe_frame_updates"] = 8; result["total_frame_updates"] += 8
            tracker = new_tracker(config["tracker_parameters"])
            arrays = {**data, "frame_offsets": [0], "track_id": [], "boxes": [], "score": []}
            statistics = {"frames": [], "empty_output_frames": 0}
            for ordinal, image in enumerate(video["images"]):
                guard()
                begin, end = map(int, data["det_offsets"][ordinal:ordinal + 2])
                output = step(tracker, data["det_boxes"][begin:end], data["det_score"][begin:end])
                arrays["boxes"].extend(output[:, :4].tolist()); arrays["track_id"].extend(output[:, 4].astype(np.int64).tolist())
                arrays["score"].extend(output[:, 5].tolist()); arrays["frame_offsets"].append(len(arrays["track_id"]))
                statistics["frames"].append({"ordinal": ordinal, "input_detections": end - begin, "output_tracks": len(output)})
                statistics["empty_output_frames"] += len(output) == 0
                result["production_frame_updates"] += 1; result["total_frame_updates"] += 1
            if detector_digest(data) != before or sha256_file(source) != source_before:
                raise ValueError("Frozen detector input mutated")
            arrays["boxes"] = np.asarray(arrays["boxes"], dtype=np.float32).reshape(-1, 4)
            arrays["score"] = np.asarray(arrays["score"], dtype=np.float32)
            arrays["track_id"] = np.asarray(arrays["track_id"], dtype=np.int64); arrays["frame_offsets"] = np.asarray(arrays["frame_offsets"], dtype=np.int64)
            if detector_digest(arrays) != before: raise ValueError("No raw detector array transformations")
            result["videos"].append(seal_video(RUN, video, arrays, digest, source_before, config["tracker_source_sha256"], statistics))
            atomic_json(RUN / "prediction_manifest.json", result)
            print(json.dumps({"completed_videos": len(result["videos"]), "production_frame_updates": result["production_frame_updates"]}), flush=True)
        if result["production_frame_updates"] != 64 or result["total_frame_updates"] != 72 or "torch" in sys.modules:
            raise ValueError("Wrong production/probe count or unexpected Torch import")
        guard()
        result.update(status="SEALED_BOUNDED_RPN_BYTETRACK_NOT_QUALIFICATION", torch_imported=False,
                      input_arrays_unchanged=True, tracker_source_unchanged=sha256_file(ROOT / config["tracker_source"]) == config["tracker_source_sha256"])
        if not result["tracker_source_unchanged"]: raise ValueError("Classical source changed during run")
    except Exception as exc:
        result.update(status="BLOCKED_CLASSICAL_RPN_BYTETRACK", error=type(exc).__name__ + ": " + str(exc))
        import traceback
        traceback.print_exc()
    result.update(elapsed_seconds=time.monotonic() - started, peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(RUN / "prediction_manifest.json", result)
    return 0 if result["status"].startswith("SEALED") else 1


def parent(commit):
    from scripts.trackocd_core.install_masa_runtime import allocated_bytes
    if RUN.exists(): raise ValueError("Preserve any completed/partial replay; no automatic repeat")
    config, plan, files, digest = load_plan(); delivery = json.loads(DELIVERY.read_text())
    if delivery["commit"] != commit or not delivery["remote_verified"] or delivery["config_sha256"] != digest:
        raise ValueError("Exact remote preregistration required")
    for relative in (str(PUBLIC.relative_to(ROOT)), "src/trackocd_core/rpn_bytetrack.py", "src/trackocd_core/evaluation/physical_clip.py",
                     "scripts/trackocd_core/run_rpn_bytetrack_diagnostic.py", "scripts/trackocd_core/evaluate_rpn_bytetrack_diagnostic.py"):
        if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != (ROOT / relative).read_bytes():
            raise ValueError("Preregistered new source/config changed")
    limits, mem = config["limits"], memory()
    if (mem["MemAvailable"] * 1024 - limits["max_host_rss_bytes"] < mem["MemTotal"] * 1024 * limits["minimum_ram_headroom_fraction"]
            or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]
            or allocated_bytes([ROOT, Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")]) + limits["max_output_bytes"] > limits["overall_soft_bytes"]):
        raise RuntimeError("Resource wait: RAM/disk/goal budget headroom")
    (RUN / "shards").mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    started, peak_rss, error = time.monotonic(), 0, None
    with (RUN / "worker.log").open("xb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"status": "RUNNING_CPU_RPN_BYTETRACK", "owned_pid": process.pid}), flush=True)
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, owned_process_tree_rss(process.pid)); mem = memory()
                if (time.monotonic() - started > limits["max_seconds"] or peak_rss > limits["max_host_rss_bytes"]
                        or allocated_bytes([RUN]) > limits["max_output_bytes"]
                        or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]):
                    raise RuntimeError("Owned classical supervisor resource stop")
                time.sleep(1)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=30)  # Exact newly owned session only.
    result = {"schema_version": "trackocd.core.rpn_bytetrack_supervisor.v1", "preregistration_commit": commit,
              "config_sha256": digest, "worker_returncode": process.returncode, "error": error,
              "elapsed_seconds": time.monotonic() - started, "peak_owned_tree_rss_bytes": peak_rss, "gpu_used": False,
              "private_allocated_bytes": allocated_bytes([RUN])}
    if process.returncode == 0 and error is None:
        prediction = json.loads((RUN / "prediction_manifest.json").read_text())
        sealed = [completed_video(RUN, v, digest, config["tracker_source_sha256"]) for v in plan["videos"]]
        if any(s is None for s in sealed) or sealed != prediction["videos"] or prediction["status"] != "SEALED_BOUNDED_RPN_BYTETRACK_NOT_QUALIFICATION":
            raise ValueError("Entire classical stream must seal before GT")
        result["prediction_seal_sha256"] = sha256_file(RUN / "prediction_manifest.json")
    atomic_json(RUN / "supervisor.json", result); print(json.dumps(result), flush=True)
    return 0 if process.returncode == 0 and error is None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true"); parser.add_argument("--preregistration-commit")
    args = parser.parse_args()
    if args.worker: raise SystemExit(worker())
    if not args.preregistration_commit: parser.error("--preregistration-commit required")
    raise SystemExit(parent(args.preregistration_commit))
