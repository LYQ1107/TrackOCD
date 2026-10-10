"""Exact-preregistered ownership and resources for fixed full cached CPU audit."""
from __future__ import annotations
import datetime as dt
import importlib.metadata
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
from src.trackocd_core.rpn_bytetrack import PARAMETERS, completed_video
from src.trackocd_core.physical_qualification import json_digest
from scripts.trackocd_core.install_masa_runtime import allocated_bytes
from scripts.trackocd_core.smoke_masa_native import owned_process_tree_rss
from scripts.trackocd_core.smoke_masa_amg import memory

PUBLIC = ROOT / "configs/trackocd_core/rpn_bytetrack_full_quality.json"
RUN = ROOT / "outputs/trackocd_core/physical/rpn_bytetrack_full_val_annotated"
SUMMARY = ROOT / "outputs/trackocd_core/audit/rpn_bytetrack_full_val_physical_result.json"
DELIVERY = ROOT / "outputs/trackocd_core/audit/rpn_bytetrack_full_preregistration_delivery.json"
ENV = Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")
REGISTERED = (
    "configs/trackocd_core/rpn_bytetrack_full_quality.json",
    "src/trackocd_core/rpn_bytetrack_full.py", "src/trackocd_core/evaluation/physical_full.py",
    "scripts/trackocd_core/rpn_bytetrack_full_runtime.py",
    "scripts/trackocd_core/run_rpn_bytetrack_full.py", "scripts/trackocd_core/evaluate_rpn_bytetrack_full.py")


def load_plan():
    config = json.loads(PUBLIC.read_text()); digest = sha256_file(PUBLIC)
    current_runtime = {"python": sys.version.split()[0], "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy")}
    if current_runtime != config["runtime"]: raise ValueError("Frozen existing CPU runtime changed")
    for relative, expected in config["unchanged_sources_sha256"].items():
        if sha256_file(ROOT / relative) != expected: raise ValueError("Pinned inherited source changed: " + relative)
    if config["tracker_parameters"] != PARAMETERS or config["limits"]["cpu_workers"] != 1:
        raise ValueError("Fixed tracker/one CPU; no search")
    for key in ("training", "test_access", "pixel_inference_repeated", "parameter_search", "primary_freeze_permitted", "semantic_feedback"):
        if config[key]: raise ValueError("Original M1 scope boundary")
    private = ROOT / config["private_plan"]
    if sha256_file(private) != config["private_plan_sha256"]: raise ValueError("Entire frozen metadata universe required")
    plan = json.loads(private.read_text())
    ids = [v["video_id"] for v in plan["videos"]]
    if (ids != sorted(set(ids)) or len(ids) != config["videos"]
            or sum(len(v["images"]) for v in plan["videos"]) != config["images"]):
        raise ValueError("No video/frame omission or target enrichment")
    for video in plan["videos"]:
        frames = [i["frame_index"] for i in video["images"]]
        if not frames or any(b <= a for a, b in zip(frames, frames[1:])): raise ValueError("Strict entire-video chronology")
    if sha256_file(ROOT / config["preflight"]) != config["preflight_sha256"]:
        raise ValueError("Full raw source preflight changed")
    source = ROOT / config["input_source"]; manifest = source / "prediction_manifest.json"
    if sha256_file(manifest) != config["input_manifest_sha256"] or sha256_file(ROOT / config["input_config"]) != config["input_config_sha256"]:
        raise ValueError("Frozen native detector source identity changed")
    receipt = json.loads(manifest.read_text())
    records = {r["video_id"]: r for r in receipt["videos"]}
    if (receipt["status"] != "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM" or sorted(records) != ids
            or receipt["images"] != config["images"] or receipt["total_detection_rows"] != config["raw_detections"]):
        raise ValueError("Incomplete original detector stream")
    return config, plan, records, digest


def verified_source(video, config, records):
    row = records[video["video_id"]]; source = ROOT / config["input_source"]
    marker = source / "shards" / f"video_{video['video_id']:04d}.complete.json"
    filename = Path(row["npz_filename"]); path = source / "shards" / filename
    if (marker.is_symlink() or json.loads(marker.read_text()) != row or row["status"] != "COMPLETE_VIDEO"
            or row["config_sha256"] != config["input_config_sha256"] or row["video_plan_sha256"] != json_digest(video)
            or row["images"] != len(video["images"]) or not row["frame_statistics"]["frozen_state_unchanged"]
            or filename.name != str(filename) or path.is_symlink() or path.stat().st_size != row["npz_bytes"]
            or sha256_file(path) != row["npz_sha256"]):
        raise ValueError("Source complete marker/payload/config/whole-video identity changed")
    return path


def owned_output_bytes():
    """Snapshot owned tree, tolerating completed per-video scratch disposal."""
    seen, total, pending = set(), 0, [RUN]
    while pending:
        path = pending.pop()
        try: stat = path.lstat()
        except FileNotFoundError: continue
        identity = stat.st_dev, stat.st_ino
        if identity in seen: continue
        seen.add(identity); total += stat.st_blocks * 512
        if path.is_dir() and not path.is_symlink():
            try:
                with os.scandir(path) as entries: pending.extend(Path(entry.path) for entry in entries)
            except FileNotFoundError: continue
    return total


def resource_guard(config, mode, started, check_storage=False):
    limits = config["limits"]; mem = memory()
    if (time.monotonic() - started > limits[mode + "_seconds"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits[mode + "_host_rss_bytes"]
            or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]
            or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]):
        raise RuntimeError("Registered fresh-worker CPU/time/RAM/disk guard")
    if check_storage and owned_output_bytes() > limits["max_run_allocated_bytes"]:
        raise RuntimeError("Registered owned output budget")


def verify_prediction(config, plan, records, digest):
    prediction_path = RUN / "prediction_manifest.json"
    prediction = json.loads(prediction_path.read_text())
    supervisor = json.loads((RUN / "prediction_supervisor.json").read_text())
    sealed = [completed_video(RUN, v, digest, config["tracker_source_sha256"]) for v in plan["videos"]]
    if (any(s is None for s in sealed) or sealed != prediction["videos"]
            or prediction["status"] != "SEALED_COMPLETE_FULL_VAL_CLASSICAL_RPN_BYTETRACK"
            or prediction["config_sha256"] != digest or prediction["raw_detections"] != config["raw_detections"]
            or prediction["production_frame_updates"] != config["images"]
            or prediction["total_frame_updates"] != config["limits"]["total_frame_updates_with_probe"]
            or not prediction["causality_probe"]["pass"] or prediction["torch_imported"]
            or prediction["ordered_source_and_detector_array_identity_sha256"] != config["ordered_source_and_detector_array_identity_sha256"]
            or supervisor["worker_returncode"] != 0 or supervisor["error"] is not None
            or supervisor["prediction_manifest_sha256"] != sha256_file(prediction_path)):
        raise ValueError("All whole-video classical predictions/successful supervisor must seal before GT")
    for video, row in zip(plan["videos"], sealed):
        if row["source_npz_sha256"] != records[video["video_id"]]["npz_sha256"]:
            raise ValueError("Exact frozen source lineage before evaluation")
    return prediction, sealed


def supervise(mode, script, commit):
    config, plan, records, digest = load_plan()
    if mode == "prediction" and RUN.exists(): raise ValueError("Preserve any complete/partial replay; no automatic restart")
    if mode == "evaluation":
        if SUMMARY.exists() or (RUN / "evaluation_attempt.json").exists(): raise ValueError("Preserve any completed/failed evaluator")
        verify_prediction(config, plan, records, digest)
    delivery = json.loads(DELIVERY.read_text())
    if delivery["commit"] != commit or not delivery["remote_verified"] or delivery["config_sha256"] != digest:
        raise ValueError("Exact remote full-quality preregistration required")
    for relative in REGISTERED:
        if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != (ROOT / relative).read_bytes():
            raise ValueError("Full-quality preregistered source/config changed")
    limits, mem = config["limits"], memory()
    if (mem["MemAvailable"] * 1024 - limits[mode + "_host_rss_bytes"] < mem["MemTotal"] * 1024 * limits["minimum_ram_headroom_fraction"]
            or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]
            or allocated_bytes([ROOT, ENV]) + limits["max_run_allocated_bytes"] + limits["max_public_result_bytes"] > limits["overall_soft_bytes"]):
        raise RuntimeError("Resource wait; don't expand budget or affect foreign jobs")
    for video in plan["videos"]: verified_source(video, config, records)
    if mode == "prediction": (RUN / "shards").mkdir(parents=True, exist_ok=False)
    else: atomic_json(RUN / "evaluation_attempt.json", {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "preregistration_commit": commit})
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    started, peak, error = time.monotonic(), 0, None
    with (RUN / (mode + "_worker.log")).open("xb") as log:
        process = subprocess.Popen([sys.executable, str(script), "--worker"], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"status": "RUNNING_OWNED_FULL_CPU_" + mode.upper(), "owned_pid": process.pid}), flush=True)
        try:
            while process.poll() is None:
                peak = max(peak, owned_process_tree_rss(process.pid)); mem = memory()
                if (time.monotonic() - started > limits[mode + "_seconds"] or peak > limits[mode + "_host_rss_bytes"]
                        or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]
                        or owned_output_bytes() > limits["max_run_allocated_bytes"]
                        or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]):
                    raise RuntimeError("Owned full CPU supervisor resource stop")
                time.sleep(1)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()  # Newly created owned session only.
    result = {"schema_version": "trackocd.core.rpn_bytetrack_full_supervisor.v1", "mode": mode,
              "preregistration_commit": commit, "config_sha256": digest, "worker_returncode": process.returncode,
              "error": error, "elapsed_seconds": time.monotonic() - started, "peak_owned_tree_rss_bytes": peak,
              "gpu_used": False, "private_allocated_bytes": owned_output_bytes()}
    if mode == "prediction" and process.returncode == 0 and error is None:
        result["prediction_manifest_sha256"] = sha256_file(RUN / "prediction_manifest.json")
    atomic_json(RUN / (mode + "_supervisor.json"), result)
    if mode == "prediction" and process.returncode == 0 and error is None: verify_prediction(config, plan, records, digest)
    print(json.dumps(result), flush=True)
    return 0 if process.returncode == 0 and error is None else 1
