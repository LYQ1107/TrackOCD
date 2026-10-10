#!/usr/bin/env python3
"""Preregistered one-worker Train-only SAM-grid interface smoke, not a benchmark."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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
from scripts.trackocd_core.smoke_masa_native import (
    ISOLATED, owned_process_tree_rss, select_private_image_paths, snapshot_instances, validate_train_images)

PLAN = ROOT / "configs/trackocd_core/masa_amg_smoke.json"
RUN = ROOT / "outputs/trackocd_core/audit/masa_amg_smoke_private"
SUMMARY = ROOT / "outputs/trackocd_core/audit/masa_amg_smoke.json"
DELIVERY = ROOT / "outputs/trackocd_core/audit/masa_amg_preregistration_delivery.json"


def memory():
    return {k: int(v.split()[0]) for k, v in
            (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}


def worker_input_barrier():
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("Frozen AMG worker prohibits network/external child execution")
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = os.fsdecode(args[0])
            if ("/TAO-Amodal/annotations/" in path or "/recovered_splits/" in path
                    or path.endswith(("/roles.json", "/train_labels.parquet", "/gt_train_known/selection_plan.json"))
                    or "/frames/test/" in path or "/frames/val/" in path):
                raise PermissionError("AMG worker prohibits GT/role/Val/Test inputs")
    sys.addaudithook(audit)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def worker():
    started = time.monotonic()
    config = json.loads(PLAN.read_text())
    inp = json.loads((RUN / "worker_input.json").read_text())
    images = validate_train_images(inp["image_paths"])
    worker_input_barrier()
    result = {"schema_version": "trackocd.core.masa_amg_smoke_result.v1", "status": "RUNNING",
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "replays": [],
              "real_image_forwards_started": 0, "preregistration_commit": inp["preregistration_commit"],
              "config_sha256": sha256_file(PLAN), "primary_freeze_permitted": False,
              "scientific_pass_permitted": False, "training": False, "optimizer_used": False,
              "formal_cache_started": False, "val_or_test_access": False, "gt_runtime_input": False,
              "new_image_or_checkpoint_download": False, "external_process_interference": False,
              "worker_network_child_annotation_barrier_installed": True}
    try:
        import numpy as np
        import torch
        from PIL import Image
        from src.trackocd_core.masa_native import build_frozen_model
        from src.trackocd_core.masa_amg import infer_current_frame, require_defaults, verified_sources
        require_defaults(config["amg_defaults"])
        torch.set_num_threads(1); torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        model, proof = build_frozen_model()
        if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise ValueError("Nonfinite frozen model state")
        result.update(checkpoint_sha256=proof["checkpoint_sha256"], strict_state_tensor_keys=proof["strict_state_tensor_keys"],
                      source_files=verified_sources(), model_state_initial_sha256=state_digest(model),
                      input={"unique_train_images": len(images), "image_sha256": [sha256_file(p) for p in images],
                             "source_plan_sha256": inp["source_plan_sha256"], "selection_bias": inp["selection_bias"],
                             "temporal_spacing": inp["temporal_spacing"]})
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., config["limits"]["gpu_reserved_bytes"] / total), 0)
        model.to("cuda:0"); torch.cuda.reset_peak_memory_stats()
        def guard():
            limits, mem = config["limits"], memory()
            if (time.monotonic() - started > limits["wall_seconds"]
                    or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["host_rss_bytes"]
                    or torch.cuda.max_memory_reserved() > limits["gpu_reserved_bytes"]
                    or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]):
                raise RuntimeError("Registered wall-time/RAM/GPU guard; stop this worker only")
        for replay in range(2):
            model.tracker.reset()
            rows = []; result["replays"].append({"changed_future_pixels": bool(replay), "frames": rows})
            for ordinal, path in enumerate(images):
                guard()
                with Image.open(path) as raw:
                    current = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
                if replay and ordinal >= 2:
                    current = 255 - current
                result["real_image_forwards_started"] += 1
                atomic_json(RUN / "worker_result.json", result)
                frame_started = time.monotonic()
                detections, tracks, masks = infer_current_frame(model, current, ordinal, "cuda:0", guard)
                torch.cuda.synchronize(); guard()
                rows.append({"ordinal": ordinal, "current_pixels_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                             "detections": snapshot_instances(detections, False), "tracks": snapshot_instances(tracks, True),
                             "masks": masks, "elapsed_seconds": time.monotonic() - frame_started})
                atomic_json(RUN / "worker_result.json", result)
                print(json.dumps({"replay": replay, "ordinal": ordinal, "detections": len(detections),
                                  "tracks": len(tracks), "elapsed_seconds": rows[-1]["elapsed_seconds"]}), flush=True)
        first, changed = [r["frames"] for r in result["replays"]]
        if not sum(r["detections"]["count"] for r in first) or not sum(r["tracks"]["count"] for r in first):
            raise ValueError("All-empty real proposal/track interface; not a nontrivial causal PASS")
        comparisons = [{"ordinal": i, "same_current_pixels": first[i]["current_pixels_sha256"] == changed[i]["current_pixels_sha256"],
                        "exact_arrays_equal": all(first[i][k]["array_sha256"] == changed[i][k]["array_sha256"]
                                                  for k in ("detections", "tracks"))} for i in (0, 1)]
        result["prefix_invariance"] = {"comparisons": comparisons,
            "future_pixels_actually_changed": all(first[i]["current_pixels_sha256"] != changed[i]["current_pixels_sha256"] for i in (2, 3)),
            "scope": "Two four-frame replays only; not an all-Val causality certificate"}
        if not result["prefix_invariance"]["future_pixels_actually_changed"] or not all(c["same_current_pixels"] and c["exact_arrays_equal"] for c in comparisons):
            raise ValueError("Exact prefix invariance failed; no tolerance or tracker repair")
        result["model_state_final_sha256"] = state_digest(model)
        result["all_parameters_frozen"] = not model.training and not any(p.requires_grad for p in model.parameters())
        if result["model_state_initial_sha256"] != result["model_state_final_sha256"] or not result["all_parameters_frozen"]:
            raise ValueError("Frozen model tensors changed")
        guard()
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["status"] = "PASS_BOUNDED_SAM_GRID_INTERFACE_NOT_PRIMARY_QUALIFICATION"
    except Exception as exc:
        result.update(status="BLOCKED_AMG_SMOKE_ENGINEERING", error=type(exc).__name__ + ": " + str(exc))
        import traceback
        traceback.print_exc()
    result.update(elapsed_seconds=time.monotonic() - started,
                  peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(RUN / "worker_result.json", result)
    return 0 if result["status"].startswith("PASS") else 1


def preflight(commit):
    from scripts.trackocd_core.install_masa_runtime import distribution_receipt, matches_install_provenance
    from src.trackocd_core.masa_amg import require_defaults, verified_sources
    if RUN.exists() or SUMMARY.exists():
        raise ValueError("Preserve completed/partial AMG attempt; no automatic rerun")
    delivery, config = json.loads(DELIVERY.read_text()), json.loads(PLAN.read_text())
    if delivery["commit"] != commit or delivery["remote_verified"] is not True or delivery["config_sha256"] != sha256_file(PLAN):
        raise ValueError("Exact remote-verified preregistration required")
    for relative in ("configs/trackocd_core/masa_amg_smoke.json", "configs/trackocd_core/sam_amg_source_manifest.json",
                     "src/trackocd_core/masa_amg.py", "scripts/trackocd_core/smoke_masa_amg.py"):
        if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != (ROOT / relative).read_bytes():
            raise ValueError("AMG source/config changed since preregistration")
    for relative, digest in config["unchanged_sources_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError("Inherited source/config changed: " + relative)
    verified_sources(); require_defaults(config["amg_defaults"])
    install = json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_install.json").read_text())
    if (install["status"] != "PASS_HASH_LOCKED_ISOLATED_INSTALL_NOT_MODEL_COMPATIBILITY"
            or len(install["completed_packages"]) != 48
            or install["dependency_lock_sha256"] != sha256_file(ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution/pylock.toml")):
        raise ValueError("Existing isolated install/lock is incomplete or changed")
    installed = distribution_receipt(ISOLATED)
    wheels = json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_wheel_inventory.json").read_text())["wheels"]
    for row in wheels:
        name = row["name"].lower().replace("_", "-")
        if not matches_install_provenance(installed.get(name, {}), row, install["verified_install_operations"].get(name)):
            raise ValueError("Runtime distribution changed: " + name)
    mem, limits = memory(), config["limits"]
    if (mem["MemAvailable"] * 1024 - limits["host_rss_bytes"] < mem["MemTotal"] * 1024 * limits["minimum_ram_headroom_fraction"]
            or shutil.disk_usage(ROOT).free < limits["minimum_disk_headroom_bytes"]):
        raise RuntimeError("Resource wait: preserve registered RAM/disk headroom")
    used = sum(int(line.split()[0]) for line in subprocess.check_output(
        ["du", "-s", "-B1", str(ROOT), str(ISOLATED.parent.parent)], text=True).splitlines())
    if used + limits["output_allocated_bytes"] > limits["overall_soft_bytes"]:
        raise RuntimeError("Conservative repository + owned environment exceeds soft budget")
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=10).split())
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10)
    selected = next((row.split(",")[0].strip() for row in snapshot.splitlines()
                     if (row.split(",")[0].strip() not in occupied and int(row.split(",")[1]) < 100
                         and int(row.split(",")[2]) >= 10240 and int(row.split(",")[3]) == 0)), None)
    if not selected:
        raise RuntimeError("Resource wait: no freshly idle GPU; do not disturb external processes")
    return config, selected, snapshot


def parent(commit):
    from scripts.trackocd_core.install_masa_runtime import allocated_bytes
    config, gpu, snapshot = preflight(commit)
    selection = {**select_private_image_paths(), "preregistration_commit": commit}
    RUN.mkdir(parents=True, exist_ok=False)
    atomic_json(RUN / "worker_input.json", selection)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(RUN / "matplotlib"),
               TORCH_HOME=str(RUN / "torch"), HF_HOME=str(RUN / "hf"))
    started, peak_rss, error = time.monotonic(), 0, None
    with (RUN / "worker.log").open("xb") as log:
        process = subprocess.Popen([str(ISOLATED), str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                   env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"status": "RUNNING_BOUNDED_TRAIN_AMG_SMOKE", "gpu_uuid": gpu, "owned_pid": process.pid}), flush=True)
        try:
            while process.poll() is None:
                limits, mem = config["limits"], memory()
                peak_rss = max(peak_rss, owned_process_tree_rss(process.pid))
                if (time.monotonic() - started > limits["wall_seconds"] or peak_rss > limits["host_rss_bytes"]
                        or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]
                        or allocated_bytes([RUN]) > limits["output_allocated_bytes"]):
                    raise RuntimeError("Owned worker exceeded registered time/RAM/output guard")
                time.sleep(1)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)  # Exact newly spawned owned session only.
                process.wait(timeout=30)
    result_path = RUN / "worker_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "BLOCKED_AMG_WORKER_NO_RESULT"}
    if error or process.returncode != 0:
        result.update(status="BLOCKED_AMG_SMOKE_RESOURCE_OR_ENGINEERING", supervisor_error=error,
                      primary_freeze_permitted=False, scientific_pass_permitted=False)
    result["supervisor"] = {"gpu_uuid": gpu, "fresh_gpu_query": snapshot, "worker_returncode": process.returncode,
                            "peak_owned_tree_rss_bytes": peak_rss, "private_allocated_bytes": allocated_bytes([RUN]),
                            "preregistration_commit": commit, "worker_result_sha256": sha256_file(result_path) if result_path.exists() else None}
    atomic_json(SUMMARY, result)
    if result["status"].startswith("PASS"):
        atomic_json(RUN / ".done", {"status": result["status"], "summary_sha256": sha256_file(SUMMARY), "worker_returncode": 0})
    print(json.dumps({"status": result["status"], "elapsed_seconds": result.get("elapsed_seconds"),
                      "summary_sha256": sha256_file(SUMMARY), "worker_returncode": process.returncode}), flush=True)
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-commit")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    if not args.preregistration_commit:
        parser.error("--preregistration-commit required")
    raise SystemExit(parent(args.preregistration_commit))
