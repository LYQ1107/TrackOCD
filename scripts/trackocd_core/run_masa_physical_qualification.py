#!/usr/bin/env python3
"""Resumable video-atomic M1 frozen inference, never GT scoring or training."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import completed_video, seal_video, val_image, json_digest
from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children, owned_process_tree_rss, snapshot_instances

PUBLIC = ROOT / "configs/trackocd_core/masa_physical_qualification.json"
PRIVATE = ROOT / "outputs/trackocd_core/audit/masa_physical_qualification_plan.json"
RUN = ROOT / "outputs/trackocd_core/physical/masa_full_val_annotated"
ENV = Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke")
FRAMES = Path("/data3/liuyeqiang/TAO-Amodal/frames")


def load_plan():
    config, plan = json.loads(PUBLIC.read_text()), json.loads(PRIVATE.read_text())
    if sha256_file(PRIVATE) != config["private_plan_sha256"] or len(plan["videos"]) != config["videos"] or sum(len(v["images"]) for v in plan["videos"]) != config["images"]:
        raise ValueError("Entire frozen plan changed")
    return config, plan, sha256_file(PUBLIC)


def frozen_state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def worker(attempt: str, index: int, count: int) -> int:
    started = time.monotonic()
    receipt = RUN / "attempts" / attempt / f"worker_{index}.json"
    result = {"schema_version": "trackocd.core.masa_full_val_worker.v1", "status": "RUNNING", "attempt": attempt,
              "worker_index": index, "worker_count": count, "pid": os.getpid(), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "actual_forwards_started_this_attempt": 0, "actual_forwards_completed_this_attempt": 0,
              "completed_video_ids_this_attempt": [], "reused_complete_videos": 0,
              "boundary": {"training": False, "gt_or_role_model_input": False, "text_or_future_model_input": False,
                           "test_access": False, "posthoc_filtering": False, "new_weight_download": False, "foreign_process_interference": False}}
    atomic_json(receipt, result)
    try:
        config, plan, digest = load_plan()
        if not 0 <= index < count <= config["limits"]["max_gpu_workers"]:
            raise ValueError("Unregistered worker partition")
        deny_network_and_external_children()
        def deny_annotations(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)):
                path = os.fsdecode(args[0])
                if "/TAO-Amodal/annotations/" in path or "/recovered_splits/" in path or path.endswith("/roles.json") or "/frames/test/" in path:
                    raise PermissionError("Physical predictor prohibits GT/role/Test input")
        sys.addaudithook(deny_annotations)
        import numpy as np
        import torch
        from PIL import Image
        from src.trackocd_core.masa_native import build_frozen_model, infer_current_frame
        torch.set_num_threads(1); torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        model, proof = build_frozen_model()
        if (json_digest(proof["native_model_config"]) != config["native_model_config_sha256"]
                or proof["checkpoint_sha256"] != config["native_checkpoint_sha256"]
                or any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values())):
            raise ValueError("Frozen state/configuration identity or finiteness changed")
        initial_state = frozen_state_digest(model)
        result.update(config_sha256=digest, frozen_model=proof, initial_frozen_state_sha256=initial_state)
        torch.cuda.set_per_process_memory_fraction(min(1., config["limits"]["max_gpu_bytes_per_worker"] / torch.cuda.get_device_properties(0).total_memory))
        model.to("cuda:0"); torch.cuda.reset_peak_memory_stats()
        for position, video in enumerate(plan["videos"]):
            if position % count != index:
                continue
            if completed_video(RUN, video, digest) is not None:
                result["reused_complete_videos"] += 1
                continue
            model.tracker.reset()
            arrays = {"image_id": [], "frame_index": [], "frame_offsets": [0], "track_id": [], "boxes": [], "score": [],
                      "det_offsets": [0], "det_boxes": [], "det_score": []}
            counts = {"frames": 0, "roi_cap50_frames": 0, "empty_detection_frames": 0, "empty_tracking_frames": 0}
            for ordinal, image in enumerate(video["images"]):
                path = val_image(FRAMES, image["image_path"])
                if path.stat().st_size != image["image_bytes"] or sha256_file(path) != image["image_sha256"]:
                    raise ValueError("Registered current image changed; do not replace")
                result.update(current_video_id=video["video_id"], current_ordinal=ordinal,
                              heartbeat_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                result["actual_forwards_started_this_attempt"] += 1
                atomic_json(receipt, result)
                with Image.open(path) as raw:
                    pixels = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
                detections, tracks = infer_current_frame(model, pixels, ordinal, "cuda:0")
                torch.cuda.synchronize()
                snapshot_instances(detections, False); snapshot_instances(tracks, True)
                result["actual_forwards_completed_this_attempt"] += 1
                counts["frames"] += 1; counts["roi_cap50_frames"] += len(detections) == 50
                counts["empty_detection_frames"] += len(detections) == 0; counts["empty_tracking_frames"] += len(tracks) == 0
                arrays["image_id"].append(image["image_id"]); arrays["frame_index"].append(image["frame_index"])
                arrays["track_id"].extend(tracks.instances_id.cpu().numpy().tolist())
                arrays["boxes"].extend(tracks.bboxes.cpu().numpy().tolist()); arrays["score"].extend(tracks.scores.cpu().numpy().tolist())
                arrays["frame_offsets"].append(len(arrays["track_id"]))
                arrays["det_boxes"].extend(detections.bboxes.cpu().numpy().tolist()); arrays["det_score"].extend(detections.scores.cpu().numpy().tolist())
                arrays["det_offsets"].append(len(arrays["det_score"]))
                if torch.cuda.max_memory_reserved() > config["limits"]["max_gpu_bytes_per_worker"]:
                    raise RuntimeError("Owned GPU allocation ceiling")
            arrays = {k: np.asarray(v, dtype=np.float32 if k in {"boxes", "score", "det_boxes", "det_score"} else np.int64) for k, v in arrays.items()}
            arrays["boxes"] = arrays["boxes"].reshape(-1, 4); arrays["det_boxes"] = arrays["det_boxes"].reshape(-1, 4)
            if frozen_state_digest(model) != initial_state or any(p.requires_grad for p in model.parameters()) or model.training:
                raise ValueError("Frozen tensor state changed; no completion marker")
            counts["frozen_state_unchanged"] = True
            seal_video(RUN, video, arrays, digest, attempt + f"w{index}", counts)
            result["completed_video_ids_this_attempt"].append(video["video_id"])
            atomic_json(receipt, result)
            if len(result["completed_video_ids_this_attempt"]) % 10 == 0:
                print(json.dumps({"worker": index, "completed_videos": len(result["completed_video_ids_this_attempt"]),
                                  "forwards": result["actual_forwards_completed_this_attempt"]}), flush=True)
        result.update(status="COMPLETE_WORKER_PARTITION", final_frozen_state_sha256=frozen_state_digest(model),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        if result["final_frozen_state_sha256"] != initial_state:
            raise ValueError("Final frozen state changed")
    except Exception as exc:
        result.update(status="BLOCKED_OWNED_PHYSICAL_WORKER", error=type(exc).__name__ + ": " + str(exc))
        import traceback; traceback.print_exc()
    result.update(elapsed_seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(receipt, result)
    return 0 if result["status"] == "COMPLETE_WORKER_PARTITION" else 1


def candidate_roots() -> list[Path]:
    install = json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_install.json").read_text())
    roots = [ENV, RUN, PRIVATE, ROOT / "outputs/trackocd_core/physical/masa_val4_first16", ROOT / "checkpoints/masa_sam_candidate",
             ROOT / "outputs/trackocd_core/audit/frontend_alternatives_source", ROOT / "outputs/trackocd_core/audit/masa_candidate_source",
             ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution"]
    roots.extend((ROOT / "outputs/trackocd_core/audit").glob("masa-native-smoke-*"))
    roots.extend(ROOT / p for p in [install["task_temporary_directory"], *install.get("retained_temporary_directories", [])])
    return roots


def select_idle_gpus(limit: int):
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10)
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=10).split())
    selected = []
    for line in query.splitlines():
        gpu, used, free, util = [s.strip() for s in line.split(",")]
        if gpu not in occupied and int(used) < 100 and int(free) >= 9216 and int(util) == 0:
            selected.append(gpu)
    if not selected:
        raise RuntimeError("Resource wait: no freshly idle GPU with headroom")
    return selected[:limit], query


def parent(commit: str, resume: bool) -> int:
    from scripts.trackocd_core.install_masa_runtime import allocated_bytes
    config, plan, digest = load_plan()
    frozen = subprocess.check_output(["git", "show", commit + ":configs/trackocd_core/masa_physical_qualification.json"], cwd=ROOT)
    delivery = json.loads((ROOT / "outputs/trackocd_core/audit/masa_qualification_preregistration_delivery.json").read_text())
    if hashlib.sha256(frozen).hexdigest() != digest or delivery["commit"] != commit or not delivery["remote_verified"] or delivery["config_sha256"] != digest:
        raise ValueError("Remote-verified committed all-Val preregistration required")
    for relative in ("scripts/trackocd_core/run_masa_physical_qualification.py", "src/trackocd_core/physical_qualification.py", "src/trackocd_core/masa_native.py"):
        if subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT) != (ROOT / relative).read_bytes():
            raise ValueError("Committed inference implementation changed")
    if RUN.exists() and not resume:
        raise ValueError("Preserve run; explicit resume of verified completed videos required")
    if RUN.is_symlink() or not RUN.resolve().is_relative_to((ROOT / "outputs/trackocd_core/physical").resolve()):
        raise ValueError("Unexpected owned output link")
    existing = RUN.exists()
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # No historical PID or external process manipulation.
        owner = RUN / "run_owner.json"
        if existing:
            if not owner.is_file() or owner.is_symlink() or json.loads(owner.read_text()).get("config_sha256") != digest:
                raise ValueError("Existing directory has no matching task-owned provenance")
        else:
            atomic_json(owner, {"schema_version": "trackocd.core.masa_full_val_run_owner.v1", "config_sha256": digest,
                                "private_plan_sha256": config["private_plan_sha256"], "preregistration_commit": commit})
        return supervise(commit, config, plan, digest, allocated_bytes)


def supervise(commit, config, plan, digest, allocated_bytes):
    if (RUN / "prediction_manifest.json").exists():
        old = json.loads((RUN / "prediction_manifest.json").read_text())
        if old["status"] == "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM":
            raise ValueError("Entire stream already complete; evaluate, never repeat inference")
    before = sum(completed_video(RUN, v, digest) is not None for v in plan["videos"])
    limits = config["limits"]
    def resources(processes):
        mem = dict(s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())
        ram_fraction = int(mem["MemAvailable"].split()[0]) / int(mem["MemTotal"].split()[0])
        stat = os.statvfs(ROOT)
        rss = [owned_process_tree_rss(p.pid) if p.poll() is None else 0 for p in processes]
        current = {"candidate_allocated_bytes": allocated_bytes(candidate_roots()), "output_allocated_bytes": allocated_bytes([RUN]),
                   "conservative_repo_and_isolated_env_allocated_bytes": allocated_bytes([ROOT, ENV]), "worker_rss_bytes": rss,
                   "ram_available_fraction": ram_fraction, "ram_total_bytes": int(mem["MemTotal"].split()[0]) * 1024,
                   "disk_available_bytes": stat.f_bavail * stat.f_frsize}
        if (ram_fraction < limits["minimum_ram_headroom_fraction"] or current["disk_available_bytes"] < limits["minimum_disk_headroom_bytes"]
                or max(rss, default=0) > limits["max_host_rss_bytes_per_worker"]
                or current["candidate_allocated_bytes"] > limits["max_candidate_allocated_bytes"]
                or current["output_allocated_bytes"] > limits["max_output_allocated_bytes"]
                or current["conservative_repo_and_isolated_env_allocated_bytes"] > limits["overall_soft_bytes"]):
            raise RuntimeError("Registered physical resource guard: " + json.dumps(current))
        return current
    initial = resources([])
    selected, query = select_idle_gpus(limits["max_gpu_workers"])
    if initial["ram_available_fraction"] * initial["ram_total_bytes"] - len(selected) * limits["max_host_rss_bytes_per_worker"] < .25 * initial["ram_total_bytes"]:
        raise RuntimeError("Resource wait: insufficient projected RAM for selected workers")
    attempt = uuid.uuid4().hex
    directory = RUN / "attempts" / attempt; directory.mkdir(parents=True)
    (RUN / "shards").mkdir(exist_ok=True)
    result = {"schema_version": "trackocd.core.masa_full_val_supervisor.v1", "status": "RUNNING", "attempt": attempt,
              "preregistration_commit": commit, "remote_preregistration_verified": True, "config_sha256": digest,
              "supervisor_pid": os.getpid(), "initial_reused_complete_videos": before, "fresh_gpu_query": query,
              "selected_gpu_uuids": selected, "resources_initial": initial, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    processes, logs, started = [], [], time.monotonic()
    peak_candidate = peak_output = peak_rss = 0
    atomic_json(directory / "supervisor.json", result)
    atomic_json(RUN / "prediction_manifest.json", {**result, "videos": config["videos"], "images": config["images"]})
    print(json.dumps({"status": "START_FROZEN_M1_FULL_VAL_PHYSICAL_AUDIT", "workers": len(selected), "reused_videos": before, "attempt": attempt}), flush=True)
    try:
        for index, gpu in enumerate(selected):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR=str(directory / f"matplotlib_{index}"),
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
            log = (directory / f"worker_{index}.log").open("xb"); logs.append(log)
            processes.append(subprocess.Popen([str(ENV / "bin/python"), str(Path(__file__).resolve()), "--worker", "--attempt", attempt,
                                              "--worker-index", str(index), "--worker-count", str(len(selected))], cwd=ROOT, env=env,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        last_check = 0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("Owned partition failed; preserve all completed and partial attempts")
            if time.monotonic() - last_check >= 15:
                current = resources(processes); last_check = time.monotonic()
                peak_candidate = max(peak_candidate, current["candidate_allocated_bytes"])
                peak_output = max(peak_output, current["output_allocated_bytes"])
                peak_rss = max(peak_rss, sum(current["worker_rss_bytes"]))
                progress = []
                for index in range(len(processes)):
                    receipt = directory / f"worker_{index}.json"
                    if receipt.exists():
                        data = json.loads(receipt.read_text())
                        progress.append({k: data[k] for k in ("status", "actual_forwards_completed_this_attempt", "completed_video_ids_this_attempt", "reused_complete_videos")})
                result.update(resources_latest=current, worker_progress=progress, elapsed_seconds=time.monotonic() - started,
                              heartbeat_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                atomic_json(directory / "supervisor.json", result)
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("Owned worker failed")
        records = [completed_video(RUN, v, digest) for v in plan["videos"]]
        if any(r is None for r in records) or sum(r["images"] for r in records) != config["images"]:
            raise ValueError("Missing video/frame; full universe cannot be shrunk")
        workers = [json.loads((directory / f"worker_{i}.json").read_text()) for i in range(len(processes))]
        final = {"schema_version": "trackocd.core.masa_full_val_prediction.v1", "status": "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM",
                 "config_sha256": digest, "private_plan_sha256": config["private_plan_sha256"], "preregistration_commit": commit,
                 "videos": records, "images": config["images"], "total_prediction_rows": sum(r["prediction_rows"] for r in records),
                 "total_detection_rows": sum(r["detection_rows"] for r in records), "primary_freeze_permitted": False,
                 "gt_or_role_model_input": False, "training": False, "test_access": False, "worker_receipts": workers,
                 "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(RUN / "prediction_manifest.json", final)
        result["status"] = "COMPLETE_SUPERVISED_FULL_VAL_PREDICTION"
    except BaseException as exc:
        result.update(status="BLOCKED_OWNED_FULL_VAL_PREDICTION", error=type(exc).__name__ + ": " + str(exc))
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)  # Only children created in this exact live supervisor call.
        for process in processes:
            if process.poll() is None:
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
        atomic_json(RUN / "prediction_manifest.json", {**result, "videos": config["videos"], "images": config["images"]})
    finally:
        for log in logs: log.close()
        result.update(worker_returncodes=[p.returncode for p in processes], elapsed_seconds=time.monotonic() - started,
                      peak_candidate_allocated_bytes=peak_candidate, peak_output_allocated_bytes=peak_output,
                      peak_worker_tree_rss_bytes=peak_rss, completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(directory / "supervisor.json", result)
    print(json.dumps({k: result[k] for k in ("status", "attempt", "elapsed_seconds", "worker_returncodes", "peak_candidate_allocated_bytes")}), flush=True)
    return 0 if result["status"] == "COMPLETE_SUPERVISED_FULL_VAL_PREDICTION" else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--execute", action="store_true"); p.add_argument("--resume", action="store_true")
    p.add_argument("--preregistration-commit"); p.add_argument("--worker", action="store_true"); p.add_argument("--attempt")
    p.add_argument("--worker-index", type=int); p.add_argument("--worker-count", type=int)
    a = p.parse_args()
    if a.worker:
        if not a.attempt or len(a.attempt) != 32 or any(c not in "0123456789abcdef" for c in a.attempt):
            raise SystemExit("Owned hexadecimal attempt required")
        raise SystemExit(worker(a.attempt, a.worker_index, a.worker_count))
    if not a.execute or not a.preregistration_commit:
        raise SystemExit("Explicit execute and remote-verified preregistration required")
    raise SystemExit(parent(a.preregistration_commit, a.resume))
