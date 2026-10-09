#!/usr/bin/env python3
"""Bounded frozen native proposal/causal-track smoke; never a Val benchmark.

The parent selects resources and passes only four Train image paths to a fresh,
network-denied isolated worker. Two chronological replays (changed future in the
second) consume exactly eight real-image forwards, with no GT/runtime metadata.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file

PLAN = ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json"
DATASET = Path("/data3/liuyeqiang/TAO-Amodal/frames")
ISOLATED = Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke/bin/python")


def validate_train_images(paths: list[str]) -> list[Path]:
    if len(paths) != 4:
        raise ValueError("Exactly four distinct Train images for eight forwards required")
    validated = []
    for value in paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "train":
            raise ValueError("Train-only relative image input required")
        absolute = DATASET / relative
        if not absolute.resolve().is_relative_to((DATASET / "train").resolve()) or not absolute.is_file():
            raise ValueError("Image escapes local Train tree or is missing")
        validated.append(absolute)
    if len(set(validated)) != 4 or len({p.parent for p in validated}) != 1:
        raise ValueError("One video, four unique images required")
    return validated


def select_private_image_paths() -> dict:
    """Reuse an existing Train-only selection; never read annotation files."""
    path = ROOT / "outputs/trackocd_core/pilots/gt_train_known/selection_plan.json"
    registered = json.loads((ROOT / "outputs/trackocd_core/audit/gt_pilot_preregistration.json").read_text())
    if sha256_file(path) != registered["private_plan"]["sha256"]:
        raise ValueError("Existing private image-selection plan changed")
    row = json.loads(path.read_text())["rows"][0]
    observations = row["observations"][:4]
    frame_ids = [o["frame_id"] for o in observations]
    if len(frame_ids) != 4 or any(b <= a for a, b in zip(frame_ids, frame_ids[1:])):
        raise ValueError("Private image selection is not chronological")
    paths = [o["image_path"] for o in observations]
    validate_train_images(paths)
    # Drop all GT/semantic fields before serializing the worker's input.
    return {"image_paths": paths, "source_plan_sha256": sha256_file(path),
            "selection_bias": "Existing long Known-GT pilot selection; engineering only, not coverage estimation",
            "temporal_spacing": "Subsampled chronological frames; ordinal units, not original frame indices"}


def deny_network_and_external_children() -> None:
    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "subprocess.Popen", "os.system"}:
            raise PermissionError("Frozen smoke prohibits network/external child execution: " + event)
    sys.addaudithook(audit)


def snapshot_instances(instances, tracks: bool) -> dict:
    import numpy as np
    fields = ["bboxes", "scores", "labels"] + (["instances_id"] if tracks else [])
    arrays = {name: getattr(instances, name).detach().cpu().numpy() for name in fields}
    boxes, scores, labels = arrays["bboxes"], arrays["scores"], arrays["labels"]
    if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any((scores < 0) | (scores > 1))
            or np.any(labels != 0)):
        raise ValueError("Non-finite/degenerate/non-anonymous native output; no repair or filtering")
    digest = hashlib.sha256()
    for name, array in arrays.items():
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    result = {"count": int(len(scores)), "array_sha256": digest.hexdigest(),
              "score_min": float(scores.min()) if len(scores) else None,
              "score_max": float(scores.max()) if len(scores) else None,
              "score_mean": float(scores.mean()) if len(scores) else None,
              "nonfinite_or_degenerate": 0, "export_category_contract": "anonymous foreground 1 (native tensor label 0)"}
    if tracks:
        result["unique_ids"] = sorted(int(v) for v in np.unique(arrays["instances_id"]))
    return result


def worker(run: Path) -> int:
    started = time.monotonic()
    destination = run / "result.json"
    result = {"schema_version": "trackocd.core.masa_native_smoke.v1", "status": "RUNNING",
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "replays": [], "real_image_forwards_started": 0,
              "boundary": {"physical_training": False, "gt_runtime_input": False, "semantic_runtime_input": False,
                           "val_or_test_access": False, "future_runtime_input": False, "offline_postprocessing": False,
                           "extra_weight_download": False, "foreign_process_interference": False}}
    try:
        config = json.loads((run / "worker_input.json").read_text())
        images = validate_train_images(config["image_paths"])
        deny_network_and_external_children()
        import numpy as np
        import torch
        import torchvision
        import mmcv
        import mmengine
        import mmdet
        from mmcv.ops import nms, roi_align
        from mmengine.structures import InstanceData
        from PIL import Image
        from src.trackocd_core.masa_native import build_frozen_model, prepare_current_image, infer_current_frame
        torch.set_num_threads(1)
        torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        result["runtime_versions"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                      "numpy": np.__version__, "mmcv": mmcv.__version__,
                                      "mmengine": mmengine.__version__, "mmdet": mmdet.__version__}
        result["network_and_external_children_denied"] = True
        boxes = torch.tensor([[0., 0., 2., 2.], [.1, .1, 2.1, 2.1]])
        _, keep = nms(boxes, torch.tensor([.9, .8]), .5)
        roi = roi_align(torch.ones(1, 1, 4, 4), torch.tensor([[0., 0., 0., 3., 3.]]), (2, 2))
        if keep.tolist() != [0] or not torch.isfinite(roi).all():
            raise ValueError("Binary CPU NMS/RoIAlign fixture failed")
        result["binary_cpu_nms_roi_align_pass"] = True
        model, proof = build_frozen_model()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Non-finite frozen state tensor")
        result["frozen_model"] = proof
        result["all_state_tensors_finite"] = True
        pixels, sample = prepare_current_image(np.zeros((17, 31, 3), dtype=np.uint8), "cpu")
        if tuple(pixels.shape) != (1, 3, 1024, 1024) or not torch.isfinite(pixels).all():
            raise ValueError("Synthetic preprocessing fixture failed")
        sample.set_metainfo({"frame_id": 0})
        sample.pred_instances = InstanceData(bboxes=torch.empty(0, 4), scores=torch.empty(0), labels=torch.empty(0, dtype=torch.long))
        empty = model.tracker.track(model=model, img=None, feats=None, data_sample=sample, rescale=True, with_segm=False)
        if len(empty) != 0:
            raise ValueError("Empty proposal contract failed")
        result["synthetic_shape_and_empty_proposal_pass"] = True
        result["input"] = {"unique_train_images": 4, "real_image_forward_ceiling": 8,
                           "image_sha256": [sha256_file(p) for p in images],
                           "source_plan_sha256": config["source_plan_sha256"],
                           "selection_bias": config["selection_bias"], "temporal_spacing": config["temporal_spacing"]}
        atomic_json(destination, result)
        gpu_bytes = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., 8 * 1024**3 / gpu_bytes), device=0)
        model.to("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        def check_limits():
            if time.monotonic() - started > 600 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 4 * 1024**3:
                raise RuntimeError("Registered smoke wall-time/host memory stop")
            if torch.cuda.max_memory_reserved() > 8 * 1024**3:
                raise RuntimeError("Registered GPU memory stop")
        for replay in range(2):
            model.tracker.reset()
            rows = []
            result["replays"].append({"changed_future_pixels": bool(replay), "frames": rows})
            for ordinal, path in enumerate(images):
                check_limits()
                with Image.open(path) as raw:
                    current = np.asarray(raw.convert("RGB"), dtype=np.uint8).copy()
                if replay and ordinal >= 2:
                    current = 255 - current
                frame_started = time.monotonic()
                result["real_image_forwards_started"] += 1
                atomic_json(destination, result)
                detections, tracks = infer_current_frame(model, current, ordinal, "cuda:0")
                torch.cuda.synchronize()
                rows.append({"ordinal": ordinal, "current_pixels_sha256": hashlib.sha256(current.tobytes()).hexdigest(),
                             "detections": snapshot_instances(detections, False), "tracks": snapshot_instances(tracks, True),
                             "elapsed_seconds": time.monotonic() - frame_started})
                check_limits()
                atomic_json(destination, result)
                print(json.dumps({"replay": replay, "ordinal": ordinal, "detections": len(detections), "tracks": len(tracks)}), flush=True)
        first, changed = [r["frames"] for r in result["replays"]]
        comparisons = [{"ordinal": i, "same_current_pixels": first[i]["current_pixels_sha256"] == changed[i]["current_pixels_sha256"],
                        "exact_detection_and_track_arrays_equal": all(first[i][k]["array_sha256"] == changed[i][k]["array_sha256"]
                                                                       for k in ("detections", "tracks"))} for i in (0, 1)]
        result["prefix_invariance"] = {"comparisons": comparisons,
                                       "future_pixels_actually_changed": all(first[i]["current_pixels_sha256"] != changed[i]["current_pixels_sha256"] for i in (2, 3)),
                                       "proof_scope": "Two native sequential replays only, not a formal full-Val causality certificate"}
        if not result["prefix_invariance"]["future_pixels_actually_changed"] or not all(c["same_current_pixels"] and c["exact_detection_and_track_arrays_equal"] for c in comparisons):
            raise ValueError("Changed-future prefix invariance failed; no tolerance or tracker repair")
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["status"] = "PASS_BOUNDED_FROZEN_NATIVE_SMOKE_NOT_M1_QUALIFICATION"
    except Exception as exc:
        result["status"] = "BLOCKED_NATIVE_SMOKE_ENGINEERING"
        result["error"] = type(exc).__name__ + ": " + str(exc)
        import traceback
        traceback.print_exc()
    result.update(elapsed_seconds=time.monotonic() - started,
                  peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(destination, result)
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds", "peak_host_rss_bytes")}), flush=True)
    return 0 if result["status"].startswith("PASS") else 1


def owned_process_tree_rss(pid: int) -> int:
    total, pending, seen = 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            rss = next(line.split()[1] for line in status.splitlines() if line.startswith("VmRSS:"))
            total += int(rss) * 1024
            pending.extend(int(v) for v in Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except (FileNotFoundError, StopIteration, ProcessLookupError):
            continue
    return total


def parent() -> int:
    from scripts.trackocd_core.install_masa_runtime import allocated_bytes, distribution_receipt, matches_install_provenance
    install_path = ROOT / "outputs/trackocd_core/audit/masa_runtime_install.json"
    install = json.loads(install_path.read_text())
    previous_path = ROOT / "outputs/trackocd_core/audit/masa_native_smoke.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if previous and (previous["status"].startswith("PASS") or previous.get("real_image_forwards_started", 0)
                     or any(r.get("frames") for r in previous.get("replays", []))):
        raise ValueError("Preserve completed/partially executed real-image smoke; no automatic rerun or budget expansion")
    if install["status"] != "PASS_HASH_LOCKED_ISOLATED_INSTALL_NOT_MODEL_COMPATIBILITY" or len(install["completed_packages"]) != 48:
        raise ValueError("Complete verified isolated install required; no automatic install")
    lock = ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution/pylock.toml"
    if install["dependency_lock_sha256"] != sha256_file(lock):
        raise ValueError("Installed dependency lock changed")
    wheels = json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_wheel_inventory.json").read_text())["wheels"]
    installed = distribution_receipt(ISOLATED)
    for record in wheels:
        name = record["name"].lower().replace("_", "-")
        if not matches_install_provenance(installed.get(name, {}), record, install["verified_install_operations"].get(name)):
            raise ValueError("Owned runtime distribution changed: " + name)
    selection = select_private_image_paths()
    occupied = set(subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], text=True, timeout=10).split())
    available = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=10)
    selected = None
    for line in available.splitlines():
        uuid, used, free, utilization = [v.strip() for v in line.split(",")]
        if uuid not in occupied and int(used) < 100 and int(free) >= 9216 and int(utilization) == 0:
            selected = uuid
            break
    if not selected:
        raise RuntimeError("Resource wait: no freshly idle GPU with registered headroom")
    run = Path(tempfile.mkdtemp(prefix="masa-native-smoke-", dir=ROOT / "outputs/trackocd_core/audit"))
    if previous:
        atomic_json(run / "previous_native_smoke_receipt.json", previous)
    atomic_json(run / "worker_input.json", selection)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=selected, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLCONFIGDIR=str(run / "matplotlib"), TORCH_HOME=str(run / "torch"), HF_HOME=str(run / "hf"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    roots = [ISOLATED.parent.parent, ROOT / "checkpoints/masa_sam_candidate", ROOT / "outputs/trackocd_core/audit/masa_candidate_source",
             ROOT / "outputs/trackocd_core/audit/frontend_alternatives_source", ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution", run]
    roots.extend(ROOT / p for p in [install["task_temporary_directory"], *install.get("retained_temporary_directories", [])])
    started, peak_rss, peak_storage, peak_device_mib = time.monotonic(), 0, 0, 0
    log = run / "worker.log"
    print(json.dumps({"status": "START_BOUNDED_SMOKE", "directory": str(run.relative_to(ROOT)), "gpu_uuid": selected}), flush=True)
    error = None
    with log.open("wb") as output:
        process = subprocess.Popen([str(ISOLATED), str(Path(__file__).resolve()), "--worker", str(run)],
                                   cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                peak_rss = max(peak_rss, owned_process_tree_rss(process.pid))
                peak_storage = max(peak_storage, allocated_bytes(roots))
                memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                if int(time.monotonic() - started) % 3 == 0:
                    device_mib = int(subprocess.check_output(["nvidia-smi", "--id=" + selected,
                        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
                    peak_device_mib = max(peak_device_mib, device_mib)
                    if device_mib > 8192:
                        raise RuntimeError("Registered total device-memory stop (includes context/other new activity)")
                if (time.monotonic() - started > 600 or peak_rss > 4 * 1024**3 or peak_storage > 8 * 1024**3
                        or int(memory["MemAvailable"].split()[0]) < .25 * int(memory["MemTotal"].split()[0])
                        or allocated_bytes([run]) > 10 * 1024**2):
                    raise RuntimeError("Registered time/RAM/storage/output guard stop")
                time.sleep(1)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)  # This freshly spawned owned worker only.
                process.wait(timeout=30)
    result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {"status": "BLOCKED_NATIVE_WORKER_NO_RESULT"}
    if previous:
        history = list(previous.get("prior_engineering_attempts", []))
        history.append({k: previous[k] for k in ("status", "error", "elapsed_seconds", "peak_host_rss_bytes", "real_image_forwards_started") if k in previous})
        result["prior_engineering_attempts"] = history
    if error:
        result.update(status="BLOCKED_NATIVE_SMOKE_RESOURCE_GUARD", parent_error=error)
    result["supervisor"] = {"gpu_uuid": selected, "fresh_gpu_query": available, "worker_returncode": process.returncode,
                            "peak_owned_process_tree_rss_bytes": peak_rss, "peak_candidate_allocated_bytes": peak_storage,
                            "peak_selected_device_memory_mib": peak_device_mib,
                            "install_receipt_sha256": sha256_file(install_path), "preregistered_plan_sha256": sha256_file(PLAN),
                            "elapsed_seconds": time.monotonic() - started, "private_directory": str(run.relative_to(ROOT))}
    atomic_json(run / "result.json", result)
    # Small semantic-free receipt only; worker_input and raw images stay private.
    atomic_json(ROOT / "outputs/trackocd_core/audit/masa_native_smoke.json", result)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["supervisor"]["elapsed_seconds"],
                      "worker_returncode": process.returncode, "peak_candidate_allocated_bytes": peak_storage}), flush=True)
    return 0 if process.returncode == 0 and result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.worker.parent != ROOT / "outputs/trackocd_core/audit" or not args.worker.name.startswith("masa-native-smoke-"):
            raise SystemExit("Unexpected task-owned worker directory")
        raise SystemExit(worker(args.worker))
    if not args.execute:
        raise SystemExit("Explicit --execute required; no model or dataset execution")
    raise SystemExit(parent())
