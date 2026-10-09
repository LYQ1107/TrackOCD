#!/usr/bin/env python3
"""Run the pinned OVTR-native route on frozen TAO Test.

OVTR's normal TAO serializer omits the physical lineage needed by TrackOCD
and mixes category-aware fields into its result.  The pinned Phase75A capture
hook exports the physical tracker state before serialization; this wrapper
converts that lineage to a category-free TAO stream and registers it only
after the immutable Val freeze.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OVTR_ROOT = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"
OVTR_PYTHON = Path(os.environ.get("TRACKOCD_OVTR_PYTHON", "/home/lwr/anaconda3/envs/ovtr/bin/python"))
CHECKPOINT = ROOT / "data/iclr27_phase14b/checkpoints/ovtr_5_frame.pth"
CAPTURE = ROOT / "scripts/iclr27_phase75a/ovtr_native_eval.py"
CONFIG = OVTR_ROOT / "config/ovtr_5_frame_test.py"
TEST_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json"
)
FRAMES_ROOT = ROOT / "data/raw/tao/frames"
OUTPUT_TARGET = Path("/data2/usr_for_deadline/trackocd_v2/project_outputs")


class ResourceWait(RuntimeError):
    """The route must be retried after the machine safety gate clears."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_resource_gate(gpu_index: int) -> None:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(rest.split()[0])
    if values.get("MemTotal") is None or values.get("MemAvailable", 0) < int(values["MemTotal"] * 0.25):
        raise ResourceWait("MemAvailable is below the 25% safety floor")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    selected_uuid = None
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(gpu_index):
            selected_uuid = fields[1]
            break
    if selected_uuid is None:
        raise ResourceWait(f"GPU index {gpu_index} is unavailable")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if any(line.strip().split(",", 1)[0].strip() == selected_uuid for line in apps.stdout.splitlines() if line.strip()):
        raise ResourceWait(f"GPU index {gpu_index} has an existing compute application")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _marker_path() -> Path:
    return OUTPUT_TARGET / "audit/frontend_ovtr_test.launched"


def _claim(command: list[str], run_root: Path, gpu_index: int) -> Path:
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid OVTR Test launch marker: {marker}") from exc
        if _pid_alive(pid):
            raise RuntimeError(f"OVTR Test route already owned by live PID {pid}")
        marker.unlink()
    _atomic_json(marker, {
        "pid": os.getpid(),
        "started_utc": _now(),
        "frontend": "OVTR-native",
        "split": "test",
        "gpu_index": int(gpu_index),
        "run_root": str(run_root.resolve()),
        "command": command,
    })
    return marker


def _combine_lineage(source: Path, target: Path) -> dict[str, int]:
    """Convert captured OVTR rows into one category-free TAO JSON array."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    rows = 0
    videos: set[int] = set()
    tracks: set[tuple[int, str]] = set()
    try:
        with source.open(encoding="utf-8") as input_handle, temporary.open("w", encoding="utf-8") as output_handle:
            output_handle.write("[")
            first = True
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"OVTR lineage line {line_number} is not an object")
                box = item.get("bbox_xyxy")
                if box is None:
                    # Capture emits termination records without an image
                    # observation; the physical stream contains observations
                    # only, while lifecycle evidence remains in the source.
                    continue
                if len(box) != 4:
                    raise ValueError(f"OVTR lineage line {line_number} has malformed bbox")
                values = [float(value) for value in box]
                score = float(item.get("base_score"))
                video_id = int(item["video_id"])
                image_id = int(item["image_id"])
                physical_id = str(item["physical_track_id"])
                if video_id < 0 or image_id < 0 or not all(math.isfinite(value) for value in values + [score]):
                    raise ValueError(f"OVTR lineage line {line_number} has invalid identity/box/score")
                row = {
                    "video_id": video_id,
                    "image_id": image_id,
                    "track_id": physical_id,
                    "bbox": [values[0], values[1], values[2] - values[0], values[3] - values[1]],
                    "score": score,
                    # OVTR capture exposes contiguous detector labels rather
                    # than canonical TAO IDs.  Keep the physical route
                    # category-free instead of fabricating a TETA label.
                    "category_id": 1,
                }
                if not first:
                    output_handle.write(",")
                first = False
                output_handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
                rows += 1
                videos.add(video_id)
                tracks.add((video_id, physical_id))
            output_handle.write("]")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"rows": rows, "videos": len(videos), "tracks": len(tracks)}


def run(*, gpu_index: int, run_id: str) -> dict[str, Any]:
    from src.trackocd_v2.protocol import assert_test_semantic_access_allowed

    # Guard before opening the Test annotation or launching OVTR.  No Test
    # output can enter the selected frontend before the immutable freeze.
    assert_test_semantic_access_allowed(
        OUTPUT_TARGET / "audit/FINAL_FREEZE.json",
        "run frozen OVTR-native TAO Test frontend",
    )
    for path in (OVTR_PYTHON, CHECKPOINT, CAPTURE, CONFIG, TEST_ANNOTATION, FRAMES_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    _require_resource_gate(gpu_index)
    run_root = OUTPUT_TARGET / "frontend_native" / "ovtr" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    lineage = run_root / "native_lineage.jsonl"
    command = [
        str(OVTR_PYTHON),
        str(CAPTURE.resolve()),
        "--config_file", "./config/ovtr_5_frame_test.py",
        "--dataset_file", "lvis_generated_img_seqs",
        "--batch_size", "1",
        "--num_workers", "1",
        "--with_box_refine",
        "--two_stage",
        "--pretrained", str(CHECKPOINT.resolve()),
        "--score_mode", "base",
        "--score_thresh", "0.19", "0.19", "0.19", "0.19", "0.19", "0.19", "0.19",
        "--filter_score_thresh", "0.19", "0.19", "0.19", "0.19", "0.19", "0.19", "0.19",
        "--ious_thresh", "0.45", "0.45", "0.45", "0.45", "0.45", "0.45", "0.45",
        "--miss_tolerance", "5", "5", "5", "5", "5", "5", "5",
        "--maximum_quantity", "160",
        "--area_threshold", "1",
        "--output_dir", str((run_root / "output").resolve()),
        "--result_path_track", str((run_root / "teta_results").resolve()),
        "--eval", "track",
        "--native-out", str(lineage.resolve()),
        "--trackocd-split", "test",
        "--trackocd-test-annotation", str(TEST_ANNOTATION.resolve()),
        "--trackocd-test-image-root", str(FRAMES_ROOT.resolve()),
    ]
    marker = _claim(command, run_root, gpu_index)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONPATH"] = os.pathsep.join([str(OVTR_ROOT.resolve()), str(ROOT.resolve()), environment.get("PYTHONPATH", "")])
    log_path = run_root / "native_test.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(command, cwd=OVTR_ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"OVTR Test inference returned {process.returncode}; see {log_path}")
        if not lineage.is_file():
            raise FileNotFoundError(lineage)
        native_output = run_root / "tao_track.json"
        counts = _combine_lineage(lineage, native_output)
        normalize_command = [
            sys.executable,
            str((ROOT / "scripts/trackocd_v2/register_test_frontend_stream.py").resolve()),
            "--frontend", "OVTR-native",
            "--native-input", str(native_output.resolve()),
            "--input-format", "tao_json",
            "--annotation", str(TEST_ANNOTATION.resolve()),
            "--output-dir", str((OUTPUT_TARGET / "manifests/frontend_streams/ovtr/test").resolve()),
        ]
        normalization_log = run_root / "normalization.log"
        with normalization_log.open("w", encoding="utf-8") as log_handle:
            normalized = subprocess.run(normalize_command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if normalized.returncode != 0:
            raise RuntimeError(f"OVTR Test normalization returned {normalized.returncode}; see {normalization_log}")
        result = {
            "schema_version": "trackocd.v2.ovtr_native_test_run.v1",
            "status": "COMPLETE",
            "generated_utc": _now(),
            "frontend": "OVTR-native",
            "frontend_slug": "ovtr",
            "split": "test",
            "run_root": str(run_root.resolve()),
            "native_command": command,
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "config": str(CONFIG.resolve()),
            "native_lineage": str(lineage.resolve()),
            "native_lineage_sha256": _sha256(lineage),
            "native_output": str(native_output.resolve()),
            "native_output_sha256": _sha256(native_output),
            "native_counts": counts,
            "normalization_command": normalize_command,
            "normalization_audit": str((OUTPUT_TARGET / "audit/frontend_ovtr_test_normalization.json").resolve()),
            "test_semantic_accessed": False,
            "test_evaluation_unlocked": True,
        }
        _atomic_json(OUTPUT_TARGET / "audit/frontend_ovtr_native_test_run.json", result)
        marker.unlink()
        _atomic_json(OUTPUT_TARGET / "audit/frontend_ovtr_test.done", {
            "frontend": "OVTR-native",
            "split": "test",
            "finished_utc": _now(),
            "run_root": str(run_root.resolve()),
            "native_output_sha256": result["native_output_sha256"],
        })
        return result
    except Exception:
        # Keep the marker for bounded, explicit resume logic after a failure.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    OUTPUT_TARGET.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_TARGET / "audit/frontend_ovtr_native_test_run.json"
    try:
        result = run(gpu_index=args.gpu_index, run_id=args.run_id)
    except ResourceWait as exc:
        payload = {
            "schema_version": "trackocd.v2.ovtr_native_test_run.v1",
            "status": "WAITING_RESOURCE",
            "generated_utc": _now(),
            "frontend": "OVTR-native",
            "frontend_slug": "ovtr",
            "split": "test",
            "error": str(exc),
            "test_semantic_accessed": False,
        }
        _atomic_json(audit_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.ovtr_native_test_run.v1",
            "status": "FAILED_OVTR_NATIVE_TEST_RUN",
            "generated_utc": _now(),
            "frontend": "OVTR-native",
            "frontend_slug": "ovtr",
            "split": "test",
            "error": f"{type(exc).__name__}: {exc}",
            "test_semantic_accessed": False,
        }
        _atomic_json(audit_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
