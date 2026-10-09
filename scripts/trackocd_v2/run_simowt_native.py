#!/usr/bin/env python3
"""Run the pinned SimOWT/Q0 frontend on frozen TAO Test.

The existing SimOWT model writes one JSON array per processed image and has a
legacy hard-coded Val video lookup.  This wrapper keeps that implementation
read-only: it registers the canonical Test dataset, redirects only that
legacy lookup to the frozen Test annotation, then combines the per-image
native rows into one deterministic TAO array for the normalizer.
"""

from __future__ import annotations

import argparse
import builtins
import datetime as dt
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SIMOWT_ROOT = ROOT / "third_party/SimOWT"
SIMOWT_IDOL = SIMOWT_ROOT / "projects/IDOL"
SIMOWT_PYTHON = Path(os.environ.get("TRACKOCD_SIMOWT_PYTHON", "/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python"))
CONFIG = SIMOWT_IDOL / "configs/r50_eval.yaml"
CHECKPOINT = ROOT / "data/iclr27_phase14b/checkpoints/simowt_weight.pth"
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
    return OUTPUT_TARGET / "audit/frontend_simowt_test.launched"


def _claim(command: list[str], run_root: Path, gpu_index: int) -> Path:
    marker = _marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid SimOWT Test launch marker: {marker}") from exc
        if _pid_alive(pid):
            raise RuntimeError(f"SimOWT Test route already owned by live PID {pid}")
        marker.unlink()
    _atomic_json(marker, {
        "pid": os.getpid(),
        "started_utc": _now(),
        "frontend": "SimOWT/Q0",
        "split": "test",
        "gpu_index": int(gpu_index),
        "run_root": str(run_root.resolve()),
        "command": command,
    })
    return marker


def _combine_frame_outputs(frame_dir: Path, target: Path) -> dict[str, int]:
    """Combine SimOWT's per-image arrays without loading all rows at once."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    rows = 0
    videos: set[int] = set()
    tracks: set[tuple[int, str]] = set()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("[")
            first = True
            files = sorted(frame_dir.glob("*.json"), key=lambda path: (int(path.stem), path.name))
            for path in files:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, list):
                    raise ValueError(f"SimOWT output is not a per-image array: {path}")
                for row in value:
                    if not isinstance(row, dict):
                        raise ValueError(f"SimOWT output row is not an object: {path}")
                    required = {"video_id", "image_id", "track_id", "bbox", "score"}
                    missing = required - set(row)
                    if missing:
                        raise ValueError(f"SimOWT output row misses {sorted(missing)}: {path}")
                    if not first:
                        handle.write(",")
                    first = False
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
                    rows += 1
                    video_id = int(row["video_id"])
                    videos.add(video_id)
                    tracks.add((video_id, str(row["track_id"])))
            handle.write("]")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"rows": rows, "videos": len(videos), "tracks": len(tracks), "frame_files": len(files)}


def _worker(args: argparse.Namespace) -> int:
    """Execute SimOWT inside its pinned environment."""

    run_root = args.run_root.resolve()
    frame_dir = run_root / "frame_outputs"
    frame_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SIMOWT_OUTPUT_DIR"] = str(frame_dir)
    os.environ["DETECTRON2_DATASETS"] = str((SIMOWT_ROOT / "datasets").resolve())
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(SIMOWT_ROOT.resolve()), str(SIMOWT_IDOL.resolve()), str(ROOT.resolve()), os.environ.get("PYTHONPATH", "")]
    )
    os.chdir(SIMOWT_ROOT)
    sys.path[:0] = [str(SIMOWT_ROOT), str(SIMOWT_IDOL), str(ROOT)]

    # IDOL's model constructor still opens this exact legacy Val-only path to
    # obtain video names.  Redirect only that path in this child process; all
    # image and dataset reads continue through the pinned implementation.
    original_open = builtins.open
    legacy_path = os.path.abspath("./datasets/tao/annotations/val_split/all.json")

    def guarded_open(file: Any, *open_args: Any, **open_kwargs: Any) -> Any:
        try:
            candidate = os.path.abspath(os.fspath(file))
        except TypeError:
            candidate = ""
        if candidate == legacy_path:
            return original_open(TEST_ANNOTATION, *open_args, **open_kwargs)
        return original_open(file, *open_args, **open_kwargs)

    builtins.open = guarded_open
    try:
        from detectron2.projects.idol.data.datasets.ytvis import register_tao_instances  # noqa: WPS433

        register_tao_instances("tao_test", {}, str(TEST_ANNOTATION.resolve()), str(FRAMES_ROOT.resolve()))
        train_net = importlib.import_module("train_net")
        sys.argv = [
            str((SIMOWT_IDOL / "train_net.py").resolve()),
            "--config-file", str(CONFIG.resolve()),
            "--num-gpus", "1",
            "--eval-only",
            "MODEL.WEIGHTS", str(CHECKPOINT.resolve()),
            "DATASETS.TEST", "(\"tao_test\",)",
            "DATALOADER.NUM_WORKERS", "1",
            "OUTPUT_DIR", str((run_root / "detectron_output").resolve()),
        ]
        parsed = train_net.default_argument_parser().parse_args()
        train_net.main(parsed)
    finally:
        builtins.open = original_open
    return 0


def run(*, gpu_index: int, run_id: str) -> dict[str, Any]:
    from src.trackocd_v2.protocol import assert_test_semantic_access_allowed

    # Guard before checking/opening the Test annotation.  The native route is
    # only scheduled after FINAL_FREEZE and never contributes to selection.
    assert_test_semantic_access_allowed(
        OUTPUT_TARGET / "audit/FINAL_FREEZE.json",
        "run frozen SimOWT/Q0 TAO Test frontend",
    )
    for path in (SIMOWT_PYTHON, CONFIG, CHECKPOINT, TEST_ANNOTATION, FRAMES_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    _require_resource_gate(gpu_index)
    run_root = OUTPUT_TARGET / "frontend_native" / "simowt" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    command = [
        str(SIMOWT_PYTHON),
        str(Path(__file__).resolve()),
        "--worker",
        "--run-root", str(run_root.resolve()),
    ]
    marker = _claim(command, run_root, gpu_index)
    log_path = run_root / "native_test.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONPATH"] = os.pathsep.join([str(SIMOWT_ROOT.resolve()), str(SIMOWT_IDOL.resolve()), str(ROOT.resolve()), environment.get("PYTHONPATH", "")])
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(command, cwd=SIMOWT_ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"SimOWT Test inference returned {process.returncode}; see {log_path}")
        native_output = run_root / "tao_track.json"
        counts = _combine_frame_outputs(run_root / "frame_outputs", native_output)
        if not native_output.is_file():
            raise FileNotFoundError(native_output)
        normalize_command = [
            sys.executable,
            str((ROOT / "scripts/trackocd_v2/register_test_frontend_stream.py").resolve()),
            "--frontend", "SimOWT/Q0",
            "--native-input", str(native_output.resolve()),
            "--input-format", "tao_json",
            "--annotation", str(TEST_ANNOTATION.resolve()),
            "--output-dir", str((OUTPUT_TARGET / "manifests/frontend_streams/simowt/test").resolve()),
        ]
        normalization_log = run_root / "normalization.log"
        with normalization_log.open("w", encoding="utf-8") as log_handle:
            normalized = subprocess.run(normalize_command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if normalized.returncode != 0:
            raise RuntimeError(f"SimOWT Test normalization returned {normalized.returncode}; see {normalization_log}")
        result = {
            "schema_version": "trackocd.v2.simowt_native_test_run.v1",
            "status": "COMPLETE",
            "generated_utc": _now(),
            "frontend": "SimOWT/Q0",
            "frontend_slug": "simowt",
            "split": "test",
            "run_root": str(run_root.resolve()),
            "native_command": command,
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "config": str(CONFIG.resolve()),
            "native_output": str(native_output.resolve()),
            "native_output_sha256": _sha256(native_output),
            "native_counts": counts,
            "normalization_command": normalize_command,
            "normalization_audit": str((OUTPUT_TARGET / "audit/frontend_simowt_test_normalization.json").resolve()),
            "test_semantic_accessed": False,
            "test_evaluation_unlocked": True,
        }
        audit_path = OUTPUT_TARGET / "audit/frontend_simowt_native_test_run.json"
        _atomic_json(audit_path, result)
        marker.unlink()
        _atomic_json(OUTPUT_TARGET / "audit/frontend_simowt_test.done", {
            "frontend": "SimOWT/Q0",
            "split": "test",
            "finished_utc": _now(),
            "run_root": str(run_root.resolve()),
            "native_output_sha256": result["native_output_sha256"],
        })
        return result
    except Exception:
        # Keep the marker as resumable ownership evidence after a failed child.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", choices=("test",), default="test")
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    if args.gpu_index is None or not args.run_id:
        parser.error("--gpu-index and --run-id are required outside --worker mode")
    OUTPUT_TARGET.mkdir(parents=True, exist_ok=True)
    audit_path = OUTPUT_TARGET / "audit/frontend_simowt_native_test_run.json"
    try:
        result = run(gpu_index=args.gpu_index, run_id=args.run_id)
    except ResourceWait as exc:
        payload = {
            "schema_version": "trackocd.v2.simowt_native_test_run.v1",
            "status": "WAITING_RESOURCE",
            "generated_utc": _now(),
            "frontend": "SimOWT/Q0",
            "frontend_slug": "simowt",
            "split": "test",
            "error": str(exc),
            "test_semantic_accessed": False,
        }
        _atomic_json(audit_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.simowt_native_test_run.v1",
            "status": "FAILED_SIMOWT_NATIVE_TEST_RUN",
            "generated_utc": _now(),
            "frontend": "SimOWT/Q0",
            "frontend_slug": "simowt",
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
