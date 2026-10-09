#!/usr/bin/env python3
"""Run one bounded COVTrack Val frontend route and normalize its output.

The official COVTrack implementation remains read-only under ``third_party``.
This wrapper selects the pinned checkpoint/config, runs one visible GPU with a
single test process, and then sends the resulting native TAO rows through the
TrackOCD v2 normalizer.  The ``confused_features`` override is the registered
native/no-semantic ablation; no TrackOCD category or text field is consumed by
the normalized physical stream.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
COVTRACK_ROOT = ROOT / "third_party/research_refs_phase4n/COVTrack"
_DEFAULT_COVTRACK_PYTHON = Path("/home/lwr/anaconda3/envs/ovtrack_mmdet2/bin/python")
if not _DEFAULT_COVTRACK_PYTHON.is_file():
    _DEFAULT_COVTRACK_PYTHON = Path("/home/lwr/anaconda3/envs/ovtrack/bin/python")
COVTRACK_PYTHON = Path(os.environ.get("TRACKOCD_COVTRACK_PYTHON", str(_DEFAULT_COVTRACK_PYTHON)))
COVTRACK_LAUNCHER = ROOT / "scripts/trackocd_v2/run_covtrack_legacy_test.py"
CHECKPOINT = ROOT / "data/iclr27_phase14b/checkpoints/covtrack_ctao_public.pth"
CONFIG = COVTRACK_ROOT / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
OUTPUT_TARGET = Path("/data2/usr_for_deadline/trackocd_v2/project_outputs")
FRONTENDS = {
    "covtrack_native": {"name": "COVTrack-native", "confused_features": True},
    "covtrack_nosem": {"name": "COVTrack-NoSemantic", "confused_features": False},
}
TEST_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json"
)


class ResourceWait(RuntimeError):
    """The route must be retried after the machine safety gate clears."""


def _require_resource_gate(gpu_index: int) -> None:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            meminfo[key] = int(value.split()[0])
    if meminfo.get("MemTotal") is None or meminfo.get("MemAvailable", 0) < int(meminfo["MemTotal"] * 0.25):
        raise ResourceWait("MemAvailable is below the 25% safety floor")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
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
        capture_output=True, text=True, check=False,
    )
    if any(line.strip().split(",", 1)[0].strip() == selected_uuid for line in apps.stdout.splitlines() if line.strip()):
        raise ResourceWait(f"GPU index {gpu_index} has an existing compute application")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _marker_path(slug: str, split: str) -> Path:
    suffix = "" if split == "val" else "_test"
    return OUTPUT_TARGET / "audit" / f"frontend_{slug}{suffix}.launched"


def _claim(slug: str, split: str, command: list[str], run_root: Path, gpu_index: int) -> Path:
    marker = _marker_path(slug, split)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid COVTrack launch marker: {marker}") from exc
        if _pid_alive(pid):
            raise RuntimeError(f"COVTrack route already owned by live PID {pid}: {marker}")
        marker.unlink()
    payload = {
        "pid": os.getpid(),
        "started_utc": _now(),
        "frontend": FRONTENDS[slug]["name"],
        "gpu_index": int(gpu_index),
        "run_root": str(run_root.resolve()),
        "command": command,
    }
    temporary = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return marker


def _native_output(run_root: Path) -> Path:
    return run_root / "formatted" / "tao_track.json"


def run(slug: str, *, gpu_index: int, run_id: str, split: str = "val") -> dict[str, Any]:
    if slug not in FRONTENDS:
        raise ValueError(f"unsupported COVTrack route: {slug}")
    if split not in {"val", "test"}:
        raise ValueError(f"unsupported COVTrack split: {split}")
    # This guard precedes the Test annotation check and all model-side Test
    # dataset construction.  Val remains available for frontend selection.
    if split == "test":
        from src.trackocd_v2.protocol import assert_test_semantic_access_allowed

        assert_test_semantic_access_allowed(
            OUTPUT_TARGET / "audit/FINAL_FREEZE.json",
            "run frozen COVTrack TAO Test frontend",
        )
    for path in (COVTRACK_PYTHON, CONFIG, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)
    if split == "test" and not TEST_ANNOTATION.is_file():
        raise FileNotFoundError(TEST_ANNOTATION)
    _require_resource_gate(gpu_index)
    run_root = OUTPUT_TARGET / "frontend_native" / slug / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    formatted = run_root / "formatted"
    formatted.mkdir(parents=True, exist_ok=True)
    log_path = run_root / f"native_{split}.log"
    confused = str(FRONTENDS[slug]["confused_features"]).lower()
    cfg_options = [
        "model.tracker.confused_features=" + confused,
        "data.samples_per_gpu=1",
        "data.workers_per_gpu=1",
        "data.persistent_workers=False",
    ]
    if split == "test":
        cfg_options[1:1] = [
            "model.roi_head.only_validation_categories=False",
            "model.roi_head.only_test_categories=True",
            "data.test.ann_file=" + str(TEST_ANNOTATION.resolve()),
            "data.test.img_prefix=data/tao/frames/",
        ]
    command = [
        str(COVTRACK_PYTHON),
        str(COVTRACK_LAUNCHER.resolve()),
        str(CONFIG.resolve()),
        str(CHECKPOINT.resolve()),
        "--format-only",
        "--show_score_thr", "0.0001",
        "--cfg-options",
        *cfg_options,
        "--eval-options",
        "resfile_path=" + str(formatted.resolve()),
    ]
    marker = _claim(slug, split, command, run_root, gpu_index)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    environment["PYTHONPATH"] = os.pathsep.join([str(COVTRACK_ROOT.resolve()), str(ROOT.resolve()), environment.get("PYTHONPATH", "")])
    covtrack_lib = COVTRACK_PYTHON.resolve().parent.parent / "lib"
    # The compatible Python 3.8 PIL extension needs libstdc++ from the
    # existing CLIP donor environment.  Do not expose its site-packages:
    # that would shadow the pinned COVTrack torch installation.
    clip_runtime_lib = Path("/home/lwr/anaconda3/envs/ovtrack/lib")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(clip_runtime_lib), str(covtrack_lib), environment.get("LD_LIBRARY_PATH", "")]
    )
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(command, cwd=COVTRACK_ROOT, env=environment, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        output = _native_output(run_root)
        if process.returncode != 0:
            raise RuntimeError(f"COVTrack test returned {process.returncode}; see {log_path}")
        if not output.is_file():
            raise FileNotFoundError(output)
        if split == "test":
            normalize_command = [
                sys.executable,
                str((ROOT / "scripts/trackocd_v2/register_test_frontend_stream.py").resolve()),
                "--frontend", FRONTENDS[slug]["name"],
                "--native-input", str(output.resolve()),
                "--input-format", "tao_json",
                "--annotation", str(TEST_ANNOTATION.resolve()),
                "--output-dir", str((OUTPUT_TARGET / "manifests/frontend_streams" / slug / "test").resolve()),
            ]
        else:
            normalize_command = [
                sys.executable,
                str((ROOT / "scripts/trackocd_v2/normalize_frontend_output.py").resolve()),
                "--frontend", FRONTENDS[slug]["name"],
                "--input", str(output.resolve()),
                "--input-format", "tao_json",
                "--output-dir", str((OUTPUT_TARGET / "manifests/frontend_streams" / slug).resolve()),
            ]
        normalized_log = run_root / "normalization.log"
        with normalized_log.open("w", encoding="utf-8") as log_handle:
            normalized_process = subprocess.run(normalize_command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if normalized_process.returncode != 0:
            raise RuntimeError(f"COVTrack normalization returned {normalized_process.returncode}; see {normalized_log}")
        if split == "val":
            register_command = [
                sys.executable,
                str((ROOT / "scripts/trackocd_v2/register_frontend_normalization.py").resolve()),
                "--frontend", slug,
                "--normalization-audit", str((OUTPUT_TARGET / "audit" / f"frontend_{slug}_normalization.json").resolve()),
            ]
            register_process = subprocess.run(register_command, cwd=ROOT, check=False)
            if register_process.returncode != 0:
                raise RuntimeError(f"COVTrack normalization registration returned {register_process.returncode}")
        audit_name = f"frontend_{slug}_native_run.json" if split == "val" else f"frontend_{slug}_native_test_run.json"
        audit_path = OUTPUT_TARGET / "audit" / audit_name
        result = {
            "schema_version": "trackocd.v2.covtrack_native_run.v1",
            "status": "COMPLETE",
            "generated_utc": _now(),
            "split": split,
            "frontend": FRONTENDS[slug]["name"],
            "frontend_slug": slug,
            "run_root": str(run_root.resolve()),
            "native_command": command,
            "normalization_command": normalize_command,
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "config": str(CONFIG.resolve()),
            "confused_features": FRONTENDS[slug]["confused_features"],
            "native_output": str(output.resolve()),
            "native_output_sha256": _sha256(output),
            "normalization_audit": str(
                (OUTPUT_TARGET / "audit" / f"frontend_{slug}_normalization.json").resolve()
                if split == "val"
                else (OUTPUT_TARGET / "audit" / f"frontend_{slug}_test_normalization.json").resolve()
            ),
            "test_semantic_accessed": False,
            "test_evaluation_unlocked": split == "test",
        }
        _atomic_json(audit_path, result)
        marker.unlink()
        done_suffix = "" if split == "val" else "_test"
        done_path = OUTPUT_TARGET / "audit" / f"frontend_{slug}{done_suffix}.done"
        _atomic_json(done_path, {
            "frontend": FRONTENDS[slug]["name"],
            "finished_utc": _now(),
            "run_root": str(run_root.resolve()),
            "native_output_sha256": result["native_output_sha256"],
        })
        return result
    except Exception:
        # Keep the launch marker as resumable ownership evidence.  The next
        # invocation removes it only after confirming this PID is gone.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(FRONTENDS), required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    OUTPUT_TARGET.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args.frontend, gpu_index=args.gpu_index, run_id=args.run_id, split=args.split)
    except ResourceWait as exc:
        payload = {
            "schema_version": "trackocd.v2.covtrack_native_run.v1",
            "status": "WAITING_RESOURCE",
            "generated_utc": _now(),
            "frontend_slug": args.frontend,
            "split": args.split,
            "error": str(exc),
            "test_semantic_accessed": False,
        }
        audit_name = f"frontend_{args.frontend}_native_run.json" if args.split == "val" else f"frontend_{args.frontend}_native_test_run.json"
        _atomic_json(OUTPUT_TARGET / "audit" / audit_name, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.covtrack_native_run.v1",
            "status": "FAILED_COVTRACK_NATIVE_RUN",
            "generated_utc": _now(),
            "frontend_slug": args.frontend,
            "split": args.split,
            "error": f"{type(exc).__name__}: {exc}",
            "test_semantic_accessed": False,
        }
        audit_name = f"frontend_{args.frontend}_native_run.json" if args.split == "val" else f"frontend_{args.frontend}_native_test_run.json"
        _atomic_json(OUTPUT_TARGET / "audit" / audit_name, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
