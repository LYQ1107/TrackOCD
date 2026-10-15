#!/usr/bin/env python3
"""Run the preregistered bounded anonymous COVTrack Train dense smoke.

This is a frontend inference-only route.  It deliberately does not reuse the
Val/Test wrapper: the input is an anonymous two-video Train manifest, and its
normalized stream is stored under a separate ``train_predicted`` namespace.
Raw native output is local-only; the committed receipt contains no boxes or
semantic rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_v2.run_covtrack_native import (  # noqa: E402
    CHECKPOINT,
    CONFIG,
    COVTRACK_LAUNCHER,
    COVTRACK_PYTHON,
    FRONTENDS,
    PROMPT,
    ResourceWait,
    _atomic_json,
    _now,
    _require_resource_gate,
    _sha256,
)
from src.trackocd_v2.io import OUTPUT_TARGET  # noqa: E402


TRAIN_ANNOTATION = OUTPUT_TARGET / "frontend_inputs/covtrack_train_dense_smoke_v1/anonymous_train.json"
CVPR_SMOKE_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_smoke_v1"
TRAIN_AUDIT = CVPR_SMOKE_OUTPUT / "input_audit.json"
TRAIN_FRAME_ROOT = Path("/data3/liuyeqiang/TAO-Amodal/frames")
TRAIN_CLASSES = OUTPUT_TARGET / "frontend_inputs/covtrack_val/lvis_classes_v1.txt"
FRONTEND_SLUG = "covtrack_native"
FRONTEND_NAME = FRONTENDS[FRONTEND_SLUG]["name"]


def _marker_path(run_id: str) -> Path:
    return OUTPUT_TARGET / "audit" / f"covtrack_train_dense_smoke_{run_id}.launched.json"


def _check_registration() -> dict[str, Any]:
    if not TRAIN_AUDIT.is_file() or not TRAIN_ANNOTATION.is_file():
        raise FileNotFoundError("the preregistered anonymous Train input has not been built")
    audit = json.loads(TRAIN_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_FIXED_ANONYMOUS_TRAIN_DENSE_INPUT":
        raise ValueError(f"unexpected input audit status: {audit.get('status')}")
    if audit.get("selection", {}).get("video_ids") != [0, 1]:
        raise ValueError("the smoke selection is not the preregistered fixed pair [0, 1]")
    if audit.get("boundaries", {}).get("tao_train_annotation_opened") is not False:
        raise ValueError("Train GT boundary is not closed")
    if audit.get("boundaries", {}).get("tao_val_accessed") is not False:
        raise ValueError("Val boundary is not closed")
    if audit.get("boundaries", {}).get("tao_test_accessed") is not False:
        raise ValueError("Test boundary is not closed")
    return audit


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str((ROOT / "vendor/trackocd_external/covtrack_public").resolve()), str(ROOT.resolve()), environment.get("PYTHONPATH", "")]
    )
    covtrack_lib = COVTRACK_PYTHON.resolve().parent.parent / "lib"
    clip_runtime_lib = Path("/home/lwr/anaconda3/envs/ovtrack/lib")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(clip_runtime_lib), str(covtrack_lib), environment.get("LD_LIBRARY_PATH", "")]
    )
    return environment


def run(*, gpu_index: int, run_id: str) -> dict[str, Any]:
    input_audit = _check_registration()
    for path in (COVTRACK_PYTHON, CONFIG, CHECKPOINT, PROMPT, TRAIN_ANNOTATION, TRAIN_AUDIT, TRAIN_FRAME_ROOT, TRAIN_CLASSES):
        if not path.exists():
            raise FileNotFoundError(path)
    _require_resource_gate(gpu_index)

    run_root = OUTPUT_TARGET / "frontend_native" / FRONTEND_SLUG / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing smoke run root: {run_root}")
    formatted = run_root / "formatted"
    formatted.mkdir(parents=True, exist_ok=False)
    log_path = run_root / "native_train.log"
    normalized_log = run_root / "normalization.log"
    normalization_dir = OUTPUT_TARGET / "manifests/frontend_streams" / FRONTEND_SLUG / "train_dense_smoke_v1"
    normalization_audit = OUTPUT_TARGET / "audit" / "covtrack_train_dense_smoke_v1_normalization.json"
    physical_output = normalization_dir / "physical_tracks.jsonl"
    native_evaluator_output = normalization_dir / "native_evaluator_rows.jsonl"
    run_audit = CVPR_SMOKE_OUTPUT / "run.json"
    done_marker = CVPR_SMOKE_OUTPUT / "done.json"

    cfg_options = [
        "data.test.ann_file=" + str(TRAIN_ANNOTATION.resolve()),
        "data.test.img_prefix=" + str(TRAIN_FRAME_ROOT.resolve()) + os.sep,
        "data.test.classes=" + str(TRAIN_CLASSES.resolve()),
        "model.tracker.confused_features=true",
        "model.roi_head.prompt_path=" + str(PROMPT.resolve()),
        "data.samples_per_gpu=1",
        "data.workers_per_gpu=1",
        "data.persistent_workers=False",
    ]
    command = [
        str(COVTRACK_PYTHON),
        str(COVTRACK_LAUNCHER.resolve()),
        str(CONFIG.resolve()),
        str(CHECKPOINT.resolve()),
        "--format-only",
        "--show_score_thr",
        "0.0001",
        "--cfg-options",
        *cfg_options,
        "--eval-options",
        "resfile_path=" + str(formatted.resolve()),
    ]
    marker = _marker_path(run_id)
    _atomic_json(marker, {
        "schema_version": "trackocd.cvpr.covtrack_train_dense_smoke_launch.v1",
        "status": "RUNNING",
        "started_utc": _now(),
        "frontend": FRONTEND_NAME,
        "split": "train_predicted",
        "gpu_index": gpu_index,
        "run_root": str(run_root.resolve()),
        "command": command,
        "input_audit": str(TRAIN_AUDIT.resolve()),
    })

    environment = _environment()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(command, cwd=ROOT / "vendor/trackocd_external/covtrack_public", env=environment, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        native_track = formatted / "tao_track.json"
        native_bbox = formatted / "tao_bbox.json"
        if process.returncode != 0:
            raise RuntimeError(f"COVTrack returned {process.returncode}; see {log_path}")
        for path in (native_track, native_bbox):
            if not path.is_file():
                raise FileNotFoundError(path)

        normalize_command = [
            sys.executable,
            str((ROOT / "scripts/trackocd_v2/normalize_frontend_output.py").resolve()),
            "--frontend",
            FRONTEND_NAME,
            "--input",
            str(native_track.resolve()),
            "--input-format",
            "tao_json",
            "--annotation",
            str(TRAIN_ANNOTATION.resolve()),
            "--output-dir",
            str(normalization_dir.resolve()),
            "--audit-output",
            str(normalization_audit.resolve()),
            "--source-split",
            "train_predicted",
        ]
        with normalized_log.open("w", encoding="utf-8") as log_handle:
            normalized_process = subprocess.run(normalize_command, cwd=ROOT, env=dict(os.environ), stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        if normalized_process.returncode != 0:
            raise RuntimeError(f"normalization returned {normalized_process.returncode}; see {normalized_log}")
        normalization = json.loads(normalization_audit.read_text(encoding="utf-8"))
        if normalization.get("status") not in {"COMPLETE", "COMPLETE_WITH_INVALID_BOXES"}:
            raise ValueError(f"unexpected normalization status: {normalization.get('status')}")
        if normalization.get("source_split") != "train_predicted":
            raise ValueError("normalization source split is not train_predicted")
        if normalization.get("contracts", {}).get("physical_stream_contains_category_or_text") is not False:
            raise ValueError("physical stream category/text boundary failed")
        if normalization.get("contracts", {}).get("gt_join_used_for_normalization") is not False:
            raise ValueError("GT join boundary failed")
        result = {
            "schema_version": "trackocd.cvpr.covtrack_train_dense_smoke_run.v1",
            "status": "COMPLETE_ANONYMOUS_TRAIN_DENSE_FRONTEND_SMOKE",
            "generated_utc": _now(),
            "frontend": FRONTEND_NAME,
            "frontend_slug": FRONTEND_SLUG,
            "split": "train_predicted",
            "run_root": str(run_root.resolve()),
            "gpu_index": gpu_index,
            "command": command,
            "normalization_command": normalize_command,
            "input_audit": str(TRAIN_AUDIT.resolve()),
            "input_audit_sha256": _sha256(TRAIN_AUDIT),
            "annotation": str(TRAIN_ANNOTATION.resolve()),
            "annotation_sha256": _sha256(TRAIN_ANNOTATION),
            "frame_root": str(TRAIN_FRAME_ROOT.resolve()),
            "classes": str(TRAIN_CLASSES.resolve()),
            "classes_sha256": _sha256(TRAIN_CLASSES),
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": _sha256(CHECKPOINT),
            "prompt": str(PROMPT.resolve()),
            "prompt_sha256": _sha256(PROMPT),
            "config": str(CONFIG.resolve()),
            "native_track_output": str(native_track.resolve()),
            "native_track_output_sha256": _sha256(native_track),
            "native_bbox_output": str(native_bbox.resolve()),
            "native_bbox_output_sha256": _sha256(native_bbox),
            "normalization_audit": str(normalization_audit.resolve()),
            "physical_stream": str(physical_output.resolve()),
            "native_evaluator_rows": str(native_evaluator_output.resolve()),
            "counts": normalization.get("counts", {}),
            "invalid_box_policy": normalization.get("invalid_box_policy", {}),
            "training_started": False,
            "gt_or_semantic_arrays_read": False,
            "tao_val_accessed": False,
            "tao_test_accessed": False,
            "formal_metrics": False,
            "interpretation": "Frontend/input smoke only; no OCD/MOT metrics or frontend qualification claimed.",
        }
        _atomic_json(run_audit, result)
        _atomic_json(done_marker, {
            "status": result["status"],
            "finished_utc": _now(),
            "run_audit": str(run_audit.resolve()),
            "native_track_output_sha256": result["native_track_output_sha256"],
        })
        marker.unlink(missing_ok=True)
        return result
    except Exception as exc:
        _atomic_json(run_audit, {
            "schema_version": "trackocd.cvpr.covtrack_train_dense_smoke_run.v1",
            "status": "FAILED_COVTRACK_TRAIN_DENSE_SMOKE",
            "generated_utc": _now(),
            "frontend": FRONTEND_NAME,
            "split": "train_predicted",
            "gpu_index": gpu_index,
            "run_root": str(run_root.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "training_started": False,
            "tao_val_accessed": False,
            "tao_test_accessed": False,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = run(gpu_index=args.gpu_index, run_id=args.run_id)
    except ResourceWait as exc:
        payload = {
            "schema_version": "trackocd.cvpr.covtrack_train_dense_smoke_run.v1",
            "status": "WAITING_RESOURCE",
            "generated_utc": _now(),
            "error": str(exc),
            "training_started": False,
            "tao_val_accessed": False,
            "tao_test_accessed": False,
        }
        _atomic_json(CVPR_SMOKE_OUTPUT / "run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "FAILED_COVTRACK_TRAIN_DENSE_SMOKE", "error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
