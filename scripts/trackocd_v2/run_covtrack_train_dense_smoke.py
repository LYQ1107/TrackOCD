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
TRAIN_FRAME_ALIAS = OUTPUT_TARGET / "frontend_inputs/covtrack_train_dense_smoke_v1/val_frame_alias"
TRAIN_CLASSES = OUTPUT_TARGET / "frontend_inputs/covtrack_val/lvis_classes_v1.txt"
FRONTEND_SLUG = "covtrack_native"
FRONTEND_NAME = FRONTENDS[FRONTEND_SLUG]["name"]


def _marker_path(run_id: str) -> Path:
    return OUTPUT_TARGET / "audit" / f"covtrack_train_dense_smoke_{run_id}.launched.json"


def _check_registration(train_audit: Path, train_annotation: Path, expected_video_ids: list[int]) -> dict[str, Any]:
    if not train_audit.is_file() or not train_annotation.is_file():
        raise FileNotFoundError("the preregistered anonymous Train input has not been built")
    audit = json.loads(train_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_FIXED_ANONYMOUS_TRAIN_DENSE_INPUT":
        raise ValueError(f"unexpected input audit status: {audit.get('status')}")
    if audit.get("selection", {}).get("video_ids") != expected_video_ids:
        raise ValueError("the input selection does not match the predeclared video IDs")
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


def run(*, gpu_index: int, run_id: str, train_annotation: Path = TRAIN_ANNOTATION, train_audit: Path = TRAIN_AUDIT, frame_alias: Path = TRAIN_FRAME_ALIAS, train_classes: Path = TRAIN_CLASSES, receipt_root: Path = CVPR_SMOKE_OUTPUT, output_scope: str = "train_dense_smoke_v1", expected_video_ids: list[int] | None = None, result_status: str = "COMPLETE_ANONYMOUS_TRAIN_DENSE_FRONTEND_SMOKE", track_category_consensus: bool = True) -> dict[str, Any]:
    expected_video_ids = list(range(0, 2)) if expected_video_ids is None else list(expected_video_ids)
    input_audit = _check_registration(train_audit, train_annotation, expected_video_ids)
    for path in (COVTRACK_PYTHON, CONFIG, CHECKPOINT, PROMPT, train_annotation, train_audit, TRAIN_FRAME_ROOT, frame_alias, train_classes):
        if not path.exists():
            raise FileNotFoundError(path)
    if not frame_alias.is_symlink() or frame_alias.resolve() != TRAIN_FRAME_ROOT.resolve():
        raise ValueError("Train frame compatibility alias is not the registered symlink to the Train frame root")
    _require_resource_gate(gpu_index)

    run_root = OUTPUT_TARGET / "frontend_native" / FRONTEND_SLUG / run_id
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse existing smoke run root: {run_root}")
    formatted = run_root / "formatted"
    formatted.mkdir(parents=True, exist_ok=False)
    log_path = run_root / "native_train.log"
    normalized_log = run_root / "normalization.log"
    normalization_dir = OUTPUT_TARGET / "manifests/frontend_streams" / FRONTEND_SLUG / output_scope
    normalization_audit = receipt_root / "normalization.json"
    physical_output = normalization_dir / "physical_tracks.jsonl"
    native_evaluator_output = normalization_dir / "native_evaluator_rows.jsonl"
    run_audit = receipt_root / "run.json"
    done_marker = receipt_root / "done.json"

    cfg_options = [
        "data.test.ann_file=" + str(train_annotation.resolve()),
        # COVTrack changes its working directory to the external vendor tree.
        # A relative alias therefore resolves against the wrong root and can
        # fail only after the model has started.  Resolve it at registration
        # time while retaining the symlink target contract checked above.
        "data.test.img_prefix=" + str(frame_alias.resolve()) + os.sep,
        "data.test.classes=" + str(train_classes.resolve()),
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
        "tcc=" + ("True" if track_category_consensus else "False"),
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
        "input_audit": str(train_audit.resolve()),
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
            str(train_annotation.resolve()),
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
            "status": result_status,
            "generated_utc": _now(),
            "frontend": FRONTEND_NAME,
            "frontend_slug": FRONTEND_SLUG,
            "split": "train_predicted",
            "run_root": str(run_root.resolve()),
            "gpu_index": gpu_index,
            "command": command,
            "normalization_command": normalize_command,
            "input_audit": str(train_audit.resolve()),
            "input_audit_sha256": _sha256(train_audit),
            "annotation": str(train_annotation.resolve()),
            "annotation_sha256": _sha256(train_annotation),
            "frame_root": str(TRAIN_FRAME_ROOT.resolve()),
            "frame_alias": str(frame_alias),
            "frame_alias_target": str(frame_alias.resolve()),
            "classes": str(train_classes.resolve()),
            "classes_sha256": _sha256(train_classes),
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
            "native_track_category_consensus": bool(track_category_consensus),
            "interpretation": (
                "Frontend/input inference only; native COVTrack track-category consensus is disabled because "
                "the train_predicted physical stream is category-free; no OCD/MOT metrics or frontend qualification claimed."
                if not track_category_consensus
                else "Frontend/input inference only; no OCD/MOT metrics or frontend qualification claimed."
            ),
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
            "status": "FAILED_COVTRACK_TRAIN_DENSE_RUN",
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
    parser.add_argument("--annotation", type=Path, default=TRAIN_ANNOTATION)
    parser.add_argument("--input-audit", type=Path, default=TRAIN_AUDIT)
    parser.add_argument("--frame-alias", type=Path, default=TRAIN_FRAME_ALIAS)
    parser.add_argument("--classes", type=Path, default=TRAIN_CLASSES)
    parser.add_argument("--receipt-root", type=Path, default=CVPR_SMOKE_OUTPUT)
    parser.add_argument("--output-scope", default="train_dense_smoke_v1")
    parser.add_argument("--video-id-min", type=int, default=0)
    parser.add_argument("--video-id-max", type=int, default=1)
    parser.add_argument("--use-audit-video-ids", action="store_true", help="use the committed input-audit video-ID list instead of an ordinal range")
    parser.add_argument("--disable-tcc", action="store_true", help="disable upstream track-category consensus for the anonymous category-free Train stream to reduce post-processing RAM")
    parser.add_argument("--result-status", default="COMPLETE_ANONYMOUS_TRAIN_DENSE_FRONTEND_SMOKE")
    args = parser.parse_args()
    if args.use_audit_video_ids:
        audit_payload = json.loads(args.input_audit.resolve().read_text(encoding="utf-8"))
        expected_video_ids = [int(value) for value in audit_payload.get("selection", {}).get("video_ids", [])]
    else:
        expected_video_ids = list(range(args.video_id_min, args.video_id_max + 1))
    try:
        result = run(
            gpu_index=args.gpu_index,
            run_id=args.run_id,
            train_annotation=args.annotation.resolve(),
            train_audit=args.input_audit.resolve(),
            frame_alias=args.frame_alias,
            train_classes=args.classes.resolve(),
            receipt_root=args.receipt_root.resolve(),
            output_scope=args.output_scope,
            expected_video_ids=expected_video_ids,
            result_status=args.result_status,
            track_category_consensus=not args.disable_tcc,
        )
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
        _atomic_json(args.receipt_root.resolve() / "run.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "FAILED_COVTRACK_TRAIN_DENSE_RUN", "error": f"{type(exc).__name__}: {exc}"}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
