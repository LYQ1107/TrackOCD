#!/usr/bin/env python3
"""Run the pinned TETA reference on the sealed COVTrack-native Val stream."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/trackocd_cvpr/covtrack_native_teta_recovery_v1.json"
RUNNER = ROOT / "scripts/trackocd_v2/run_teta_reference.py"

from src.trackocd_v2.io import atomic_json


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "trackocd.cvpr.covtrack-native-teta-recovery.v1":
        raise ValueError("unexpected TETA recovery configuration")
    if config.get("status") != "REGISTERED_VAL_ONLY_TETA_REFERENCE_RECOVERY_NO_SELECTION":
        raise ValueError("TETA recovery is not frozen")
    if config.get("limits") != {
        "device": "CPU",
        "workers": 1,
        "parallel": False,
        "max_detections_per_image": 300,
        "training_gpu_hours": 0,
    }:
        raise ValueError("TETA reference limits changed")
    if config.get("boundaries") != {
        "val_only": True,
        "tao_test_read": False,
        "frontend_inference_started": False,
        "training_started": False,
        "model_selection_performed": False,
        "novel_vocabulary_used_for_forward_or_selection": False,
        "external_process_intervention": False,
        "owned_evaluator_subprocess": True,
        "formal_frontend_selection": False,
        "goal_acceptance": False,
    }:
        raise ValueError("TETA boundary changed")
    for key in ("native_stream", "annotation", "known_ids", "novel_ids", "output_directory"):
        if "test" in str(config[key]).lower():
            raise ValueError("TETA recovery config may not name a Test input")
    return config


def _verify_source(source_root: Path, expected_commit: str) -> None:
    if not (source_root / "teta" / "teta" / "__init__.py").is_file():
        raise FileNotFoundError(source_root / "teta" / "teta" / "__init__.py")
    observed = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != expected_commit:
        raise ValueError(f"TETA source commit mismatch: {observed} != {expected_commit}")


def run(config_path: Path, source_root: Path) -> dict:
    config = _load_config(config_path)
    output_root = ROOT / config["output_directory"]
    if output_root.exists():
        raise ValueError("preserve every previous TETA recovery attempt")
    output_root.mkdir(parents=True)
    native = ROOT / config["native_stream"]
    annotation = ROOT / config["annotation"]
    known_ids = ROOT / config["known_ids"]
    novel_ids = ROOT / config["novel_ids"]
    for path in (native, annotation, known_ids, novel_ids, RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (annotation, known_ids, novel_ids):
        if "test" in str(path).lower():
            raise ValueError("TETA recovery attempted a Test input")
    observed_stream_sha256 = _sha256(native)
    if observed_stream_sha256 != config["native_stream_sha256"]:
        raise ValueError("sealed native stream hash changed")
    _verify_source(source_root.resolve(), config["teta_commit"])
    output_json = output_root / "teta_reference.json"
    log_path = output_root / "evaluator.log"
    command = [
        sys.executable,
        str(RUNNER.resolve()),
        "--tracker-json", str(native.resolve()),
        "--annotation", str(annotation.resolve()),
        "--output-root", str(output_root.resolve()),
        "--output-json", str(output_json.resolve()),
        "--tracker-name", "covtrack_native",
        "--known-ids", str(known_ids.resolve()),
        "--novel-ids", str(novel_ids.resolve()),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str((source_root / "teta").resolve())
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("wb") as log:
        process = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            check=False,
        )
    if not output_json.is_file():
        raise RuntimeError(f"TETA runner produced no receipt, exit={process.returncode}")
    teta = json.loads(output_json.read_text(encoding="utf-8"))
    if process.returncode != 0 or teta.get("status") != "COMPLETE":
        raise RuntimeError(f"TETA reference failed: exit={process.returncode}, status={teta.get('status')}")
    result = {
        "schema_version": "trackocd.cvpr.covtrack-native-teta-recovery-result.v1",
        "status": "PASS_VAL_TETA_REFERENCE_RECOVERED_NO_SELECTION",
        "generated_utc": _now(),
        "config": {"path": str(config_path.resolve()), "sha256": _sha256(config_path)},
        "teta_source": {
            "repository": config["teta_repository"],
            "commit": config["teta_commit"],
            "source_root": str(source_root.resolve()),
        },
        "inputs": {
            "native_stream": str(native.resolve()),
            "native_stream_sha256": observed_stream_sha256,
            "annotation": str(annotation.resolve()),
            "known_ids": str(known_ids.resolve()),
            "novel_ids": str(novel_ids.resolve()),
            "tao_test_read": False,
            "frontend_inference_started": False,
            "training_started": False,
        },
        "teta_reference": teta,
        "command": command,
        "evaluator_log": str(log_path.resolve()),
        "boundaries": config["boundaries"],
        "interpretation": config["interpretation"],
    }
    atomic_json(output_root / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.source_root.resolve())
    print(json.dumps({
        "status": result["status"],
        "teta50": result["teta_reference"].get("teta50"),
        "tao_test_read": result["inputs"]["tao_test_read"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
