#!/usr/bin/env python3
"""Audit the small, public COVTrack assets recovered for the v2 frontend.

This audit is deliberately source/configuration-only.  It does not open TAO
annotations, images, text labels, or model tensors, and it does not launch a
frontend.  The public text embedding is recorded as a fixed runtime asset;
it is not a model-selection input and never enters the TrackOCD backend.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "vendor/trackocd_external/covtrack_public"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/covtrack/ctao_public.pth"
DEFAULT_PROMPT = ROOT / "checkpoints/covtrack/detpro_prompt.pt"
DEFAULT_RUNTIME = ROOT.parent / ".venvs/covtrack-public/bin/python"
DEFAULT_CONFIG = DEFAULT_SOURCE / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
EXPECTED_SOURCE_COMMIT = "9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b"
EXPECTED_CHECKPOINT_SHA256 = "e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c"
EXPECTED_CHECKPOINT_BYTES = 368829353
EXPECTED_PROMPT_SHA256 = "00ba2f0bfb9abdb577b2fcf2afae12655d6190a18f11051aabdc97f8e4c24579"
EXPECTED_PROMPT_BYTES = 2464491
REQUIRED_RUNTIME_MODULES = (
    "torch",
    "mmcv",
    "mmdet",
    "clip",
    "addict",
    "lvis",
    "motmetrics",
    "pycocotools",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve()
    actual_bytes = path.stat().st_size if path.is_file() else None
    actual_sha256 = _sha256(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_sha256,
        "integrity": (
            "PASS"
            if actual_bytes == expected_bytes and actual_sha256 == expected_sha256
            else "FAIL"
        ),
    }


def _git_value(source: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _config_contract(config: Path) -> dict[str, Any]:
    if not config.is_file():
        return {"exists": False, "tokens": {}}
    text = config.read_text(encoding="utf-8")
    required_tokens = {
        "public_prompt_path": "prompt_path='saved_models/pretrained_models/detpro_prompt.pt'",
        "validation_category_route": "only_validation_categories=True",
        "lvis_class_count": "num_classes=1203",
        "special_text_prompt": "use_special_text_prompt=True",
        "special_image_prompt": "use_special_image_prompt=True",
        "native_semantic_cue_default": "confused_features=True",
        "ctao_base_training_annotation": "saved_models/ctao_dataset/ctao_base.json",
    }
    return {
        "exists": True,
        "path": str(config.resolve()),
        "sha256": _sha256(config),
        "tokens": {name: token in text for name, token in required_tokens.items()},
        "interpretation": {
            "is_training_config": True,
            "formal_run_uses_frozen_checkpoint": True,
            "prompt_is_runtime_frontend_asset": True,
            "prompt_used_for_model_selection": False,
            "training_started_by_this_audit": False,
        },
    }


def _runtime_probe(runtime_python: str) -> dict[str, Any]:
    probe = (
        "import importlib.util, json; "
        f"names={REQUIRED_RUNTIME_MODULES!r}; "
        "print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in names}, sort_keys=True))"
    )
    try:
        result = subprocess.run(
            [runtime_python, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"python": runtime_python, "probe_status": "ERROR", "error": str(exc)}
    if result.returncode != 0:
        return {
            "python": runtime_python,
            "probe_status": "ERROR",
            "returncode": result.returncode,
            "stderr": result.stderr[-1000:],
        }
    try:
        modules = json.loads(result.stdout)
    except json.JSONDecodeError:
        modules = {}
    missing = [name for name, present in modules.items() if not present]
    return {
        "python": runtime_python,
        "probe_status": "PASS",
        "modules": modules,
        "missing_required_modules": missing,
        "legacy_runtime_ready": not missing,
    }


def _default_runtime_python() -> str:
    """Prefer the isolated recovered COVTrack runtime over TrackOCD's env."""

    configured = os.environ.get("TRACKOCD_COVTRACK_PYTHON")
    if configured:
        return configured
    if DEFAULT_RUNTIME.is_file():
        return str(DEFAULT_RUNTIME)
    return sys.executable


def build_report(
    *,
    source_root: Path = DEFAULT_SOURCE,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    prompt: Path = DEFAULT_PROMPT,
    runtime_python: str | None = None,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    checkpoint = checkpoint.resolve()
    prompt = prompt.resolve()
    runtime_python = runtime_python or _default_runtime_python()
    config = source_root / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
    source_commit = _git_value(source_root, "rev-parse", "HEAD")
    source_status = _git_value(source_root, "status", "--porcelain", "--untracked-files=no")
    checkpoint_asset = _asset(checkpoint, EXPECTED_CHECKPOINT_BYTES, EXPECTED_CHECKPOINT_SHA256)
    prompt_asset = _asset(prompt, EXPECTED_PROMPT_BYTES, EXPECTED_PROMPT_SHA256)
    runtime = _runtime_probe(runtime_python)
    source_ok = source_commit == EXPECTED_SOURCE_COMMIT and source_status == ""
    assets_ok = checkpoint_asset["integrity"] == "PASS" and prompt_asset["integrity"] == "PASS"
    runtime_ok = runtime.get("legacy_runtime_ready") is True
    return {
        "schema_version": "trackocd.v2.public_covtrack_recovery.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PUBLIC_ASSETS_RECOVERED_RUNTIME_BLOCKED" if not runtime_ok else "PUBLIC_ASSETS_RECOVERED_RUNTIME_READY_NOT_RUN",
        "source": {
            "repository": "https://github.com/zekunqian/COVTrack",
            "root": str(source_root),
            "commit": source_commit,
            "expected_commit": EXPECTED_SOURCE_COMMIT,
            "clean_tracked_state": source_status == "",
            "integrity": "PASS" if source_ok else "FAIL",
        },
        "public_assets": {
            "checkpoint": checkpoint_asset,
            "prompt_embedding": prompt_asset,
            "model_repository": "https://huggingface.co/clarkqian/COVTrack",
            "downloaded_ctao_annotations": False,
            "downloaded_tao_test": False,
            "downloaded_feature_cache": False,
        },
        "config_contract": _config_contract(config),
        "runtime": runtime,
        "trackocd_boundary": {
            "frontend_only": True,
            "backend_category_free": True,
            "prompt_is_fixed_runtime_asset": True,
            "prompt_used_for_model_selection": False,
            "novel_vocabulary_used_for_selection": False,
            "tao_val_read": False,
            "tao_test_read": False,
            "training_started": False,
            "native_frontend_run_started": False,
            "formal_v2_admissible": False,
            "reason": (
                "Assets are hash-verified and the compatibility runtime passes "
                "import smoke, but no v2 frontend run receipt exists."
                if runtime_ok
                else "Assets are hash-verified, but the legacy runtime is not yet available and no v2 frontend run receipt exists."
            ),
        },
        "next_gate": (
            "Register a Val-only full-run input contract and run both fixed "
            "COVTrack-native and COVTrack-NoSemantic frontend routes without "
            "model selection, then perform the separately registered evaluator "
            "and protocol audit."
            if runtime_ok
            else "Prepare or recover an isolated MMCV 1.x/MMDetection 2.x/CLIP "
            "runtime, run import/config smoke only, then register a bounded "
            "TAO-Val COVTrack-native and COVTrack-NoSemantic frontend run."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--runtime-python", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/trackocd_v2/PUBLIC_COVTRACK_RECOVERY_20261016.json",
    )
    args = parser.parse_args()
    report = build_report(
        source_root=args.source_root,
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        runtime_python=args.runtime_python,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["source"]["integrity"] == "PASS" and report["public_assets"]["checkpoint"]["integrity"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
