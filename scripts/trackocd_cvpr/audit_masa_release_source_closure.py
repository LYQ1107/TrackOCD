#!/usr/bin/env python3
"""Audit the MASA release/source route boundary without model or data access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_ROOT = Path("/data3/liuyeqiang/research_references/N72R20/MASA")
CHECKPOINT_SHA256 = "441c05bf9519632fead1afd5200bb6a4f13b4a41c58f4428024490b4c2bd777c"
SOURCE_COMMIT = "c5472b9c7615f35abdf1188cb1a0c5408fe50d66"
CHECKPOINT_URL = "https://huggingface.co/dereksiyuanli/masa/resolve/main/sam_vitb_masa.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def git(reference_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(reference_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_record(reference_root: Path, relative: str) -> dict[str, Any]:
    path = reference_root / relative
    data = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": git(reference_root, "hash-object", relative),
    }


def build_receipt(reference_root: Path = DEFAULT_REFERENCE_ROOT) -> dict[str, Any]:
    reference_root = Path(reference_root).resolve()
    route = read_json(ROOT / "outputs/trackocd_core/audit/masa_training_route.json")
    runtime = read_json(ROOT / "outputs/trackocd_cvpr/cvpr5/masa_runtime_checkpoint_reconciliation_20261026.json")
    physical = read_json(ROOT / "outputs/trackocd_core/audit/masa_physical_prediction_summary.json")

    readme = (reference_root / "README.md").read_text(encoding="utf-8")
    model_zoo = (reference_root / "docs/model_zoo.md").read_text(encoding="utf-8")
    train = (reference_root / "docs/train.md").read_text(encoding="utf-8")
    release_config_path = reference_root / "configs/masa-sam/open_vocabulary_mot_test/masa_sam_vitb_open_vocabulary_test.py"
    release_config = release_config_path.read_text(encoding="utf-8")
    native_config = physical.get("frozen_model", {}).get("native_model_config", {})

    reference_head = git(reference_root, "rev-parse", "HEAD")
    checks = {
        "reference_tree_is_clean": git(reference_root, "status", "--porcelain") == "",
        "reference_head_matches_pinned_source": reference_head == SOURCE_COMMIT,
        "model_zoo_links_pinned_checkpoint": CHECKPOINT_URL in model_zoo and CHECKPOINT_URL in readme,
        "training_docs_describe_generic_sa1b_route": (
            "500K images sampled from SA-1B" in train
            and "tools/train.py" in train
            and "sam_vit_b_01ec64_mmdet.pth" in train
        ),
        "official_tao_release_config_exists": release_config_path.is_file(),
        "official_tao_release_uses_public_detections": (
            "load_public_dets = True" in release_config
            and "public_det_path" in release_config
        ),
        "official_tao_release_uses_pretrained_sam_b_init": (
            "sam_vit_b_01ec64_mmdet.pth" in release_config
            and "type='Pretrained'" in release_config
        ),
        "current_candidate_is_image_only": (
            native_config.get("load_public_dets") is False
            and native_config.get("public_det_path") is None
            and native_config.get("given_dets") is False
        ),
        "exact_checkpoint_stage_binding_still_open": (
            route.get("exact_checkpoint_stage_config_supervision_binding_verified") is False
            and runtime.get("qualification_boundary", {}).get("exact_release_stage_training_binding_verified") is False
        ),
    }
    status = (
        "PASS_SOURCE_CLOSURE_RELEASE_ROUTE_MISMATCH_QUALIFICATION_OPEN"
        if all(checks.values())
        else "FAIL_SOURCE_CLOSURE_AUDIT"
    )
    return {
        "schema_version": "trackocd.cvpr.masa-release-source-closure.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_audit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "status": status,
        "reference_repository": {
            "path": str(reference_root),
            "head": reference_head,
            "source_commit": SOURCE_COMMIT,
            "tag_count": int(git(reference_root, "tag", "--list", "--format=%(refname:short)").count("\n"))
            if git(reference_root, "tag", "--list", "--format=%(refname:short)")
            else 0,
        },
        "checks": checks,
        "release_route": {
            "checkpoint_url": CHECKPOINT_URL,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "official_tao_config": "configs/masa-sam/open_vocabulary_mot_test/masa_sam_vitb_open_vocabulary_test.py",
            "official_load_public_dets": True,
            "official_public_det_path_required": True,
            "current_trackocd_candidate_load_public_dets": native_config.get("load_public_dets"),
            "current_trackocd_candidate_public_det_path": native_config.get("public_det_path"),
            "current_trackocd_candidate_given_dets": native_config.get("given_dets"),
        },
        "source_evidence": [
            source_record(reference_root, "README.md"),
            source_record(reference_root, "docs/model_zoo.md"),
            source_record(reference_root, "docs/train.md"),
            source_record(reference_root, "configs/masa-sam/open_vocabulary_mot_test/masa_sam_vitb_open_vocabulary_test.py"),
            source_record(reference_root, "configs/masa-sam/sam-vitb.py"),
        ],
        "qualification_boundary": {
            "official_release_route_reproduced": False,
            "current_image_only_candidate_is_official_tao_route": False,
            "runtime_checkpoint_compatibility_reconciled": runtime.get("qualification_boundary", {}).get("runtime_checkpoint_compatibility_reconciled"),
            "exact_release_stage_training_binding_verified": False,
            "dense_full_train_or_val_frontend_qualified": False,
            "formal_v2_cache_admitted": False,
            "formal_ocd_or_mot_metrics_created": False,
            "model_selection_performed": False,
        },
        "boundary": {
            "model_constructed_or_run": False,
            "new_inference_or_training": False,
            "data_or_annotations_read": False,
            "tao_test_read": False,
            "weights_downloaded": False,
            "external_processes_touched": False,
        },
        "next_valid_gate": (
            "Either recover the exact public-detection input and official release route under a frozen v2 contract, "
            "or register a separately justified image-only route; neither may be silently promoted to the other."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/masa_release_source_closure_20261026.json"),
    )
    args = parser.parse_args()
    result = build_receipt(args.reference_root)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
