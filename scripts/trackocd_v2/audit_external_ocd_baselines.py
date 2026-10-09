#!/usr/bin/env python3
"""Audit locally available OCD prior-art code without running a model.

The TrackOCD v2 baselines operate on causal physical-track prefixes.  AGE and
TALON are useful prior-art references, but their checked-in entry points use
image-level datasets and classifiers/prototype memories.  This audit records
that interface boundary explicitly so an unavailable or incompatible method
is not silently presented as a benchmark result.  PACO remains an
unconfirmed-code entry and is recorded as such.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


AUDIT = OUTPUT_TARGET / "audit/external_ocd_baseline_audit.json"


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_evidence(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _repo_record(
    *,
    name: str,
    official_repo: str,
    local_path: str | None,
    method_family: str,
    checked_entrypoints: list[str],
    protocol_boundary: str,
    run_status: str,
    license_note: str,
    checkpoint_note: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "official_repo": official_repo,
        "method_family": method_family,
        "local_path": str((ROOT / local_path).resolve()) if local_path else None,
        "official_code_status": "LOCAL_PIN_PRESENT" if local_path else "NO_CONFIRMED_OFFICIAL_CODE",
        "checked_entrypoints": checked_entrypoints,
        "trackocd_v2_drop_in": False,
        "drop_in_reason": protocol_boundary,
        "run_status": run_status,
        "license_note": license_note,
        "checkpoint_note": checkpoint_note,
        "test_semantic_accessed": False,
    }
    if local_path:
        repo = ROOT / local_path
        record["local_path_exists"] = repo.is_dir()
        record["git_sha"] = _git(repo, "rev-parse", "HEAD")
        record["git_remote"] = _git(repo, "remote", "get-url", "origin")
        evidence: dict[str, Any] = {}
        for relative in ["README.md", "LICENSE", "pyproject.toml", *checked_entrypoints]:
            item = _file_evidence(repo / relative)
            if item:
                evidence[relative] = item
        record["source_evidence"] = evidence
    else:
        record["local_path_exists"] = False
        record["source_evidence"] = {}
    return record


def main() -> int:
    out = ensure_output_layout()
    records = [
        _repo_record(
            name="AGE",
            official_repo="https://github.com/Ashengl/AGE",
            local_path="third_party/research_refs_phase4f/AGE",
            method_family="image-level adaptive Gaussian online category expansion",
            checked_entrypoints=["README.md", "main.py", "project_utils/AGE.py"],
            protocol_boundary=(
                "The pinned entry point consumes image datasets/class labels and emits image-level OCD assignments; "
                "it has no TAO physical-track-prefix input, causal track-order contract, or MOT evaluator adapter."
            ),
            run_status="REFERENCE_ONLY_NOT_RUN",
            license_note="MIT (checked-in LICENSE)",
            checkpoint_note="README says pretrained checkpoints will be released soon; no checkpoint is registered in the v2 namespace.",
        ),
        _repo_record(
            name="TALON",
            official_repo="https://github.com/ynanwu/TALON",
            local_path="third_party/research_refs_phase4f/TALON",
            method_family="image-level test-time prototype/model adaptation for OCD",
            checked_entrypoints=["README.md", "train.py", "test.py", "methods/talon/model.py", "methods/talon/trainer.py"],
            protocol_boundary=(
                "The pinned entry point consumes image datasets and updates classifier/prototype state at test time; "
                "it has no TAO physical-track-prefix API and is not a category-agnostic MOT frontend."
            ),
            run_status="REFERENCE_ONLY_NOT_RUN",
            license_note="No checked-in LICENSE found in the pinned repository",
            checkpoint_note="README documents external dataset checkpoints; none is registered as a v2 TAO checkpoint.",
        ),
        _repo_record(
            name="PACO",
            official_repo="https://arxiv.org/abs/2604.11484",
            local_path=None,
            method_family="reported image-level proxy-task/calibration OCD method",
            checked_entrypoints=[],
            protocol_boundary="No confirmed official code is present locally, so no implementation or benchmark result may be claimed.",
            run_status="NOT_RUN_NO_CONFIRMED_OFFICIAL_CODE",
            license_note="N/A",
            checkpoint_note="N/A",
        ),
    ]
    payload = {
        "schema_version": "trackocd.v2.external_ocd_baseline_audit.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "Register prior-art code availability and physical-track interface compatibility before baseline extension.",
        "selection_consequence": (
            "AGE and TALON remain prior-art references only; neither is a drop-in v2 physical frontend. "
            "PACO remains unavailable. No Test semantic data was accessed."
        ),
        "records": records,
        "test_semantic_accessed": False,
    }
    atomic_json(AUDIT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
