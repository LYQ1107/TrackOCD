#!/usr/bin/env python3
"""Verify preservation of the pre-CVPR TrackOCD research lineage.

This is a metadata-only audit.  It checks the immutable historical parent,
the protected local branch refs, the current ancestry relation, and the
existing CVPR-0 manifest's preservation declarations.  It does not open
streams, data, weights, labels, or external processes, and it never moves a
Git ref.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.legacy-preservation-audit.v1"
HISTORICAL_PARENT = "7546527e5f9540d351785ade34c8e3292a74441d"
EXPECTED_REFS = {
    "refs/heads/main": "7b1d3c09f2553450a078a7cee70518a18ef60613",
    "refs/heads/codex/trackocd-v2": "cac66467af7ca137fc53bdbef34896dde47030d8",
    "refs/heads/codex/trackocd-core-training-limited-masa": HISTORICAL_PARENT,
}
MANIFEST_PATH = "outputs/trackocd_cvpr/cvpr0/experiment_manifest.json"
MANIFEST_GENERATOR = "scripts/trackocd_cvpr/write_cvpr0_experiment_manifest.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_exists(repo: Path, *args: str) -> bool:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
    ).returncode == 0


def file_evidence(repo: Path, relative_path: str) -> dict[str, Any]:
    path = repo / relative_path
    return {
        "path": relative_path,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def build_audit(repo: Path) -> dict[str, Any]:
    manifest_path = repo / MANIFEST_PATH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generator_path = repo / MANIFEST_GENERATOR

    current_branch = git(repo, "branch", "--show-current")
    current_head = git(repo, "rev-parse", "HEAD")
    parent_exists = git_exists(repo, "cat-file", "-e", f"{HISTORICAL_PARENT}^{{commit}}")
    ancestry_verified = parent_exists and git_exists(
        repo, "merge-base", "--is-ancestor", HISTORICAL_PARENT, "HEAD"
    )

    refs: dict[str, dict[str, Any]] = {}
    for ref, expected in EXPECTED_REFS.items():
        actual_result = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        actual = actual_result.stdout.strip() if actual_result.returncode == 0 else None
        refs[ref] = {
            "expected": expected,
            "actual": actual,
            "exists": actual is not None,
            "matches_expected": actual == expected,
        }

    preserved_failed_attempts = manifest.get("preserved_failed_attempts", [])
    all_preserved = bool(preserved_failed_attempts) and all(
        item.get("not_deleted") is True for item in preserved_failed_attempts
    )
    manifest_binding = {
        "historical_parent": manifest.get("historical_parent"),
        "matches_historical_parent": manifest.get("historical_parent") == HISTORICAL_PARENT,
        "manifest_generator_source_sha256_matches": manifest.get(
            "manifest_generator_source_sha256"
        )
        == sha256_file(generator_path),
        "no_unique_causal_root_or_new_method_success_claim": manifest.get(
            "no_unique_causal_root_or_new_method_success_claim"
        ),
        "preserved_failed_attempt_count": len(preserved_failed_attempts),
        "all_preserved_failed_attempts_marked_not_deleted": all_preserved,
    }
    protected_refs_ok = all(item["matches_expected"] for item in refs.values())
    required_files = [
        MANIFEST_PATH,
        MANIFEST_GENERATOR,
        "docs/trackocd_cvpr/RESEARCH_LOG.md",
        "docs/trackocd_cvpr/CVPR0_ROOT_CAUSE_REPORT.md",
    ]
    files_ok = all((repo / path).is_file() for path in required_files)
    status = (
        "PASS_LEGACY_RESEARCH_PRESERVATION_VERIFIED"
        if current_branch == "research/cvpr-trackocd-autonomous-v2"
        and parent_exists
        and ancestry_verified
        and protected_refs_ok
        and manifest_binding["matches_historical_parent"]
        and manifest_binding["manifest_generator_source_sha256_matches"]
        and manifest_binding["no_unique_causal_root_or_new_method_success_claim"] is True
        and all_preserved
        and files_ok
        else "INCOMPLETE_LEGACY_PRESERVATION_AUDIT"
    )

    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": {
            "git_metadata_only": True,
            "large_streams_opened": False,
            "data_or_weights_read": False,
            "tao_test_read": False,
            "external_processes_touched": False,
            "refs_moved": False,
            "model_selection_performed": False,
        },
        "repository": {
            "branch": current_branch,
            "head": current_head,
            "historical_parent": HISTORICAL_PARENT,
            "historical_parent_exists": parent_exists,
            "current_head_is_descendant_of_historical_parent": ancestry_verified,
        },
        "protected_refs": refs,
        "manifest_binding": manifest_binding,
        "preservation_evidence": [file_evidence(repo, path) for path in required_files],
        "boundary": {
            "parent_namespace_was_not_assumed": not git_exists(
                repo, "cat-file", "-e", f"{HISTORICAL_PARENT}:outputs/trackocd_cvpr"
            ),
            "interpretation": "Lineage and preservation metadata are verified; this does not accept any scientific goal or metric.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/legacy_preservation_audit_20261026.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = build_audit(repo)
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output)}))
    return 0 if result["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
