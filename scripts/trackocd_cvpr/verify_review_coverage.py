#!/usr/bin/env python3
"""Verify the frozen CVPR-1 source-review coverage index.

This is a small metadata-only verifier.  It does not fetch repositories,
open datasets, import reviewed projects, execute their code, or change the
scientific acceptance flags.  It checks that each counted review in the
existing index still has its audit JSON, a pinned repository commit, and
non-empty recorded model/loss/train/evaluation file evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "outputs/trackocd_cvpr/cvpr1/review_coverage_index.json"
OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr1/review_coverage_verification.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(index_path: Path = INDEX) -> dict[str, Any]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    violations: list[str] = []
    counted: list[dict[str, Any]] = []
    audit_hashes: dict[str, str] = {}

    for entry in index.get("entries", []):
        audit_rel = str(entry.get("audit_file", ""))
        audit_path = ROOT / audit_rel
        if not audit_path.is_file():
            violations.append(f"missing_audit:{audit_rel}")
            continue
        audit_hashes[audit_rel] = _sha256(audit_path)
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        if not entry.get("actual_core_scope_review_countable"):
            continue
        if not str(entry.get("repository_commit", "")):
            violations.append(f"missing_pinned_commit:{audit_rel}")
        files = payload.get("actual_files_read")
        if not isinstance(files, list) or not files:
            violations.append(f"missing_actual_files_read:{audit_rel}")
        if int(entry.get("actual_path_scopes_read", 0)) <= 0:
            violations.append(f"empty_scope_count:{audit_rel}")
        counted.append({
            "audit_file": audit_rel,
            "paper_title": entry.get("paper_title"),
            "repository_commit": entry.get("repository_commit"),
            "actual_path_scopes_read": entry.get("actual_path_scopes_read"),
            "licensed_extra_for_inspection_substep": bool(
                entry.get("licensed_extra_for_inspection_substep")
            ),
        })

    required = int(index.get("minimum_required_actual_core_reviews", 0))
    licensed_required = int(index.get("minimum_required_licensed_extras", 0))
    licensed_count = sum(1 for row in counted if row["licensed_extra_for_inspection_substep"])
    if len(counted) < required:
        violations.append(f"count_below_required:{len(counted)}<{required}")
    if licensed_count < licensed_required:
        violations.append(f"licensed_extra_below_required:{licensed_count}<{licensed_required}")
    if int(index.get("scoped_actual_core_reviews", -1)) != len(counted):
        violations.append(
            f"index_count_mismatch:{index.get('scoped_actual_core_reviews')}!={len(counted)}"
        )

    return {
        "schema_version": "trackocd.cvpr1.review-coverage-verification.v1",
        "status": "PASS_MINIMUM_SCOPED_SOURCE_REVIEW_COVERAGE_VERIFIED" if not violations else "FAIL_REVIEW_COVERAGE_INCONSISTENT",
        "index": str(index_path.relative_to(ROOT)),
        "index_sha256": _sha256(index_path),
        "audit_json_sha256": audit_hashes,
        "total_index_entries": len(index.get("entries", [])),
        "counted_actual_core_reviews": len(counted),
        "minimum_required_actual_core_reviews": required,
        "licensed_extra_reviews": licensed_count,
        "minimum_required_licensed_extras": licensed_required,
        "scientific_goal_complete_in_index": bool(index.get("scientific_goal_complete")),
        "source_execution_or_new_training": bool(index.get("source_execution_or_new_training")),
        "counted_reviews": counted,
        "violations": violations,
        "interpretation": (
            "The scoped literature/source-review count is verified; this is not "
            "CVPR-1 stage acceptance, mechanism evidence, frontend qualification, "
            "or Goal A/B/C acceptance."
        ),
    }


def main() -> int:
    result = verify()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
