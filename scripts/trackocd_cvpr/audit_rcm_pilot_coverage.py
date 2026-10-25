#!/usr/bin/env python3
"""Audit whether the bounded R/C/M pilot actually exercised M supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "trackocd.cvpr.rcm-pilot-coverage.v1"
SOURCE = "outputs/trackocd_cvpr/cvpr5/joint_rcm_policy_pilot_v1_repair2/result.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_audit(repo: Path) -> dict[str, Any]:
    source_path = repo / SOURCE
    result = json.loads(source_path.read_text(encoding="utf-8"))
    summary = result.get("summary") or {}
    curve = summary.get("loss_curve") or []
    episodes = summary.get("episodes") or []
    m_loss_steps = sum(row.get("M_loss") is not None for row in curve)
    positive_other_video = sum(
        int(row.get("positive_other_video_groups_this_frame", 0)) for row in curve
    )
    m_reason_counts: dict[str, int] = {}
    for episode in episodes:
        for key, value in (episode.get("M_loss_plan_reasons") or {}).items():
            m_reason_counts[key] = m_reason_counts.get(key, 0) + int(value)
    exercised = m_loss_steps > 0 and positive_other_video > 0
    return {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_RCM_M_SUPPORT_COVERAGE" if exercised else "FAIL_RCM_PILOT_M_SUPPORT_COVERAGE_ZERO",
        "source": {
            "path": SOURCE,
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "pilot_status": result.get("status"),
        },
        "coverage": {
            "optimizer_steps": int(summary.get("optimizer_steps", 0)),
            "loss_curve_rows": len(curve),
            "m_loss_steps": int(m_loss_steps),
            "positive_other_video_groups": int(positive_other_video),
            "m_loss_plan_reasons": dict(sorted(m_reason_counts.items())),
            "joint_graph_reached_r_c_m": summary.get("joint_graph_reached_R_C_and_M"),
        },
        "boundary": {
            "m_branch_scientifically_exercised": bool(exercised),
            "pilot_is_development_or_goal_acceptance": summary.get("development_or_goal_acceptance") is True,
            "new_nonKnown_semantics_read": summary.get("dev_final_Val_Test_or_new_nonKnown_semantics_read") is True,
            "model_selection_performed": result.get("model_selection_or_goal_acceptance") is True,
            "goal_a_b_c_accepted": False,
        },
        "interpretation": (
            "The pilot exercised R/C numeric gradients and the M module's dependency path, "
            "but had no M support-agreement loss or cross-video positive support. It is "
            "therefore an engineering integration result, not evidence for self-correcting "
            "persistent memory. A legal cross-video Known-support fold must be frozen before "
            "any M mechanism claim or selection."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trackocd_cvpr/cvpr5/rcm_pilot_coverage_audit_20261026.json"),
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
