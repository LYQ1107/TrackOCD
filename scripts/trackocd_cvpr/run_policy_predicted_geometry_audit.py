#!/usr/bin/env python3
"""Run the frozen category-free geometry audit on policy_train_pseudo videos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.audit_predicted_train_geometry import audit  # noqa: E402


CONFIG = ROOT / "configs/trackocd_cvpr/policy_predicted_geometry_audit_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assignment_path = ROOT / cfg["role_assignment"]
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    fixed_ids = sorted(int(value) for value in assignment["assignment"]["video_ids"][cfg["role"]])
    if fixed_ids != sorted(cfg["video_ids"]):
        raise ValueError("policy video allowlist differs from frozen assignment")
    if sha256(assignment_path) != cfg["role_assignment_sha256"]:
        raise ValueError("frozen role assignment hash mismatch")
    geometry = ROOT / cfg["geometry_gt"]
    if sha256(geometry) != cfg["geometry_gt_sha256"]:
        raise ValueError("frozen geometry-only GT hash mismatch")
    output = ROOT / cfg["output"]
    if output.exists():
        raise RuntimeError("refusing to overwrite the policy geometry audit receipt")
    result = audit(
        geometry,
        ROOT / cfg["stream_root"],
        ROOT / cfg["reconciliation"],
        output,
        video_ids=set(fixed_ids),
    )
    result["frozen_config"] = str(CONFIG.relative_to(ROOT))
    result["frozen_config_sha256"] = sha256(CONFIG)
    result["role_binding"] = {"role": cfg["role"], "video_ids": fixed_ids, "assignment_sha256": cfg["role_assignment_sha256"]}
    result["registered_boundaries"] = cfg["boundaries"]
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "videos": result["scope"]["videos_processed"], "aggregate_geometry": result["aggregate_geometry"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
