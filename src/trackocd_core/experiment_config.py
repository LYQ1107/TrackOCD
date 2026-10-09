"""Resolve the single registered correction without altering base protocol."""
import json
from pathlib import Path

from src.trackocd_v2.io import sha256_file


def representation_config(root: Path, path: Path) -> dict:
    config = json.loads(path.read_text())
    if "parent_config" not in config:
        return config
    parent = root / config["parent_config"]
    if sha256_file(parent) != config["parent_config_sha256"]:
        raise ValueError("Original preregistration changed")
    allowed = {"schema_version", "status", "parent_config", "parent_config_sha256", "rationale",
               "teacher_geometry_loss_weight", "teacher_geometry_loss", "output_directory",
               "root_cause_correction_rounds_used", "unchanged", "original_checkpoints_and_results_preserved",
               "after_this_round_further_root_cause_correction_forbidden"}
    if set(config) - allowed or config.get("root_cause_correction_rounds_used") != 1:
        raise ValueError("Only one loss-only controlled correction; no protocol/capacity/split edits")
    base = json.loads(parent.read_text())
    return base | config
