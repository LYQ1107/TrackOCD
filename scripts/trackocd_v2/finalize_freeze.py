#!/usr/bin/env python3
"""Write the immutable TrackOCD v2 Val-to-Test freeze manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _required(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"FINAL_FREEZE prerequisite missing: {label}: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build() -> dict[str, Any]:
    selection_path = OUTPUT_TARGET / "audit/frontend_selection.json"
    representation_path = OUTPUT_TARGET / "audit/representation_decision.json"
    val_selection_path = OUTPUT_TARGET / "audit/val_final_selection.json"
    adapter_path = OUTPUT_TARGET / "audit/semantic_adapter_selection.json"
    controller_path = OUTPUT_TARGET / "audit/controller_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
    representation = json.loads(representation_path.read_text(encoding="utf-8")) if representation_path.is_file() else {}
    if selection.get("status") != "FINAL_PHYSICAL_FRONTEND_SELECTED":
        raise RuntimeError("FINAL_FREEZE requires FINAL_PHYSICAL_FRONTEND_SELECTED")
    if representation.get("status") != "FINAL_REPRESENTATION_SELECTED":
        raise RuntimeError("FINAL_FREEZE requires FINAL_REPRESENTATION_SELECTED")
    if not val_selection_path.is_file():
        raise RuntimeError("FINAL_FREEZE requires Val final selection artifact")
    val_selection = json.loads(val_selection_path.read_text(encoding="utf-8"))
    if val_selection.get("status") not in {"FINAL_VAL_SELECTION", "FINAL_VAL_SELECTION_NO_IMPROVEMENT"}:
        raise RuntimeError(f"Val final selection is not frozen: {val_selection.get('status')}")
    for path, label in ((adapter_path, "semantic adapter selection"), (controller_path, "controller selection")):
        _required(path, label)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema_version": "trackocd.v2.final_freeze.v1",
        "status": "FINAL_FREEZE",
        "generated_utc": _now(),
        "git_sha": git_sha,
        "frontend": selection.get("selected_frontend"),
        "representation": {
            "selection": _required(selection_path, "frontend selection"),
            "decision": _required(representation_path, "representation decision"),
            "adapter_selection": _required(adapter_path, "semantic adapter selection"),
        },
        "controller": {"selection": _required(controller_path, "controller selection")},
        "val_selection": _required(val_selection_path, "Val final selection"),
        "stream_orders": ["main", "seed1027", "seed1028", "seed1029"],
        "evaluation_code": {
            "standard": _required(ROOT / "src/trackocd_v2/evaluation/standard_ocd.py", "standard evaluator"),
            "persistent": _required(ROOT / "src/trackocd_v2/evaluation/persistent.py", "persistent evaluator"),
        },
        "test_semantic_accessed": False,
        "test_selection_or_tuning": False,
        "immutable_after_write": True,
    }
    if FREEZE.exists():
        existing = json.loads(FREEZE.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("FINAL_FREEZE already exists with different contents; refusing overwrite")
        return existing
    atomic_json(FREEZE, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    ensure_output_layout()
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
