#!/usr/bin/env python3
"""Audit the frozen legacy H3 checkpoint before v2 bake-off use.

This is a provenance/shape audit only.  It deliberately emits no evaluation
metric: the historical H3 representation is not the v2 representation, so a
successful checkpoint load must not be mistaken for a comparable result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import load_ids  # noqa: E402


SELECTION = Path("/data2/usr_for_deadline/trackocd_phase90/project_outputs/audit/h3_repro_frozen_selection.json")
KNOWN_IDS = ROOT / "data/tao_ow_ocd_v1/splits/known_ids.json"
OLD_SUPPORTED_IDS = ROOT / "data/iclr27_phase19r/sources/supported_known_ids.json"
OLD_STREAM = ROOT / "src/iclr27_phase19r/data/stream.py"
V2_BUILDER = ROOT / "scripts/trackocd_v2/build_common_features.py"


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(x) for x in shape] if shape is not None else None


def _load_checkpoint(path: Path) -> dict[str, Any]:
    # Keep torch local to this audit so the rest of the v2 CPU tooling remains
    # usable with the system Python installation.
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid H3 checkpoint payload: {path}")
    return payload


def audit(selection: dict[str, Any]) -> dict[str, Any]:
    v2_known = load_ids(KNOWN_IDS)
    old_supported = load_ids(OLD_SUPPORTED_IDS)
    folds = []
    shape_contracts = []
    for item in selection.get("folds", []):
        path = Path(str(item["checkpoint"]))
        row: dict[str, Any] = {
            "fold": int(item["fold"]),
            "checkpoint": str(path.resolve()),
            "checkpoint_exists": path.exists(),
            "checkpoint_sha256": sha256_file(path) if path.exists() else None,
        }
        if path.exists():
            payload = _load_checkpoint(path)
            model = payload["model"]
            shapes = {key: _shape(value) for key, value in model.items() if key in {
                "known_prototypes", "active_known_mask",
                "track_encoder.gru.weight_ih_l0", "track_encoder.gru.weight_hh_l0",
                "open_world_router.net.0.weight", "action_head.0.weight",
            }}
            row.update({
                "architecture": payload.get("architecture"),
                "route": payload.get("route"),
                "step": payload.get("step"),
                "support_mode": payload.get("support_mode"),
                "manifest_sha256": payload.get("manifest_sha256"),
                "semantic_contract_sha256": payload.get("semantic_contract_sha256"),
                "model_shapes": shapes,
                "known_count": int(shapes.get("known_prototypes", [0])[0]) if shapes.get("known_prototypes") else 0,
                "raw_dim": int(shapes.get("known_prototypes", [0, 0])[1]) if shapes.get("known_prototypes") and len(shapes["known_prototypes"]) > 1 else 0,
                "geometry_dim_from_gru": int(shapes.get("track_encoder.gru.weight_ih_l0", [0, 0])[1] - 768) if shapes.get("track_encoder.gru.weight_ih_l0") else 0,
            })
            shape_contracts.append(row)
        folds.append(row)

    all_shapes_match = bool(shape_contracts) and all(
        row.get("known_count") == 48
        and row.get("raw_dim") == 768
        and row.get("geometry_dim_from_gru") == 15
        for row in shape_contracts
    )
    representation = {
        "legacy_h3_raw": "0.8 * DINO CLS + 0.2 * DINO ROI, normalized",
        "v2_raw": "DINOv2 x_norm_clstoken crop descriptor, normalized",
        "raw_contract_matches": False,
        "legacy_h3_geometry": "15 Phase19R fields normalized with fold-local Phase19R fit statistics",
        "v2_geometry": "not present in common-feature JSON; v2 geometry is audited separately",
        "geometry_contract_matches": False,
        "legacy_source_sha256": sha256_file(OLD_STREAM),
        "v2_builder_sha256": sha256_file(V2_BUILDER),
    }
    return {
        "schema_version": "trackocd.v2.current_model_contract.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": "H3_REPRO",
        "selection_artifact": str(SELECTION.resolve()),
        "selection_artifact_sha256": sha256_file(SELECTION),
        "selection_status": selection.get("status"),
        "folds": folds,
        "shape_contract": {"all_folds_match": all_shapes_match, "expected": {"known_count": 48, "raw_dim": 768, "geometry_dim": 15}},
        "known_role_coverage": {
            "v2_known_count": len(v2_known),
            "legacy_supported_count": len(old_supported),
            "overlap_count": len(v2_known & old_supported),
            "uncovered_v2_known_ids": sorted(v2_known - old_supported),
            "coverage_fraction": len(v2_known & old_supported) / max(len(v2_known), 1),
        },
        "representation_contract": representation,
        "decision": "STRUCTURALLY_LOADABLE_BUT_NOT_COMPARABLE",
        "required_before_metric": [
            "use an explicit adapter or retrain under the v2 raw/geometry contract",
            "report 48/78 known coverage and legacy provenance",
            "keep Test semantic labels inaccessible until FINAL_FREEZE",
        ],
        "test_semantic_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=SELECTION)
    args = parser.parse_args()
    out = ensure_output_layout()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = audit(selection)
    atomic_json(out / "audit/current_model_contract.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["shape_contract"]["all_folds_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
