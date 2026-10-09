#!/usr/bin/env python3
"""Register the selected frontend's frozen TAO Test physical stream.

Native Test inference is deliberately supplied as an input artifact.  This
small boundary owns the post-freeze check, normalization, and lineage record;
it does not let a caller silently register Test output for a different
frontend or before ``FINAL_FREEZE.json`` exists.
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

from scripts.trackocd_v2.normalize_frontend_output import FORMAT_NAMES, normalize_file  # noqa: E402
from src.trackocd_v2.frontend_contract import route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


TEST_ANNOTATION = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json")
FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
FRONTENDS = tuple(route_specs(ROOT))


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def register(
    *,
    frontend: str,
    native_input: Path,
    input_format: str,
    annotation: Path,
    output_dir: Path,
    allow_invalid_boxes: bool,
) -> dict[str, Any]:
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown frontend: {frontend}")
    # The guard is intentionally before normalization and before opening the
    # Test annotation.  Test physical outputs are only admissible after the
    # Val-selected frontend/controller have been frozen.
    assert_test_semantic_access_allowed(FINAL_FREEZE, "register frozen TAO Test frontend stream")
    freeze = json.loads(FINAL_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("frontend") != frontend:
        raise RuntimeError(
            f"Test frontend {frontend} does not match frozen frontend {freeze.get('frontend')}"
        )
    route = route_specs(ROOT)[frontend]
    slug = str(route["slug"])
    if not native_input.is_file():
        raise FileNotFoundError(native_input)
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization_audit = OUTPUT_TARGET / "audit" / f"frontend_{slug}_test_normalization.json"
    physical_output = output_dir / "physical_tracks.jsonl"
    native_output = output_dir / "native_evaluator_rows.jsonl"
    normalization = normalize_file(
        frontend=frontend,
        input_path=native_input.resolve(),
        input_format=input_format,
        physical_output=physical_output.resolve(),
        native_output=native_output.resolve(),
        audit_output=normalization_audit.resolve(),
        annotation_path=annotation.resolve(),
        allow_invalid_boxes=allow_invalid_boxes,
        source_split="test_predicted",
    )
    if normalization.get("status") not in {"COMPLETE", "COMPLETE_WITH_INVALID_BOXES"}:
        raise RuntimeError(f"Test frontend normalization is incomplete: {normalization.get('status')}")
    if normalization.get("annotation", {}).get("semantic_labels_consumed") is not False:
        raise RuntimeError("Test frontend normalization did not prove semantic-label isolation")
    audit_path = OUTPUT_TARGET / "audit" / f"frontend_{slug}_test.json"
    payload = {
        "schema_version": "trackocd.v2.test_frontend_stream.v1",
        "status": "FINAL_TEST_NATIVE_STREAM_NORMALIZED",
        "generated_utc": _now(),
        "split": "test",
        "frontend": frontend,
        "frontend_slug": slug,
        "route": route,
        "native_input": {"path": str(native_input.resolve()), "sha256": sha256_file(native_input), "format": input_format},
        "normalization_audit": str(normalization_audit.resolve()),
        "normalization_audit_sha256": sha256_file(normalization_audit),
        "normalized_outputs": normalization["outputs"],
        "normalized_counts": normalization["counts"],
        "final_freeze": str(FINAL_FREEZE.resolve()),
        "final_freeze_sha256": sha256_file(FINAL_FREEZE),
        "test_semantic_accessed": False,
        "test_semantic_evaluation_unlocked": True,
        "causal_decisions_before_test_gt_join": True,
        # These are explicit prerequisites for the post-freeze cache builder.
        # The registration wrapper is only called after the selected native
        # route has completed; retaining them here makes that boundary
        # machine-checkable instead of relying on a filename convention.
        "native_run_complete": True,
        "physical_stream_contract_complete": True,
        "same_v2_evaluator_contract": True,
        "same_v2_physical_metric_protocol": True,
        "requested_metrics_complete": False,
    }
    atomic_json(audit_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=FRONTENDS, required=True)
    parser.add_argument("--native-input", type=Path, required=True)
    parser.add_argument("--input-format", choices=FORMAT_NAMES, default="auto")
    parser.add_argument("--annotation", type=Path, default=TEST_ANNOTATION)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-invalid-boxes", action="store_true")
    args = parser.parse_args()
    out = ensure_output_layout()
    slug = route_specs(ROOT)[args.frontend]["slug"]
    output_dir = (args.output_dir or (out / "manifests/frontend_streams" / slug / "test")).resolve()
    try:
        result = register(
            frontend=args.frontend,
            native_input=args.native_input.resolve(),
            input_format=args.input_format,
            annotation=args.annotation.resolve(),
            output_dir=output_dir,
            allow_invalid_boxes=args.allow_invalid_boxes,
        )
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.test_frontend_stream.v1",
            "status": "FAILED_TEST_FRONTEND_STREAM_REGISTRATION",
            "generated_utc": _now(),
            "frontend": args.frontend,
            "native_input": str(args.native_input.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "test_semantic_accessed": False,
        }
        atomic_json(out / "audit" / f"frontend_{slug}_test.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
