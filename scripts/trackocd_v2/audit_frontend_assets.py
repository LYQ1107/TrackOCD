#!/usr/bin/env python3
"""Audit physical-frontend assets before the TrackOCD v2 bake-off.

The repository contains several historical frontend outputs, but they were
not all produced on the v2 stream or with the v2 OCD evaluator.  This script
registers their provenance and reports historical numbers only as references;
it never selects a final frontend from incomparable metrics.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import evaluator_contract, route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


PRED_STREAM = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl"
PRED_STREAM_AUDIT = OUTPUT_TARGET / "audit/predicted_stream_statistics.json"
PRED_FEATURE_PAUSE = OUTPUT_TARGET / "audit/predicted_feature_pause.json"
SIMOWT_SUMMARY = ROOT / "outputs/iclr27_closure/tracking_eval/simowt/summary.csv"
SIMOWT_METRICS = ROOT / "outputs/iclr27_closure/tracking_eval/simowt/class_agnostic_metrics.json"
SIMOWT_PREDICTIONS = ROOT / "outputs/simowt/val_predictions.json"
OVTR_Q0 = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
OVTR_TRACK_EVAL = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval/OVTR_Q0"
COVTRACK_TCO = ROOT / "outputs/iclr27_phase4p/covtrack_tco/covtrack_tco_metrics.csv"
BACKBONE_DOC = ROOT / "docs/iclr27_phase4p/BACKBONE_EVAL_COMPARISON.md"
OVTR_DOC = ROOT / "docs/iclr27_phase67/PHASE68_OVTR_FULL_SEQUENCE_BASELINE_REPORT.md"
COVTRACK_DOC = ROOT / "docs/iclr27_phase4p/COVTRACK_TRAJECTORY_OBJECTNESS_AUDIT.md"
PRIOR_ART_DOC = ROOT / "docs/iclr27_phase14b/PRIOR_ART.md"
CHECKPOINTS = {
    "SimOWT/Q0": ROOT / "data/iclr27_phase14b/checkpoints/simowt_weight.pth",
    "OVTR-native": ROOT / "data/iclr27_phase14b/checkpoints/ovtr_5_frame.pth",
    "COVTrack-native": ROOT / "data/iclr27_phase14b/checkpoints/covtrack_ctao_public.pth",
}
REPOS = {
    "SimOWT": ROOT / "third_party/SimOWT",
    "OVTR": ROOT / "third_party/research_refs_phase4n/OVTR",
    "COVTrack": ROOT / "third_party/research_refs_phase4n/COVTrack",
}
OUTPUT = OUTPUT_TARGET / "audit/frontend_asset_audit.json"


def _metadata(path: Path, *, hash_small: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
    }
    if resolved.is_file() and hash_small and stat.st_size <= 20 * 1024 * 1024:
        result["sha256"] = sha256_file(resolved)
    else:
        result["sha256"] = None
        result["sha256_note"] = "omitted for directory or large historical asset; source path and byte size retained"
    return result


def _git_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
        status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, stderr=subprocess.STDOUT).strip()
        return {"path": str(path.resolve()), "exists": True, "commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"path": str(path.resolve()), "exists": True, "status": "GIT_METADATA_UNAVAILABLE", "error": str(exc)}


def _first_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return None


def _simowt_export_schema() -> dict[str, Any]:
    row = _first_json_object(PRED_STREAM)
    if row is None:
        return {"status": "UNAVAILABLE"}
    embedding_fields = sorted(
        key for key in row
        if any(token in key.lower() for token in ("embed", "feature", "appearance", "reid"))
    )
    return {
        "status": "INSPECTED",
        "public_stream_fields": sorted(row),
        "embedding_like_fields_in_export": embedding_fields,
        "legal_exported_category_agnostic_appearance_embedding": bool(embedding_fields),
        "source_stream_sha256": sha256_file(PRED_STREAM),
        "note": "SimOWT internal embedding tensors are not present in the exported public stream; merge_simowt_output.py drops them.",
    }


def _csv_first(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader, None)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    out = ensure_output_layout()
    pause = json.loads(PRED_FEATURE_PAUSE.read_text(encoding="utf-8")) if PRED_FEATURE_PAUSE.exists() else {}
    stream_audit = json.loads(PRED_STREAM_AUDIT.read_text(encoding="utf-8")) if PRED_STREAM_AUDIT.exists() else {}
    stream_totals = stream_audit.get("stream_totals", {})
    simowt_summary = _csv_first(SIMOWT_SUMMARY)
    result = {
        "schema_version": "trackocd.v2.frontend_asset_audit.v1",
        "status": "AUDIT_COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_position": "frontend asset audit precedes FINAL_PHYSICAL_FRONTEND and formal common-feature cache",
        "bakeoff_contract": evaluator_contract(ROOT),
        "route_specs": route_specs(ROOT),
        "predicted_stream": {
            "source": _metadata(PRED_STREAM),
            "v2_audit": str(PRED_STREAM_AUDIT.resolve()) if PRED_STREAM_AUDIT.exists() else None,
            "tracks": stream_totals.get("total_tracks"),
            "observations": stream_totals.get("total_observations"),
            "videos": stream_totals.get("videos"),
            "feature_pause": pause.get("status"),
        },
        "repositories": {name: _git_info(path) for name, path in REPOS.items()},
        "checkpoints": {name: _metadata(path, hash_small=False) for name, path in CHECKPOINTS.items()},
        "historical_assets": {
            "SimOWT/Q0": {
                "physical_stream": "public SimOWT/Q0 merged predicted track stream",
                "checkpoint": _metadata(CHECKPOINTS["SimOWT/Q0"], hash_small=False),
                "prediction_asset": _metadata(SIMOWT_PREDICTIONS, hash_small=False),
                "export_schema": _simowt_export_schema(),
                "historical_trackeval": {
                    "summary": _metadata(SIMOWT_SUMMARY),
                    "class_agnostic_metrics": _metadata(SIMOWT_METRICS),
                    "all": {
                        "HOTA": _number((simowt_summary or {}).get("HOTA")),
                        "DetA": _number((simowt_summary or {}).get("DetA")),
                        "AssA": _number((simowt_summary or {}).get("AssA")),
                        "OWTA": _number((simowt_summary or {}).get("OWTA")),
                    },
                    "metric_units": "historical TrackEval aggregate; not v2 Standard/Persistent OCD",
                },
                "v2_execution": "EXPORTED_STREAM_REGISTERED; no new SimOWT inference launched in this reprioritized stage",
            },
            "OVTR-native": {
                "physical_stream": "historical OVTR Q0 full-sequence output",
                "checkpoint": _metadata(CHECKPOINTS["OVTR-native"], hash_small=False),
                "prediction_asset": _metadata(OVTR_Q0, hash_small=False),
                "trackeval_asset_root": _metadata(OVTR_TRACK_EVAL, hash_small=False),
                "historical_metrics": {
                    "TETA_combined": 35.34,
                    "TETA_novel": 30.47,
                    "AssA_macro": 1.282274,
                    "Novel_Observability": None,
                    "Persistent_Observability": None,
                    "sources": [str(BACKBONE_DOC.relative_to(ROOT)), str(OVTR_DOC.relative_to(ROOT))],
                    "metric_units": "TETA percentage and per-class TrackEval macro are historical, incompatible scales",
                },
                "v2_execution": "HISTORICAL_Q0_ASSET_AUDIT_ONLY; not a v2 physical-stream rerun",
            },
            "COVTrack-native": {
                "physical_stream": "historical COVTrack native TAO output and trajectory-objectness audit",
                "checkpoint": _metadata(CHECKPOINTS["COVTrack-native"], hash_small=False),
                "historical_metrics": {
                    "TETA_combined": 37.13,
                    "TETA_novel": 32.50,
                    "AssA": None,
                    "Novel_Observability": None,
                    "Persistent_Observability": None,
                    "sources": [str(BACKBONE_DOC.relative_to(ROOT)), str(COVTRACK_DOC.relative_to(ROOT))],
                    "metric_units": "TETA percentage; prior COVTrack TCO values are detector/FP diagnostics, not v2 OCD",
                },
                "trajectory_objectness_asset": _metadata(COVTRACK_TCO),
                "environment_status": "historical local checkpoint construction was blocked by incompatible MMCV/clip dependencies; no new native run here",
                "v2_execution": "HISTORICAL_ASSET_AUDIT_ONLY",
            },
            "COVTrack-NoSemantic": {
                "physical_stream": "COVTrack native stream with semantic path disabled",
                "historical_metrics": {
                    "TETA_combined": None,
                    "TETA_novel": None,
                    "AssA": None,
                    "Novel_Observability": None,
                    "Persistent_Observability": None,
                },
                "v2_execution": "NOT_YET_RUN",
                "note": "No native no-semantic output is registered; no value is inferred from COVTrack-native results.",
            },
        },
        "requested_bakeoff_metrics": {
            "category_free_physical": ["OWTA", "AssA", "DetRe", "LocA"],
            "native_category_aware_reference": ["TETA"],
            "observability": ["Novel Observability", "Persistent Observability"],
            "ocd_safety": ["Commit-CT", "False Assignment"],
        },
        "comparability": {
            "same_v2_predicted_stream": False,
            "same_v2_evaluator_contract": False,
            "same_v2_physical_metric_protocol": False,
            "independent_native_streams_allowed": True,
            "same_v2_standard_persistent_evaluator": False,
            "all_requested_metrics_available_on_one_protocol": False,
            "selection_status": "DEFER_FINAL_PHYSICAL_FRONTEND",
            "reason": "Available OVTR/COVTrack/SimOWT numbers are historical references with different output, annotation, and metric protocols; COVTrack-NoSemantic has no run.",
        },
        "sources": {
            "backbone_comparison": _metadata(BACKBONE_DOC),
            "ovtr_phase68_report": _metadata(OVTR_DOC),
            "covtrack_audit": _metadata(COVTRACK_DOC),
            "prior_art": _metadata(PRIOR_ART_DOC),
        },
        "test_semantic_accessed": False,
    }
    atomic_json(out / "audit/frontend_asset_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
