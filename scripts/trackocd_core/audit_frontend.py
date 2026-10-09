#!/usr/bin/env python3
"""Read-only M1 evidence/fragmentation audit of existing Val frontends.

No inference, tracker training, semantic scores, Test data, crops or feature
arrays are consumed. This does not freeze a frontend or optimize PANDAS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.frontend_contract import route_specs
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_core.audit_assets import PANDAS, file_record

FROZEN_PANDAS = PANDAS / "outputs/score_fix_full"
TRACK_ROOT = FROZEN_PANDAS / "bytetrack/BT-FG-0"
EVAL_ROOT = FROZEN_PANDAS / "trackeval/BT-FG-0"


def length_statistics(lengths: np.ndarray) -> dict:
    count = len(lengths)
    total = int(lengths.sum())
    levels = [0, .1, .25, .5, .75, .9, .95, .99, 1]
    quantiles = np.quantile(lengths, levels) if count else [None] * len(levels)
    return {
        "physical_tracks": count,
        "observations": total,
        "mean_observations_per_track": total / count if count else None,
        "observation_length_quantiles": dict(zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
            [float(v) if v is not None else None for v in quantiles],
        )),
        "exact_length_histogram": {str(k): int(v) for k, v in sorted(Counter(lengths.tolist()).items())},
        "single_observation_tracks": int(np.sum(lengths == 1)),
        "single_observation_fraction": float(np.mean(lengths == 1)) if count else None,
        "at_most_four_observations_fraction": float(np.mean(lengths <= 4)) if count else None,
        "at_most_sixteen_observations_fraction": float(np.mean(lengths <= 16)) if count else None,
        "counts_come_from_full_frame_stream_not_only_annotated_frames": True,
    }


def audit_npz_tracks(root: Path) -> dict:
    paths = sorted(root.glob("video_*.npz"))
    if not paths:
        raise ValueError("No existing frozen NPZ tracks")
    counts = []
    videos = set()
    frames = observations = payload_bytes = max_track_id_bytes = 0
    for path in paths:
        video_id = int(path.stem.removeprefix("video_"))
        if video_id in videos:
            raise ValueError("Duplicate video archive")
        videos.add(video_id)
        with zipfile.ZipFile(path) as archive:
            track_bytes = archive.getinfo("track_id.npy").file_size
        max_track_id_bytes = max(max_track_id_bytes, track_bytes)
        # One sequential worker; never materialize a whole dataset of rows.
        if track_bytes > 256 * 2**20:
            raise ValueError("Per-video ID payload exceeds the diagnostic RAM plan")
        with np.load(path, allow_pickle=False) as data:
            frame_index = data["frame_index"]
            offsets = data["frame_offsets"]
            track_ids = data["track_id"]
            if len(offsets) != len(frame_index) + 1 or offsets[0] != 0 or offsets[-1] != len(track_ids):
                raise ValueError("Frame/track offset mismatch: " + path.name)
            if np.any(np.diff(frame_index) <= 0) or np.any(np.diff(offsets) < 0):
                raise ValueError("Non-causal frame order or offsets: " + path.name)
            _, lengths = np.unique(track_ids, return_counts=True)
            # IDs are local to their video: counts must never merge videos.
            counts.append(lengths)
            frames += len(frame_index)
            observations += len(track_ids)
        payload_bytes += path.stat().st_size
    lengths = np.concatenate(counts)
    stats = length_statistics(lengths)
    if stats["observations"] != observations:
        raise ValueError("Track length conservation failure")
    return {
        "videos": len(videos), "video_ids": sorted(videos), "frames": frames,
        **stats,
        "existing_compressed_npz_bytes": payload_bytes,
        "largest_track_id_member_bytes": max_track_id_bytes,
        "namespace": "(video_id, local_track_id)",
        "per_video_arrays_only": ["frame_index", "frame_offsets", "track_id"],
        "full_npz_file_hashes_computed": False,
        "semantic_scores_or_prototype_ids_consumed": False,
    }


def recovered_nas_evidence(root: Path) -> dict | None:
    """Preserve old claims, but qualify the frontend using actual source evidence."""
    path = root / "outputs/trackocd_core/audit/nas_m1_provenance.json"
    if not path.is_file():
        return None
    evidence = json.loads(path.read_text())
    for record in evidence["metadata_files"]:
        payload = root / record["private_local_copy"]
        if payload.stat().st_size != record["bytes"] or sha256_file(payload) != record["sha256"]:
            raise ValueError("Recovered NAS metadata byte mismatch: " + record["name"])
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trackocd_core/audit/frontend.json")
    args = parser.parse_args()
    started = time.monotonic()
    mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    if mem["MemAvailable"] < mem["MemTotal"] * .25:
        raise RuntimeError("Resource wait: less than 25% RAM headroom")
    runtime_path = TRACK_ROOT / "bytetrack_runtime.json"
    metrics_path = EVAL_ROOT / "trackeval_summary.json"
    export_path = EVAL_ROOT / "export_trackeval_complete.json"
    runtime = json.loads(runtime_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    export = json.loads(export_path.read_text())
    if metrics.get("split") != "val" or metrics.get("subset") != "all" or metrics.get("class_mode") != "class-agnostic":
        raise ValueError("Not the frozen full-Val class-agnostic reference")
    if runtime.get("score_key") != "foreground_scores" or export.get("semantic_mapping_used") is not False:
        raise ValueError("Unexpected foreground-score/association contract")
    track_stats = audit_npz_tracks(TRACK_ROOT)
    for key, runtime_key in (("videos", "videos"), ("frames", "frames"), ("physical_tracks", "tracks"), ("observations", "track_rows")):
        if track_stats[key] != runtime[runtime_key]:
            raise ValueError("Frozen runtime/current array count mismatch: " + key)
    prediction = PANDAS / export["prediction_path"]
    # Streaming hash only; do not parse or duplicate the large prediction JSON.
    prediction_hash = sha256_file(prediction)
    if prediction_hash != export["prediction_sha256"]:
        raise ValueError("Frozen exported physical predictions changed")
    raw_hota = metrics["raw_combined"]["HOTA"]
    coverage_path = ROOT / "outputs/trackocd_core/audit/frontend_coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.is_file() else None
    reference = {
        "role": "FROZEN_EXTERNAL_REFERENCE_ONLY_NOT_SELECTED_MAIN_FRONTEND",
        "physical_metrics_0_to_1": {
            "HOTA": metrics["metrics"]["HOTA_mean"],
            "AssA": metrics["metrics"]["AssocA_mean"],
            "DetA": metrics["metrics"]["DetA_mean"],
            "DetRe": float(np.mean(raw_hota["DetRe"])),
            "IDF1": metrics["metrics"]["IDF1"],
        },
        "tracking_array_statistics": track_stats,
        "annotated_evaluation_rows": export["track_rows_exported"],
        "annotated_evaluation_frames": export["annotated_frames_exported"],
        "prediction_sha256_currently_verified": prediction_hash,
        "runtime_evidence": file_record(runtime_path),
        "tracking_evidence": file_record(metrics_path),
        "export_evidence": file_record(export_path),
        "inference_config": {k: runtime[k] for k in (
            "track_thresh", "low_thresh", "match_thresh", "track_buffer", "frame_rate",
            "score_key", "new_track_thresh", "filter_mot_aspect",
        )},
        "no_inference_or_parameter_search_performed": True,
        "base_novel_track_coverage": coverage["by_role"] if coverage else {
            "status": "NOT_MEASURED", "reason": "Run the separate fixed geometry-only coverage diagnostic"},
        "coverage_evidence": file_record(coverage_path),
        "coverage_is_evaluator_diagnostic_not_method_input": True,
        "frontend_qualification": "NOT_QUALIFIED_AS_CORE_PRIMARY_BY_THIS_AUDIT",
    }
    nas = json.loads((ROOT / "outputs/trackocd_core/audit/nas_source_audit.json").read_text())
    provenance = recovered_nas_evidence(ROOT)
    candidate_specs = route_specs(ROOT)
    covtrack = {
        "status": "BLOCKED_PROVENANCE_AND_ASSET_RECOVERY",
        "declared_legacy_contract": candidate_specs["COVTrack-native"],
        "contract_booleans_are_not_native_vocabulary_or_supervision_proof": True,
        "source_stream_snapshot": nas["physical_stream"],
        "source_cache_snapshot": nas["formal_cache"],
        "native_config_vocabulary_and_supervision_verified": False,
        "causal_native_inference_verified": False,
        "physical_tracking_metrics_and_role_coverage_verified": False,
        "present_on_a100": False,
        "missing_evidence": [
            "Val frontend selection/stage/metrics/native-run JSON files",
            "Actual native config and referenced vocabulary/supervision sources",
            "Checkpoint/config/physical stream lineage tied to the actual run",
            "Native causal-inference evidence and Base/Novel coverage/fragmentation",
        ],
    }
    if provenance is not None:
        covtrack.update({
            "status": provenance["qualification"]["native_core_comparability"],
            "legal_gate": provenance["qualification"]["native_legal_gate"],
            "historical_clean_declarations_contradicted_by_actual_source": True,
            "native_vocabulary_source_inspected": True,
            "native_config_vocabulary_and_supervision_verified": False,
            "exact_historical_config_bytes_verified": False,
            "training_supervision_fully_verified": False,
            "vocabulary_evidence": provenance["vocabulary"],
            "historical_runtime_log_evidence": provenance["historical_runtime_log"],
            "static_causality_evidence": provenance["causality"],
            "historical_native_metrics": provenance["historical_native_metrics"],
            "metrics_recomputed_on_a100": False,
            "metadata_byte_identity_verified_on_a100": True,
            "source_provenance_evidence": file_record(ROOT / "outputs/trackocd_core/audit/nas_m1_provenance.json"),
            "missing_evidence": [
                "Exact historical configuration/prompt hashes and complete training supervision history",
                "Historical stage/input snapshots for the two differing reference hashes",
                "A lawful category-agnostic replacement stream; removing output labels is insufficient",
            ],
        })
    result = {
        "schema_version": "trackocd.core.frontend_audit.v1",
        "status": "M1_DIAGNOSTICS_COMPLETE_FRONTEND_QUALIFICATION_BLOCKED" if provenance else "M1_EVIDENCE_INCOMPLETE",
        "frontend_gate": "BLOCKED_FRONTEND_QUALITY",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pandas_bytetrack_frozen_reference": reference,
        "covtrack_native_recovery_candidate": covtrack,
        "other_candidates": {
            "SimOWT/Q0": {"status": "BLOCKED_ASSET_AND_PROVENANCE_RECOVERY", "historical_track_count": 649378,
                          "stream_present_on_a100": False,
                          "historical_summary": provenance["simowt_historical_summary"] if provenance else None},
            "OVTR-native": {"status": "INCOMPARABLE", "reason": "Legacy contract explicitly labels it vocabulary-assisted; not a clean main physical frontend"},
            "COVTrack-NoSemantic": {"status": "INCOMPARABLE_VOCABULARY_ASSISTED_REFERENCE" if provenance else "BLOCKED_PROVENANCE_AND_ASSET_RECOVERY",
                                    "reason": "Association toggle does not remove the shared detector's Val Novel vocabulary"},
            "MASA/AED": {"status": "UNAVAILABLE", "reason": "No existing lawful TAO output/checkpoint route established"},
        },
        "selected_frontend": None,
        "FROZEN_PHYSICAL_FRONTEND_written": False,
        "gt_track_feasibility_exception_permitted": True,
        "gt_feasibility_is_not_predicted_main_result": True,
        "resources": {
            "worker_count": 1, "gpu_used": False,
            "ram_plan": "Sequential NPZ ID-column audit; largest member capped at 256 MiB; <2 GiB working plan, retain 25% system headroom",
            "initial_mem_available_kib": mem["MemAvailable"],
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_seconds": time.monotonic() - started,
            "available_disk_bytes": os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        },
        "training_started": False, "feature_extraction_started": False,
        "test_data_accessed": False, "external_process_interference": False,
        "large_assets_copied": False,
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "frontend_gate": result["frontend_gate"],
        "videos": track_stats["videos"], "tracks": track_stats["physical_tracks"],
        "single_observation_fraction": track_stats["single_observation_fraction"],
        "median_observations": track_stats["observation_length_quantiles"]["p50"],
        "wall_seconds": result["resources"]["wall_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
