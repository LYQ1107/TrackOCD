#!/usr/bin/env python3
"""Publish only aggregate identities/resources of a fully sealed CPU prediction."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_core.rpn_bytetrack_full_runtime import RUN, load_plan, verify_prediction, owned_output_bytes

SUMMARY = ROOT / "outputs/trackocd_core/audit/rpn_bytetrack_full_val_prediction_summary.json"


def main():
    if SUMMARY.exists(): raise ValueError("Preserve completed prediction summary")
    config, plan, records, digest = load_plan()
    prediction, sealed = verify_prediction(config, plan, records, digest)
    supervisor = json.loads((RUN / "prediction_supervisor.json").read_text())
    result = {"schema_version": "trackocd.core.rpn_bytetrack_full_prediction_summary.v1",
              "status": prediction["status"], "videos": len(sealed), "images": prediction["production_frame_updates"],
              "total_frame_updates_including_probe": prediction["total_frame_updates"], "raw_detection_rows": prediction["raw_detections"],
              "prediction_rows": sum(s["prediction_rows"] for s in sealed), "compressed_npz_bytes": sum(s["npz_bytes"] for s in sealed),
              "empty_output_frames": sum(s["frame_statistics"]["empty_output_frames"] for s in sealed),
              "all_classical_markers_and_detector_lineage_verified": True, "input_arrays_unchanged": prediction["input_arrays_unchanged"],
              "torch_imported": prediction["torch_imported"], "tracker_has_learned_weights": False, "nn_frozen_state_claim": False,
              "config_sha256": digest, "prediction_manifest_sha256": sha256_file(RUN / "prediction_manifest.json"),
              "source_prediction_manifest_sha256": config["input_manifest_sha256"], "source_checkpoint_sha256": config["source_checkpoint_sha256"],
              "ordered_source_and_detector_array_identity_sha256": prediction["ordered_source_and_detector_array_identity_sha256"],
              "tracker_source_sha256": config["tracker_source_sha256"], "tracker_parameters": config["tracker_parameters"],
              "runtime": prediction["runtime"], "causality_probe": prediction["causality_probe"], "boundary": prediction["boundary"],
              "cadence_boundary": config["cadence_boundary"], "quality_contract": config["quality_contract"], "provenance": config["provenance"],
              "resources": {"cpu_workers": 1, "gpu_used": False, "fresh_worker": True,
                            "worker_seconds": prediction["elapsed_seconds"], "worker_peak_rss_bytes": prediction["peak_host_rss_bytes"],
                            "supervisor_seconds": supervisor["elapsed_seconds"], "supervisor_sampled_peak_owned_tree_rss_bytes": supervisor["peak_owned_tree_rss_bytes"],
                            "private_allocated_bytes_before_evaluation": owned_output_bytes()},
              "worker_returncode": supervisor["worker_returncode"], "supervisor_error": supervisor["error"],
              "independent_gt_evaluation_complete": False, "primary_freeze_permitted": False, "scientific_pass_permitted": False,
              "formal_feature_cache_started": False, "ocd_or_m9_metrics": False}
    atomic_json(SUMMARY, result)
    print(json.dumps({k:result[k] for k in ("status", "videos", "images", "prediction_rows", "compressed_npz_bytes", "resources")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
