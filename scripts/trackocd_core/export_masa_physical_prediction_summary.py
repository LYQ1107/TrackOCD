#!/usr/bin/env python3
"""Small sealed-prediction receipt: no raw boxes, labels or input paths."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_core.run_masa_physical_qualification import RUN, PUBLIC


def main():
    path = RUN / "prediction_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["status"] != "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM":
        raise ValueError("Not the sealed complete stream")
    workers = manifest["worker_receipts"]
    attempt = workers[0]["attempt"]
    supervisor = json.loads((RUN / "attempts" / attempt / "supervisor.json").read_text())
    if supervisor["status"] != "COMPLETE_SUPERVISED_FULL_VAL_PREDICTION" or any(supervisor["worker_returncodes"]):
        raise ValueError("Successful owned supervisor receipt required")
    proof = workers[0]["frozen_model"]
    if (len({w["initial_frozen_state_sha256"] for w in workers}) != 1
            or any(w["initial_frozen_state_sha256"] != w["final_frozen_state_sha256"] or w["frozen_model"] != proof for w in workers)):
        raise ValueError("Inconsistent frozen model across workers")
    videos = manifest["videos"]
    output = {"schema_version": "trackocd.core.masa_full_val_prediction_summary.v1", "status": manifest["status"],
              "scope": json.loads(PUBLIC.read_text())["scope"], "preregistration_commit": manifest["preregistration_commit"],
              "config_sha256": manifest["config_sha256"], "private_plan_sha256": manifest["private_plan_sha256"],
              "private_manifest_sha256": sha256_file(path), "sealed_utc": manifest["sealed_utc"],
              "videos": len(videos), "images": manifest["images"], "prediction_rows": manifest["total_prediction_rows"],
              "detection_rows": manifest["total_detection_rows"], "compressed_npz_bytes": sum(v["npz_bytes"] for v in videos),
              "all_complete_video_states_unchanged": all(v["frame_statistics"]["frozen_state_unchanged"] for v in videos),
              "frame_statistics": {k: sum(v["frame_statistics"][k] for v in videos) for k in ("frames", "roi_cap50_frames", "empty_detection_frames", "empty_tracking_frames")},
              "worker_resources": [{k: w[k] for k in ("worker_index", "status", "elapsed_seconds", "actual_forwards_started_this_attempt", "actual_forwards_completed_this_attempt", "reused_complete_videos", "peak_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "initial_frozen_state_sha256", "final_frozen_state_sha256")} for w in workers],
              "frozen_model": proof, "supervisor": supervisor,
              "boundary": {"gt_or_role_model_input": False, "training": False, "test_access": False, "new_weight_or_image_copy": False,
                           "physical_tuning": False, "m9_or_semantic_result": False, "primary_qualification_claim": False},
              "independent_gt_evaluation_complete": False}
    atomic_json(ROOT / "outputs/trackocd_core/audit/masa_full_val_prediction_summary.json", output)
    print(json.dumps({k: output[k] for k in ("status", "videos", "images", "prediction_rows", "detection_rows", "compressed_npz_bytes", "frame_statistics")}))


if __name__ == "__main__":
    main()
