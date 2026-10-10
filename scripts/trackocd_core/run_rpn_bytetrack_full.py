#!/usr/bin/env python3
"""Preregistered complete cached CPU replay, no detector/training/GT/pixels."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.rpn_bytetrack import read_detector_video, detector_digest, causal_probe, seal_video
from src.trackocd_core.rpn_bytetrack_full import replay_video
from scripts.trackocd_core.run_rpn_bytetrack_diagnostic import input_barrier
from scripts.trackocd_core.rpn_bytetrack_full_runtime import RUN, load_plan, verified_source, resource_guard, supervise


def worker():
    started = time.monotonic(); config, plan, records, digest = load_plan(); input_barrier()
    result = {"schema_version": "trackocd.core.rpn_bytetrack_full_prediction.v1", "status": "RUNNING",
              "runtime": config["runtime"],
              "config_sha256": digest, "tracker_source_sha256": config["tracker_source_sha256"], "videos": [],
              "production_frame_updates": 0, "total_frame_updates": 0, "raw_detections": 0,
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "primary_freeze_permitted": False,
              "boundary": {"training": False, "gt_or_semantic_or_native_track_input": False, "test_access": False,
                           "pixel_inference": False, "new_weights": False, "score_repair": False, "parameter_search": False,
                           "gpu": False, "foreign_process_interference": False}}
    identity = hashlib.sha256()
    try:
        for index, video in enumerate(plan["videos"]):
            resource_guard(config, "prediction", started, True)
            source = verified_source(video, config, records); before = sha256_file(source)
            data = read_detector_video(source, video); arrays_hash = detector_digest(data)
            identity.update(str(video["video_id"]).encode() + before.encode() + arrays_hash.encode())
            if index == 0:
                result["causality_probe"] = causal_probe(data, config["tracker_parameters"])
                result["total_frame_updates"] += 8
            arrays, stats = replay_video(data, video, config["tracker_parameters"], lambda: resource_guard(config, "prediction", started))
            if sha256_file(source) != before: raise ValueError("Frozen raw source changed during association")
            result["production_frame_updates"] += len(video["images"])
            result["total_frame_updates"] += len(video["images"]); result["raw_detections"] += len(data["det_score"])
            result["videos"].append(seal_video(RUN, video, arrays, digest, before, config["tracker_source_sha256"], stats))
            result["completed_videos"] = len(result["videos"])
            atomic_json(RUN / "prediction_manifest.json", result)
            if (index + 1) % 50 == 0: print(json.dumps({"completed_videos": index + 1, "production_frame_updates": result["production_frame_updates"]}), flush=True)
        if (result["production_frame_updates"] != config["images"] or result["raw_detections"] != config["raw_detections"]
                or result["total_frame_updates"] != config["limits"]["total_frame_updates_with_probe"] or "torch" in sys.modules
                or identity.hexdigest() != config["ordered_source_and_detector_array_identity_sha256"]
                or sha256_file(ROOT / config["tracker_source"]) != config["tracker_source_sha256"]
                or sha256_file(ROOT / config["input_source"] / "prediction_manifest.json") != config["input_manifest_sha256"]):
            raise ValueError("Complete source/array identity, exact calls and unchanged CPU tracker required")
        resource_guard(config, "prediction", started, True)
        result.update(status="SEALED_COMPLETE_FULL_VAL_CLASSICAL_RPN_BYTETRACK", torch_imported=False,
                      input_arrays_unchanged=True, ordered_source_and_detector_array_identity_sha256=identity.hexdigest())
    except Exception as exc:
        import traceback
        traceback.print_exc(); result.update(status="FAILED_FULL_CLASSICAL_RPN_BYTETRACK", error=type(exc).__name__ + ": " + str(exc))
    result.update(elapsed_seconds=time.monotonic() - started, peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(RUN / "prediction_manifest.json", result)
    return 0 if result["status"].startswith("SEALED") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true"); parser.add_argument("--preregistration-commit")
    args = parser.parse_args()
    if args.worker: raise SystemExit(worker())
    if not args.preregistration_commit: parser.error("--preregistration-commit required")
    raise SystemExit(supervise("prediction", Path(__file__).resolve(), args.preregistration_commit))
