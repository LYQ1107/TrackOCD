#!/usr/bin/env python3
"""Independent one-CPU identical-GT canonical comparison of four fixed streams."""
from __future__ import annotations
import copy
import json
import os
import resource
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.rpn_bytetrack import completed_video, read_detector_video, detector_digest
from src.trackocd_core.evaluation.physical_clip import score_route
from scripts.trackocd_core.run_rpn_bytetrack_diagnostic import RUN, load_plan
from scripts.trackocd_core.audit_frontend import TRACK_ROOT
from scripts.trackocd_core.evaluate_masa_physical_qualification import CANONICAL
from scripts.trackocd_core.install_masa_runtime import allocated_bytes
from scripts.trackocd_core.smoke_masa_amg import memory

SUMMARY = ROOT / "outputs/trackocd_core/audit/rpn_bytetrack_diagnostic_result.json"


def resource_guard(config, started):
    mem, limits = memory(), config["limits"]
    if (time.monotonic() - started > limits["max_evaluator_seconds"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["max_evaluator_host_rss_bytes"]
            or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]
            or allocated_bytes([RUN]) > limits["max_output_bytes"]):
        raise RuntimeError("Registered single-CPU evaluator resource guard")


def main():
    started = time.monotonic()
    if SUMMARY.exists() or (RUN / "canonical_gt").exists(): raise ValueError("Preserve any evaluator attempt; no overwrite")
    config, plan, raw_files, digest = load_plan()
    prediction_path = RUN / "prediction_manifest.json"
    prediction = json.loads(prediction_path.read_text()); supervisor = json.loads((RUN / "supervisor.json").read_text())
    sealed = [completed_video(RUN, v, digest, config["tracker_source_sha256"]) for v in plan["videos"]]
    if (any(s is None for s in sealed) or sealed != prediction["videos"] or prediction["status"] != "SEALED_BOUNDED_RPN_BYTETRACK_NOT_QUALIFICATION"
            or prediction["production_frame_updates"] != 64 or prediction["total_frame_updates"] != 72 or not prediction["causality_probe"]["pass"]
            or prediction["config_sha256"] != digest or supervisor["worker_returncode"] != 0 or supervisor["error"] is not None
            or supervisor["prediction_seal_sha256"] != sha256_file(prediction_path)):
        raise ValueError("Entire new classical prediction must seal before GT")
    for video, marker in zip(plan["videos"], sealed):
        if (sha256_file(raw_files[video["video_id"]]) != marker["source_npz_sha256"]
                or detector_digest(read_detector_video(raw_files[video["video_id"]], video)) != marker["source_detector_arrays_sha256"]):
            raise ValueError("New association must use identical frozen raw detections")
    history_path = ROOT / config["old_three_route_result"]
    if sha256_file(history_path) != config["old_three_route_result_sha256"]: raise ValueError("Historical comparison changed")
    history = json.loads(history_path.read_text())
    files = {"RPN_BYTETRACK": {v["video_id"]: RUN / "shards" / f"video_{v['video_id']:04d}.npz" for v in plan["videos"]},
             "MASA_NATIVE": raw_files, "PANDAS_BT_FG0": {v["video_id"]: TRACK_ROOT / f"video_{v['video_id']:04d}.npz" for v in plan["videos"]},
             "MASA_SAM_GRID": {v["video_id"]: ROOT / "outputs/trackocd_core/physical/masa_amg_val4_first16/shards" / f"video_{v['video_id']:04d}.npz" for v in plan["videos"]}}
    for name in history["results"]:
        for row in history["results"][name]["source_npz"]:
            if sha256_file(files[name][row["video_id"]]) != row["sha256"]: raise ValueError("Old frozen NPZ changed")
    for relative, expected in (("trackeval/datasets/tao_ow.py", config["canonical_adapter_sha256"]),
                               ("trackeval/metrics/hota.py", config["canonical_hota_sha256"])):
        if sha256_file(CANONICAL / relative) != expected: raise ValueError("Previously tested canonical evaluator changed")
    resource_guard(config, started)
    # First GT/role access in this independent process, after all seals/source checks.
    annotation, roles_path = Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json"), ROOT / "configs/trackocd_core/roles.json"
    if sha256_file(annotation) != config["annotation_sha256"] or sha256_file(roles_path) != config["roles_sha256"]: raise ValueError("Frozen GT/roles changed")
    gt, roles = json.loads(annotation.read_text()), json.loads(roles_path.read_text())
    selected_images = {i["image_id"] for v in plan["videos"] for i in v["images"]}
    subset = copy.deepcopy(gt)
    subset["images"] = [i for i in subset["images"] if int(i["id"]) in selected_images]
    subset["videos"] = [v for v in subset["videos"] if int(v["id"]) in config["video_ids"]]
    subset["annotations"] = [a for a in subset["annotations"] if int(a["image_id"]) in selected_images]
    tids = {int(a["track_id"]) for a in subset["annotations"]}; subset["tracks"] = [t for t in subset["tracks"] if int(t["id"]) in tids]
    gt_folder = RUN / "canonical_gt"; gt_folder.mkdir(); atomic_json(gt_folder / "validation.json", subset); del gt
    if sha256_file(gt_folder / "validation.json") != history["canonical_gt_subset_sha256"]: raise ValueError("Identical original clipped GT required")
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "int"): np.int = int
    sys.path.insert(0, str(CANONICAL)); import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(CANONICAL.resolve()): raise ValueError("Wrong TrackEval imported")
    known, novel = set(roles["known_ids"]), set(roles["novel_ids"])
    results = {name: score_route(name, paths, plan, subset, known, novel, gt_folder, RUN, trackeval,
                                 lambda: resource_guard(config, started)) for name, paths in files.items()}
    differences = {name: {k: abs(results[name]["canonical_tracking"][k] - old["canonical_tracking"][k]) for k in ("HOTA", "AssA", "DetA", "DetRe")}
                   for name, old in history["results"].items()}
    replay_ok = all(v <= config["baseline_absolute_tolerance"] for row in differences.values() for v in row.values())
    for name, old in history["results"].items():
        replay_ok = replay_ok and results[name]["coverage"] == old["coverage"] and results[name]["purity"] == old["purity"]
    result = {"schema_version": "trackocd.core.rpn_bytetrack_diagnostic_result.v1",
              "status": "BOUNDED_RPN_BYTETRACK_COMPLETE_NOT_PRIMARY_QUALIFICATION" if replay_ok else "INCOMPARABLE_BASELINE_REPLAY_MISMATCH",
              "scope": config["scope"], "selected_images": len(selected_images), "selected_gt_rows": len(subset["annotations"]),
              "video_ids": config["video_ids"], "results": results, "config_sha256": digest, "prediction_seal_sha256": sha256_file(prediction_path),
              "canonical_gt_subset_sha256": sha256_file(gt_folder / "validation.json"), "annotation_sha256": config["annotation_sha256"],
              "roles_sha256": config["roles_sha256"], "canonical_adapter_sha256": config["canonical_adapter_sha256"],
              "canonical_hota_sha256": config["canonical_hota_sha256"], "historical_baselines_reproduced": replay_ok,
              "baseline_absolute_metric_differences": differences, "sampling_boundary": config["sampling_boundary"],
              "provenance_boundary": config["provenance_boundary"], "quality_contract": config["quality_contract"],
              "labels_for_model_input_or_tuning": False, "primary_freeze_permitted": False, "scientific_pass_permitted": False,
              "ocd_or_m9_metrics": False, "semantic_feedback": False,
              "resources": {"cpu_workers": 1, "gpu_used": False, "wall_seconds": time.monotonic() - started,
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    resource_guard(config, started); atomic_json(SUMMARY, result)
    print(json.dumps({"status": result["status"], "results": {k: {q: v[q] for q in ("canonical_tracking", "coverage")} for k, v in results.items()}, "resources": result["resources"]}), flush=True)
    return 0 if replay_ok else 1


if __name__ == "__main__":
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    raise SystemExit(main())
