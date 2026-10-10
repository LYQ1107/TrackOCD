#!/usr/bin/env python3
"""Fresh one-CPU complete-quality audit after entire classical prediction seals."""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import copy
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.rpn_bytetrack import read_detector_video, detector_digest
from src.trackocd_core.evaluation.physical_full import score_video, combine_route, compare_reference
from scripts.trackocd_core.rpn_bytetrack_full_runtime import RUN, SUMMARY, load_plan, verified_source, verify_prediction, resource_guard, supervise
from scripts.trackocd_core.audit_frontend import TRACK_ROOT
from scripts.trackocd_core.evaluate_masa_physical_qualification import CANONICAL


def worker():
    started = time.monotonic(); config, plan, records, digest = load_plan()
    prediction, sealed = verify_prediction(config, plan, records, digest)
    history_path = ROOT / config["historical_full_result"]
    if sha256_file(history_path) != config["historical_full_result_sha256"]:
        raise ValueError("Existing complete baseline result changed")
    history = json.loads(history_path.read_text())
    old = {name: {r["video_id"]: r for r in history["results"][name]["per_video"]} for name in ("MASA_NATIVE", "PANDAS_BT_FG0")}
    sources = {"RPN_BYTETRACK": {}, "MASA_NATIVE": {}, "PANDAS_BT_FG0": {}}
    for video, seal in zip(plan["videos"], sealed):
        resource_guard(config, "evaluation", started)
        vid = video["video_id"]
        native = verified_source(video, config, records)
        if detector_digest(read_detector_video(native, video)) != seal["source_detector_arrays_sha256"]:
            raise ValueError("Identical frozen raw detections before GT")
        sources["RPN_BYTETRACK"][vid] = RUN / "shards" / seal["npz_filename"]
        sources["MASA_NATIVE"][vid] = native
        sources["PANDAS_BT_FG0"][vid] = TRACK_ROOT / f"video_{vid:04d}.npz"
        for name in old:
            source = sources[name][vid]; record = old[name][vid]
            if source.is_symlink() or source.stat().st_size != record["source_npz_bytes"] or sha256_file(source) != record["source_npz_sha256"]:
                raise ValueError("Do not rerun/replace existing complete baseline source")
    for relative, expected in (("trackeval/datasets/tao_ow.py", config["canonical_adapter_sha256"]), ("trackeval/metrics/hota.py", config["canonical_hota_sha256"])):
        if sha256_file(CANONICAL / relative) != expected: raise ValueError("Previously tested canonical source changed")
    # First GT/role reads in the independent fresh evaluator, after all source/seal checks.
    annotation = Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json")
    roles_path = ROOT / "configs/trackocd_core/roles.json"
    if sha256_file(annotation) != config["annotation_sha256"] or sha256_file(roles_path) != config["roles_sha256"]:
        raise ValueError("Unchanged entire GT/role inputs required")
    gt, roles = json.loads(annotation.read_text()), json.loads(roles_path.read_text())
    known, novel = set(roles["known_ids"]), set(roles["novel_ids"])
    videos = {int(v["id"]): v for v in gt["videos"]}; images = {int(i["id"]): i for i in gt["images"]}
    by_video, tracks = defaultdict(list), defaultdict(list)
    for ann in gt["annotations"]: by_video[int(ann["video_id"])].append(ann)
    for track in gt["tracks"]: tracks[int(track["video_id"])].append(track)
    if (set(videos) != {v["video_id"] for v in plan["videos"]} or set(images) != {i["image_id"] for v in plan["videos"] for i in v["images"]}
            or len(gt["annotations"]) != config["expected_gt_rows"] or history["gt_rows"] != len(gt["annotations"])):
        raise ValueError("Entire unchanged Val GT denominator, not a selected target subset")
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "int"): np.int = int
    sys.path.insert(0, str(CANONICAL)); import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(CANONICAL.resolve()): raise ValueError("Wrong canonical import")
    names = tuple(sources); sequences = {n: {} for n in names}; summaries = {n: [] for n in names}; lengths = {n: Counter() for n in names}
    gt_identity = []; metric = trackeval.metrics.HOTA()
    for index, video in enumerate(plan["videos"]):
        resource_guard(config, "evaluation", started, True)
        vid = video["video_id"]
        subset = {"categories": copy.deepcopy(gt["categories"]), "videos": [copy.deepcopy(videos[vid])],
                  "images": [copy.deepcopy(images[i["image_id"]]) for i in video["images"]],
                  "tracks": copy.deepcopy(tracks[vid]), "annotations": copy.deepcopy(by_video[vid])}
        # Fresh owned disposable scratch only. Original NPZ/GT and result evidence retained.
        with tempfile.TemporaryDirectory(prefix=f".eval_video_{vid}_", dir=RUN) as directory:
            scratch = Path(directory); gt_folder = scratch / "gt"; gt_folder.mkdir()
            atomic_json(gt_folder / "validation.json", subset)
            gt_identity.append({"video_id": vid, "sha256": sha256_file(gt_folder / "validation.json")})
            for name in names:
                sequence, summary, histogram = score_video(name, sources[name][vid], video, subset, known, novel, gt_folder, scratch, trackeval,
                                                           lambda: resource_guard(config, "evaluation", started))
                summaries[name].append(summary); sequences[name][str(vid)] = sequence; lengths[name].update(histogram)
        atomic_json(RUN / "evaluation_progress.json", {"status": "EVALUATING_COMPLETE_SEALED_FULL_VAL_STREAMS", "completed_videos": index + 1,
                    "total_videos": config["videos"], "elapsed_seconds": time.monotonic() - started})
        if (index + 1) % 50 == 0: print(json.dumps({"evaluated_videos": index + 1}), flush=True)
    results = {name: combine_route(metric, sequences[name], summaries[name], lengths[name]) for name in names}
    reproduction = {name: compare_reference(results[name], history["results"][name], config["evaluation"]["full_baseline_absolute_tolerance"]) for name in old}
    replay_ok = all(row["pass"] for row in reproduction.values())
    for result in results.values():
        if result["coverage"]["known"]["gt_tracks"] != config["evaluation"]["expected_known_gt_tracks"] or result["coverage"]["novel"]["gt_tracks"] != config["evaluation"]["expected_novel_gt_tracks"]:
            replay_ok = False
    result = {"schema_version": "trackocd.core.rpn_bytetrack_full_val_physical_result.v1",
              "status": "FULL_VAL_RPN_BYTETRACK_QUALITY_AUDIT_COMPLETE_NOT_AUTOMATIC_PRIMARY_PASS" if replay_ok else "INCOMPARABLE_FULL_BASELINE_REPLAY_MISMATCH",
              "scope": config["scope"], "videos": config["videos"], "images": config["images"], "gt_rows": len(gt["annotations"]), "results": results,
              "historical_full_baselines_reproduced": replay_ok, "baseline_reproduction": reproduction,
              "config_sha256": digest, "prediction_manifest_sha256": sha256_file(RUN / "prediction_manifest.json"),
              "source_prediction_manifest_sha256": config["input_manifest_sha256"], "historical_full_result_sha256": config["historical_full_result_sha256"],
              "annotation_sha256": config["annotation_sha256"], "roles_sha256": config["roles_sha256"],
              "canonical_adapter_sha256": config["canonical_adapter_sha256"], "canonical_hota_sha256": config["canonical_hota_sha256"],
              "per_video_identical_gt": gt_identity, "provenance": config["provenance"], "cadence_boundary": config["cadence_boundary"],
              "quality_contract": config["quality_contract"], "labels_for_model_input_or_tuning": False, "primary_freeze_permitted": False,
              "scientific_pass_permitted": False, "ocd_or_m9_metrics": False, "semantic_feedback": False,
              "resources": {"cpu_workers": 1, "gpu_used": False, "fresh_worker": True, "wall_seconds": time.monotonic() - started,
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    resource_guard(config, "evaluation", started, True); atomic_json(SUMMARY, result)
    if SUMMARY.stat().st_size > config["limits"]["max_public_result_bytes"]: raise RuntimeError("Registered public summary size guard")
    print(json.dumps({"status": result["status"], "metrics": {n: {k:r[k] for k in ("canonical_tracking", "coverage")} for n,r in results.items()}, "resources": result["resources"]}), flush=True)
    return 0 if replay_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--worker", action="store_true"); parser.add_argument("--preregistration-commit")
    args = parser.parse_args()
    if args.worker: raise SystemExit(worker())
    if not args.preregistration_commit: parser.error("--preregistration-commit required")
    raise SystemExit(supervise("evaluation", Path(__file__).resolve(), args.preregistration_commit))
