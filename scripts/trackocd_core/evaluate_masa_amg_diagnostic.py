#!/usr/bin/env python3
"""One CPU, sealed SAM-grid clips and unchanged canonical three-route audit."""
from __future__ import annotations
from collections import defaultdict
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
from src.trackocd_core.amg_physical_diagnostic import completed_video
from src.trackocd_core.physical_diagnostic import observed_purity
from scripts.trackocd_core.run_masa_amg_diagnostic import RUN, load_plan, PUBLIC
from scripts.trackocd_core.evaluate_masa_physical_qualification import read_projection, CANONICAL
from scripts.trackocd_core.audit_frontend import TRACK_ROOT, length_statistics
from scripts.trackocd_core.audit_frontend_coverage import match_video
from scripts.trackocd_core.install_masa_runtime import allocated_bytes

SUMMARY = ROOT / "outputs/trackocd_core/audit/masa_amg_physical_diagnostic_result.json"


def resource_guard(config, started):
    mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    limits = config["limits"]
    if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["max_evaluator_host_rss_bytes"]
            or time.monotonic() - started > limits["max_evaluator_seconds"]
            or mem["MemAvailable"] < mem["MemTotal"] * limits["minimum_ram_headroom_fraction"]
            or allocated_bytes([RUN]) > limits["max_output_bytes"]):
        raise RuntimeError("Registered single-CPU evaluator resource guard")


def main():
    started = time.monotonic()
    if SUMMARY.exists() or (RUN / "canonical_gt").exists():
        raise ValueError("Preserve completed/attempted evaluator; no automatic overwrite")
    config, plan, digest = load_plan()
    prediction_path = RUN / "prediction_manifest.json"
    prediction = json.loads(prediction_path.read_text()); supervisor = json.loads((RUN / "supervisor.json").read_text())
    sealed = [completed_video(RUN, v, digest) for v in plan["videos"]]
    if (prediction["status"] != "SEALED_BOUNDED_AMG_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION"
            or prediction["real_image_forwards_started"] != 64 or prediction["real_image_forwards_completed"] != 64
            or prediction["config_sha256"] != digest or any(s is None for s in sealed) or sealed != prediction["videos"]
            or supervisor["worker_returncode"] != 0 or supervisor["error"] is not None
            or sha256_file(prediction_path) != supervisor["full_prediction_seal_sha256"]):
        raise ValueError("All new pixels must seal before independent GT access")
    historical_path = ROOT / config["historical_result"]
    native_run = ROOT / config["historical_native_run"]
    if (sha256_file(historical_path) != config["historical_result_sha256"]
            or sha256_file(native_run / "prediction_manifest.json") != config["historical_native_manifest_sha256"]):
        raise ValueError("Historical independent baselines changed")
    historical = json.loads(historical_path.read_text())
    native = json.loads((native_run / "prediction_manifest.json").read_text())
    if native["status"] != "SEALED_BOUNDED_PHYSICAL_PREDICTIONS_NOT_QUALIFICATION" or native["real_image_forwards_started"] != 64:
        raise ValueError("Historical native baseline incomplete")
    native_files = {v["video_id"]: native_run / v["npz_filename"] for v in native["videos"]}
    old_hashes = {name: {r["video_id"]: r["sha256"] for r in rows["source_npz"]} for name, rows in historical["results"].items()}
    for vid in config["video_ids"]:
        if (sha256_file(native_files[vid]) != old_hashes["MASA_NATIVE"][vid]
                or sha256_file(TRACK_ROOT / f"video_{vid:04d}.npz") != old_hashes["PANDAS_BT_FG0"][vid]):
            raise ValueError("Do not rerun/replace baseline bytes")
    for relative, expected in (("trackeval/datasets/tao_ow.py", config["canonical_adapter_sha256"]),
                               ("trackeval/metrics/hota.py", config["canonical_hota_sha256"])):
        if sha256_file(CANONICAL / relative) != expected:
            raise ValueError("Previously tested canonical evaluator changed")
    resource_guard(config, started)
    # First GT/role access occurs only after all new/old sealed sources checked.
    annotation = Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json")
    roles_path = ROOT / "configs/trackocd_core/roles.json"
    if sha256_file(annotation) != config["annotation_sha256"] or sha256_file(roles_path) != config["roles_sha256"]:
        raise ValueError("Frozen GT/role evidence changed")
    gt, roles = json.loads(annotation.read_text()), json.loads(roles_path.read_text())
    known, novel = set(roles["known_ids"]), set(roles["novel_ids"])
    selected_images = {i["image_id"] for v in plan["videos"] for i in v["images"]}
    subset = copy.deepcopy(gt)
    subset["images"] = [i for i in subset["images"] if int(i["id"]) in selected_images]
    subset["videos"] = [v for v in subset["videos"] if int(v["id"]) in config["video_ids"]]
    subset["annotations"] = [a for a in subset["annotations"] if int(a["image_id"]) in selected_images]
    selected_tracks = {int(a["track_id"]) for a in subset["annotations"]}
    subset["tracks"] = [t for t in subset["tracks"] if int(t["id"]) in selected_tracks]
    gt_folder = RUN / "canonical_gt"; gt_folder.mkdir()
    atomic_json(gt_folder / "validation.json", subset)
    if sha256_file(gt_folder / "validation.json") != historical["canonical_gt_subset_sha256"]:
        raise ValueError("Not identical original clipped GT universe")
    del gt
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "int"): np.int = int
    sys.path.insert(0, str(CANONICAL))
    import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(CANONICAL.resolve()):
        raise ValueError("Wrong TrackEval imported; never pip substitution")
    by_image = defaultdict(list)
    for ann in subset["annotations"]:
        by_image[int(ann["image_id"])].append(ann)
    metric, results = trackeval.metrics.HOTA(), {}
    for name in ("MASA_SAM_GRID", "MASA_NATIVE", "PANDAS_BT_FG0"):
        rows, lengths, targets, matched, purity_frames, sources = [], [], [], {}, [], []
        for video in plan["videos"]:
            vid = video["video_id"]
            path = (RUN / "shards" / f"video_{vid:04d}.npz" if name == "MASA_SAM_GRID"
                    else native_files[vid] if name == "MASA_NATIVE" else TRACK_ROOT / f"video_{vid:04d}.npz")
            frames = read_projection(path, video["images"])
            sources.append({"video_id": vid, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
            _, counts = np.unique(np.concatenate([frames[i["image_id"]][0] for i in video["images"]]), return_counts=True)
            lengths.extend(counts.tolist()); grouped = {}
            for image in video["images"]:
                iid = image["image_id"]; ids, boxes, scores = frames[iid]
                for identity, box, score in zip(ids, boxes, scores):
                    x, y, x2, y2 = map(float, box)
                    rows.append({"image_id": iid, "video_id": vid, "track_id": int(identity),
                                 "bbox": [x, y, x2 - x, y2 - y], "score": float(score), "category_id": 1})
                anns = by_image[iid]; gt_boxes = []
                for ann in anns:
                    x, y, w, h = map(float, ann["bbox"]); gt_boxes.append([x, y, x + w, y + h])
                    cat = int(ann["category_id"])
                    if cat in known | novel:
                        key = f"{vid}_{ann['track_id']}"
                        target = grouped.setdefault(key, {"key": key, "category": cat, "role": "known" if cat in known else "novel", "boxes": {}})
                        if target["category"] != cat: raise ValueError("Inconsistent GT category")
                        target["boxes"][iid] = gt_boxes[-1]
                purity_frames.append({"pred_ids": [(vid << 32) + int(i) for i in ids], "pred_boxes": boxes,
                    "gt_ids": [(vid << 32) + int(a["track_id"]) for a in anns], "gt_categories": [int(a["category_id"]) for a in anns], "gt_boxes": gt_boxes})
            targets.extend(grouped.values())
            matched.update(match_video(list(grouped.values()), {i: (r[0], r[1]) for i, r in frames.items()})["matches"])
            resource_guard(config, started)
        tracker_root = RUN / ("canonical_" + name); data_folder = tracker_root / name / "data"; data_folder.mkdir(parents=True)
        atomic_json(data_folder / "pred.json", rows)
        options = trackeval.datasets.TAO_OW.get_default_dataset_config()
        options.update(GT_FOLDER=str(gt_folder), TRACKERS_FOLDER=str(tracker_root), TRACKERS_TO_EVAL=[name],
                       TRACKER_SUB_FOLDER="data", SPLIT_TO_EVAL="val", SUBSET="all", MAX_DETECTIONS=300, PRINT_CONFIG=False)
        dataset = trackeval.datasets.TAO_OW(options)
        per_sequence, per_video, raw_count, processed_count = {}, [], 0, 0
        for seq in dataset.seq_list:
            raw = dataset.get_raw_seq_data(name, seq); data = dataset.get_preprocessed_seq_data(raw, "object")
            score = metric.eval_sequence(data); per_sequence[seq] = score
            raw_count += sum(len(v) for v in raw["tracker_ids"]); processed_count += data["num_tracker_dets"]
            per_video.append({"video_id": int(dataset.seq_name_to_seq_id[seq]), "gt_rows": data["num_gt_dets"],
                              "evaluated_pred_rows": data["num_tracker_dets"], **{k: float(np.mean(score[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")}})
            resource_guard(config, started)
        combined = metric.combine_sequences(per_sequence); coverage = {}
        for role in ("known", "novel"):
            group = [t for t in targets if t["role"] == role]; covered = sum(matched.get(t["key"], {}).get("reliable", False) for t in group)
            coverage[role] = {"gt_clip_tracks": len(group), "reliably_observed": covered,
                              "missing_or_unreliable": len(group) - covered, "coverage": covered / len(group) if group else None}
        length = length_statistics(np.asarray(lengths, dtype=np.int64))
        length.update(scope="Only fixed selected annotated-cadence clip projection, not full-video lifetime", counts_come_from_full_frame_stream_not_only_annotated_frames=False)
        results[name] = {"canonical_tracking": {k: float(np.mean(combined[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")},
                         "canonical_raw_combined": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in combined.items()},
                         "coverage": coverage, "purity": observed_purity(purity_frames), "annotated_projection_lengths": length,
                         "per_video": sorted(per_video, key=lambda r: r["video_id"]), "raw_prediction_rows": raw_count,
                         "canonical_evaluated_prediction_rows": processed_count, "canonical_preprocessing_removed_rows": raw_count - processed_count,
                         "source_npz": sources}
    differences = {name: {k: abs(results[name]["canonical_tracking"][k] - historical["results"][name]["canonical_tracking"][k])
                           for k in ("HOTA", "AssA", "DetA", "DetRe")} for name in ("MASA_NATIVE", "PANDAS_BT_FG0")}
    replay_ok = all(v <= config["baseline_absolute_tolerance"] for row in differences.values() for v in row.values())
    for name in ("MASA_NATIVE", "PANDAS_BT_FG0"):
        replay_ok = replay_ok and results[name]["coverage"] == historical["results"][name]["coverage"] and results[name]["purity"] == historical["results"][name]["purity"]
    result = {"schema_version": "trackocd.core.masa_amg_physical_diagnostic_result.v1",
              "status": "BOUNDED_AMG_DIAGNOSTIC_COMPLETE_NOT_PRIMARY_QUALIFICATION" if replay_ok else "INCOMPARABLE_BASELINE_REPLAY_MISMATCH",
              "scope": config["scope"], "video_ids": config["video_ids"], "selected_images": len(selected_images),
              "selected_gt_rows": len(subset["annotations"]), "results": results, "config_sha256": digest,
              "prediction_manifest_sha256": sha256_file(prediction_path), "canonical_gt_subset_sha256": sha256_file(gt_folder / "validation.json"),
              "annotation_sha256": config["annotation_sha256"], "roles_sha256": config["roles_sha256"],
              "canonical_adapter_sha256": config["canonical_adapter_sha256"], "canonical_hota_sha256": config["canonical_hota_sha256"],
              "historical_baselines_reproduced": replay_ok, "baseline_absolute_metric_differences": differences,
              "sampling_boundary": config["sampling_boundary"], "provenance_boundary": config["provenance_boundary"],
              "quality_contract": config["quality_contract"], "labels_for_model_input_or_tuning": False,
              "primary_freeze_permitted": False, "scientific_pass_permitted": False, "ocd_or_m9_metrics": False,
              "resources": {"cpu_workers": 1, "gpu_used": False, "wall_seconds": time.monotonic() - started,
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    resource_guard(config, started); atomic_json(SUMMARY, result)
    print(json.dumps({"status": result["status"], "results": {k: {q: v[q] for q in ("canonical_tracking", "coverage")} for k, v in results.items()}, "resources": result["resources"]}), flush=True)
    return 0 if replay_ok else 1


if __name__ == "__main__":
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    raise SystemExit(main())
