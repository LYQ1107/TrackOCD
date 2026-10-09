#!/usr/bin/env python3
"""Independent one-CPU full-Val audit, only after every prediction is sealed."""
from __future__ import annotations
from collections import defaultdict
import copy
import json
import os
import resource
import sys
import time
import uuid
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_qualification import completed_video, aggregate_purity
from src.trackocd_core.physical_diagnostic import observed_purity
from scripts.trackocd_core.run_masa_physical_qualification import RUN, PUBLIC, PRIVATE, load_plan, candidate_roots, ENV
from scripts.trackocd_core.audit_frontend import TRACK_ROOT, length_statistics
from scripts.trackocd_core.audit_frontend_coverage import match_video
from scripts.trackocd_core.install_masa_runtime import allocated_bytes

CANONICAL = Path("/data3/liuyeqiang/InterMOT/third_party/MOTIP/TrackEval")


def read_projection(path: Path, images: list[dict]):
    wanted = {i["image_id"] for i in images}
    frames = {}
    with np.load(path, allow_pickle=False) as arrays:
        image_ids, offsets = arrays["image_id"], arrays["frame_offsets"]
        ids, boxes, scores = arrays["track_id"], arrays["boxes"], arrays["score"]
        if len(offsets) != len(image_ids) + 1 or offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid physical offsets; do not fix/drop")
        for index, raw_id in enumerate(image_ids):
            image_id = int(raw_id)
            if image_id not in wanted:
                continue
            if image_id in frames:
                raise ValueError("Duplicate canonical prediction frame")
            begin, end = map(int, offsets[index:index + 2])
            local, bb, ss = ids[begin:end], boxes[begin:end], scores[begin:end]
            if (len(np.unique(local)) != len(local) or len(bb) != len(local) or len(ss) != len(local)
                    or not np.isfinite(bb).all() or not np.isfinite(ss).all()
                    or np.any(bb[:, 2:] <= bb[:, :2]) or np.any((ss < 0) | (ss > 1))):
                raise ValueError("Invalid projected stream; preserve and stop")
            frames[image_id] = (local.copy(), bb.copy(), ss.copy())
    if set(frames) != wanted:
        raise ValueError("Missing canonical prediction frame; no smaller GT universe")
    return frames


def guard(config):
    limits = config["limits"]
    mem = dict(s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())
    stat = os.statvfs(ROOT)
    if (int(mem["MemAvailable"].split()[0]) < .25 * int(mem["MemTotal"].split()[0])
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["max_host_rss_bytes_per_worker"]
            or allocated_bytes([RUN]) > limits["max_output_allocated_bytes"]
            or allocated_bytes(candidate_roots()) > limits["max_candidate_allocated_bytes"]
            or allocated_bytes([ROOT, ENV]) > limits["overall_soft_bytes"]
            or stat.f_bavail * stat.f_frsize < limits["minimum_disk_headroom_bytes"]):
        raise RuntimeError("Independent evaluator registered resource guard")


def evaluate() -> int:
    started = time.monotonic()
    destination = ROOT / "outputs/trackocd_core/audit/masa_full_val_physical_result.json"
    if destination.exists():
        raise ValueError("Preserve completed result; no automatic reevaluation")
    config, plan, digest = load_plan()
    prediction_path = RUN / "prediction_manifest.json"
    prediction = json.loads(prediction_path.read_text())
    if prediction["status"] != "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM" or prediction["config_sha256"] != digest or prediction["images"] != config["images"]:
        raise ValueError("Entire stream must be sealed before independent GT/role access")
    sealed = [completed_video(RUN, v, digest) for v in plan["videos"]]
    if any(r is None for r in sealed) or sealed != prediction["videos"]:
        raise ValueError("Entire video/shard seal mismatch")
    # First annotation/role reads in this independent process, after all predictions checked.
    annotation = Path("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json")
    roles_path = ROOT / "configs/trackocd_core/roles.json"
    if sha256_file(annotation) != config["annotation_sha256"] or sha256_file(roles_path) != config["roles_sha256"]:
        raise ValueError("Frozen evaluator input changed")
    gt, roles = json.loads(annotation.read_text()), json.loads(roles_path.read_text())
    known, novel = set(roles["known_ids"]), set(roles["novel_ids"])
    by_video, by_image = defaultdict(list), defaultdict(list)
    for ann in gt["annotations"]:
        by_video[int(ann["video_id"])].append(ann); by_image[int(ann["image_id"])].append(ann)
    videos = {int(v["id"]): v for v in gt["videos"]}
    images = {int(i["id"]): i for i in gt["images"]}
    tracks = defaultdict(list)
    for track in gt["tracks"]:
        tracks[int(track["video_id"])].append(track)
    if set(videos) != {v["video_id"] for v in plan["videos"]} or set(images) != {i["image_id"] for v in plan["videos"] for i in v["images"]}:
        raise ValueError("Not the full original GT video/image universe")
    if not hasattr(np, "float"): np.float = float
    if not hasattr(np, "int"): np.int = int
    sys.path.insert(0, str(CANONICAL))
    import trackeval
    if not Path(trackeval.__file__).resolve().is_relative_to(CANONICAL.resolve()):
        raise ValueError("Wrong canonical adapter; fresh process required")
    adapter_hash, hota_hash = sha256_file(CANONICAL / "trackeval/datasets/tao_ow.py"), sha256_file(CANONICAL / "trackeval/metrics/hota.py")
    if adapter_hash != config["canonical_adapter_sha256"] or hota_hash != config["canonical_hota_sha256"]:
        raise ValueError("Previously tested canonical evaluator changed")
    evaluation = RUN / "evaluations" / uuid.uuid4().hex; evaluation.mkdir(parents=True)
    scratch = evaluation / "scratch"; gt_folder = scratch / "gt"; gt_folder.mkdir(parents=True)
    # Scratch holds only the current video; no duplicated whole-Val GT/pred JSON.
    names = ("MASA_NATIVE", "PANDAS_BT_FG0")
    metric = trackeval.metrics.HOTA()
    per_sequence = {name: {} for name in names}
    summaries = {name: [] for name in names}; all_lengths = {name: [] for name in names}
    for position, (video, marker) in enumerate(zip(plan["videos"], sealed)):
        guard(config)
        vid = video["video_id"]
        subset = {"categories": copy.deepcopy(gt["categories"]), "videos": [copy.deepcopy(videos[vid])],
                  "images": [copy.deepcopy(images[i["image_id"]]) for i in video["images"]],
                  "tracks": copy.deepcopy(tracks[vid]), "annotations": copy.deepcopy(by_video[vid])}
        atomic_json(gt_folder / "validation.json", subset)
        grouped = {}
        for ann in by_video[vid]:
            category = int(ann["category_id"])
            if category in known | novel:
                key = f"{vid}_{ann['track_id']}"
                target = grouped.setdefault(key, {"key": key, "category": category, "role": "known" if category in known else "novel", "boxes": {}})
                if target["category"] != category: raise ValueError("Inconsistent GT category")
                x, y, w, h = map(float, ann["bbox"]); target["boxes"][int(ann["image_id"])] = [x, y, x + w, y + h]
        for name in names:
            source = RUN / "shards" / marker["npz_filename"] if name == "MASA_NATIVE" else TRACK_ROOT / f"video_{vid:04d}.npz"
            frames = read_projection(source, video["images"])
            rows, purity_frames = [], []
            counts = np.concatenate([frame[0] for frame in frames.values()])
            _, lengths = np.unique(counts, return_counts=True); all_lengths[name].extend(lengths.tolist())
            for image in video["images"]:
                image_id = image["image_id"]; ids, boxes, scores = frames[image_id]
                for local, box, score in zip(ids, boxes, scores):
                    x, y, x2, y2 = map(float, box)
                    rows.append({"image_id": image_id, "video_id": vid, "track_id": int(local), "category_id": 1,
                                 "bbox": [x, y, x2 - x, y2 - y], "score": float(score)})
                annotations = by_image[image_id]; gt_boxes = []
                for ann in annotations:
                    x, y, w, h = map(float, ann["bbox"]); gt_boxes.append([x, y, x + w, y + h])
                purity_frames.append({"pred_ids": ids, "pred_boxes": boxes, "gt_boxes": gt_boxes,
                                      "gt_ids": [int(a["track_id"]) for a in annotations], "gt_categories": [int(a["category_id"]) for a in annotations]})
            data_folder = scratch / name / name / "data"; data_folder.mkdir(parents=True, exist_ok=True)
            atomic_json(data_folder / "pred.json", rows)
            cfg = trackeval.datasets.TAO_OW.get_default_dataset_config()
            cfg.update(GT_FOLDER=str(gt_folder), TRACKERS_FOLDER=str(scratch / name), TRACKERS_TO_EVAL=[name],
                       TRACKER_SUB_FOLDER="data", SPLIT_TO_EVAL="val", SUBSET="all", MAX_DETECTIONS=300, PRINT_CONFIG=False)
            dataset = trackeval.datasets.TAO_OW(cfg)
            if len(dataset.seq_list) != 1: raise ValueError("Single-video canonical evaluation expected")
            raw = dataset.get_raw_seq_data(name, dataset.seq_list[0]); data = dataset.get_preprocessed_seq_data(raw, "object")
            scores = metric.eval_sequence(data); per_sequence[name][str(vid)] = scores
            matches = match_video(list(grouped.values()), {i: (f[0], f[1]) for i, f in frames.items()})["matches"]
            coverage = {}
            for role in ("known", "novel"):
                targets = [r for r in grouped.values() if r["role"] == role]
                covered = sum(matches.get(r["key"], {}).get("reliable", False) for r in targets)
                coverage[role] = {"gt_tracks": len(targets), "reliably_observed": covered, "missing_or_unreliable": len(targets) - covered}
            summaries[name].append({"video_id": vid, "images": len(video["images"]), "gt_rows": data["num_gt_dets"],
                                    "raw_prediction_rows": len(rows), "canonical_evaluated_prediction_rows": data["num_tracker_dets"],
                                    "coverage": coverage, "purity": observed_purity(purity_frames),
                                    "canonical_tracking": {k: float(np.mean(scores[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")},
                                    "source_npz_bytes": source.stat().st_size, "source_npz_sha256": sha256_file(source)})
            del dataset, rows, frames, purity_frames
        atomic_json(evaluation / "progress.json", {"status": "EVALUATING_SEALED_PREDICTIONS", "completed_videos": position + 1,
                                                   "total_videos": config["videos"], "elapsed_seconds": time.monotonic() - started})
        if (position + 1) % 50 == 0: print(json.dumps({"evaluated_videos": position + 1}), flush=True)
    results = {}
    for name in names:
        combined = metric.combine_sequences(per_sequence[name]); coverage = {}
        for role in ("known", "novel"):
            totals = {k: sum(r["coverage"][role][k] for r in summaries[name]) for k in ("gt_tracks", "reliably_observed", "missing_or_unreliable")}
            totals["coverage"] = totals["reliably_observed"] / totals["gt_tracks"] if totals["gt_tracks"] else None; coverage[role] = totals
        lengths = length_statistics(np.asarray(all_lengths[name], dtype=np.int64))
        lengths.update(counts_come_from_full_frame_stream_not_only_annotated_frames=False, scope="Full-Val annotated-cadence/projection lengths, not dense-frame lifetime")
        results[name] = {"canonical_tracking": {k: float(np.mean(combined[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")},
                         "canonical_raw_combined": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in combined.items()},
                         "coverage": coverage, "purity": aggregate_purity([r["purity"] for r in summaries[name]]),
                         "annotated_projection_lengths": lengths, "per_video": summaries[name],
                         "raw_prediction_rows": sum(r["raw_prediction_rows"] for r in summaries[name]),
                         "canonical_evaluated_prediction_rows": sum(r["canonical_evaluated_prediction_rows"] for r in summaries[name])}
    reference = json.loads((ROOT / "outputs/trackocd_core/audit/frontend.json").read_text())
    historical = reference["pandas_bytetrack_frozen_reference"]["physical_metrics_0_to_1"]
    result = {"schema_version": "trackocd.core.masa_full_val_physical_result.v1", "status": "FULL_VAL_PHYSICAL_AUDIT_COMPLETE_NOT_AUTOMATIC_PRIMARY_PASS",
              "scope": config["scope"], "videos": config["videos"], "images": config["images"], "gt_rows": len(gt["annotations"]),
              "results": results, "config_sha256": digest, "prediction_manifest_sha256": sha256_file(prediction_path),
              "annotation_sha256": config["annotation_sha256"], "roles_sha256": config["roles_sha256"],
              "canonical_adapter_sha256": adapter_hash, "canonical_hota_sha256": hota_hash,
              "provenance": config["provenance"], "cadence_boundary": config["cadence_boundary"],
              "historical_pandas_reference_metrics_not_overwritten": historical,
              "pandas_same_protocol_hota_absolute_difference_from_frozen_export": abs(results["PANDAS_BT_FG0"]["canonical_tracking"]["HOTA"] - historical["HOTA"]),
              "labels_for_model_input_or_tuning": False, "primary_freeze_permitted": False, "ocd_or_m9_metrics": False,
              "resources": {"cpu_workers": 1, "gpu_used": False, "wall_seconds": time.monotonic() - started,
                            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    guard(config); atomic_json(destination, result)
    print(json.dumps({"status": result["status"], "results": {name: {k: results[name][k] for k in ("canonical_tracking", "coverage", "purity")} for name in names}, "resources": result["resources"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
