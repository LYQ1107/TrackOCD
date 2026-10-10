"""Evaluator-only common scorer for already sealed small physical streams.

Same canonical/coverage/pollution formula as the independently reproduced
legacy diagnostic. No training, model state, sampling or returned GT mapping.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from src.trackocd_v2.io import atomic_json, sha256_file
from src.trackocd_core.physical_diagnostic import observed_purity
from scripts.trackocd_core.audit_frontend import length_statistics
from scripts.trackocd_core.audit_frontend_coverage import match_video
from scripts.trackocd_core.evaluate_masa_physical_qualification import read_projection


def score_route(name, files, plan, subset, known, novel, gt_folder, scratch, trackeval, guard):
    by_image = defaultdict(list)
    for ann in subset["annotations"]: by_image[int(ann["image_id"])].append(ann)
    rows, lengths, targets, matched, purity_frames, sources = [], [], [], {}, [], []
    for video in plan["videos"]:
        vid = video["video_id"]; path = files[vid]; frames = read_projection(path, video["images"])
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
        targets.extend(grouped.values()); matched.update(match_video(list(grouped.values()), {i: (r[0], r[1]) for i, r in frames.items()})["matches"])
        guard()
    tracker_root = scratch / ("canonical_" + name); data_folder = tracker_root / name / "data"; data_folder.mkdir(parents=True)
    atomic_json(data_folder / "pred.json", rows)
    options = trackeval.datasets.TAO_OW.get_default_dataset_config()
    options.update(GT_FOLDER=str(gt_folder), TRACKERS_FOLDER=str(tracker_root), TRACKERS_TO_EVAL=[name],
                   TRACKER_SUB_FOLDER="data", SPLIT_TO_EVAL="val", SUBSET="all", MAX_DETECTIONS=300, PRINT_CONFIG=False)
    dataset, metric = trackeval.datasets.TAO_OW(options), trackeval.metrics.HOTA()
    per_sequence, per_video, raw_count, processed_count = {}, [], 0, 0
    for seq in dataset.seq_list:
        raw = dataset.get_raw_seq_data(name, seq); data = dataset.get_preprocessed_seq_data(raw, "object")
        score = metric.eval_sequence(data); per_sequence[seq] = score
        raw_count += sum(len(v) for v in raw["tracker_ids"]); processed_count += data["num_tracker_dets"]
        per_video.append({"video_id": int(dataset.seq_name_to_seq_id[seq]), "gt_rows": data["num_gt_dets"],
                          "evaluated_pred_rows": data["num_tracker_dets"], **{k: float(np.mean(score[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")}})
        guard()
    combined = metric.combine_sequences(per_sequence); coverage = {}
    for role in ("known", "novel"):
        group = [t for t in targets if t["role"] == role]; covered = sum(matched.get(t["key"], {}).get("reliable", False) for t in group)
        coverage[role] = {"gt_clip_tracks": len(group), "reliably_observed": covered, "missing_or_unreliable": len(group) - covered,
                          "coverage": covered / len(group) if group else None}
    length = length_statistics(np.asarray(lengths, dtype=np.int64))
    length.update(scope="Only fixed selected annotated-cadence clip projection, not full-video lifetime", counts_come_from_full_frame_stream_not_only_annotated_frames=False)
    return {"canonical_tracking": {k: float(np.mean(combined[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")},
            "canonical_raw_combined": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in combined.items()},
            "coverage": coverage, "purity": observed_purity(purity_frames), "annotated_projection_lengths": length,
            "per_video": sorted(per_video, key=lambda r: r["video_id"]), "raw_prediction_rows": raw_count,
            "canonical_evaluated_prediction_rows": processed_count, "canonical_preprocessing_removed_rows": raw_count - processed_count,
            "source_npz": sources}
