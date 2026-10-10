"""Count-weighted full-Val composition of the unchanged per-video scorer."""
from collections import Counter
import numpy as np
from src.trackocd_core.physical_qualification import aggregate_purity
from src.trackocd_core.physical_diagnostic import observed_purity
from src.trackocd_v2.io import atomic_json, sha256_file
from scripts.trackocd_core.audit_frontend import length_statistics
from scripts.trackocd_core.audit_frontend_coverage import match_video
from scripts.trackocd_core.evaluate_masa_physical_qualification import read_projection


def score_video(name, path, video, subset, known, novel, gt_folder, scratch, trackeval, guard):
    """Exact legacy full-audit ordering/formulas, including GT-target tie order."""
    vid = video["video_id"]; grouped, by_image = {}, {}
    for ann in subset["annotations"]:
        by_image.setdefault(int(ann["image_id"]), []).append(ann)
        category = int(ann["category_id"])
        if category in known | novel:
            key = f"{vid}_{ann['track_id']}"
            target = grouped.setdefault(key, {"key": key, "category": category, "role": "known" if category in known else "novel", "boxes": {}})
            if target["category"] != category: raise ValueError("Inconsistent GT category")
            x, y, w, h = map(float, ann["bbox"])
            target["boxes"][int(ann["image_id"])] = [x, y, x + w, y + h]
    frames = read_projection(path, video["images"]); rows, purity_frames = [], []
    _, counts = np.unique(np.concatenate([frame[0] for frame in frames.values()]), return_counts=True)
    for image in video["images"]:
        iid = image["image_id"]; ids, boxes, scores = frames[iid]
        for local, box, score in zip(ids, boxes, scores):
            x, y, x2, y2 = map(float, box)
            rows.append({"image_id": iid, "video_id": vid, "track_id": int(local), "category_id": 1,
                         "bbox": [x, y, x2 - x, y2 - y], "score": float(score)})
        annotations = by_image.get(iid, []); gt_boxes = []
        for ann in annotations:
            x, y, w, h = map(float, ann["bbox"]); gt_boxes.append([x, y, x + w, y + h])
        purity_frames.append({"pred_ids": ids, "pred_boxes": boxes, "gt_boxes": gt_boxes,
                              "gt_ids": [int(a["track_id"]) for a in annotations], "gt_categories": [int(a["category_id"]) for a in annotations]})
    data_folder = scratch / name / name / "data"; data_folder.mkdir(parents=True)
    atomic_json(data_folder / "pred.json", rows)
    options = trackeval.datasets.TAO_OW.get_default_dataset_config()
    options.update(GT_FOLDER=str(gt_folder), TRACKERS_FOLDER=str(scratch / name), TRACKERS_TO_EVAL=[name],
                   TRACKER_SUB_FOLDER="data", SPLIT_TO_EVAL="val", SUBSET="all", MAX_DETECTIONS=300, PRINT_CONFIG=False)
    dataset, metric = trackeval.datasets.TAO_OW(options), trackeval.metrics.HOTA()
    if len(dataset.seq_list) != 1: raise ValueError("Single whole-video canonical evaluation expected")
    raw = dataset.get_raw_seq_data(name, dataset.seq_list[0]); data = dataset.get_preprocessed_seq_data(raw, "object")
    scores = metric.eval_sequence(data)
    matches = match_video(list(grouped.values()), {i: (f[0], f[1]) for i, f in frames.items()})["matches"]
    coverage = {}
    for role in ("known", "novel"):
        targets = [r for r in grouped.values() if r["role"] == role]
        covered = sum(matches.get(r["key"], {}).get("reliable", False) for r in targets)
        coverage[role] = {"gt_tracks": len(targets), "reliably_observed": covered, "missing_or_unreliable": len(targets) - covered}
    summary = {"video_id": vid, "images": len(video["images"]), "gt_rows": data["num_gt_dets"],
               "raw_prediction_rows": len(rows), "canonical_evaluated_prediction_rows": data["num_tracker_dets"],
               "coverage": coverage, "purity": observed_purity(purity_frames),
               "canonical_tracking": {k: float(np.mean(scores[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")},
               "source_npz_bytes": path.stat().st_size, "source_npz_sha256": sha256_file(path)}
    guard()
    return scores, summary, Counter(counts.tolist())


def combine_route(metric, sequences, summaries, histogram):
    if len(sequences) != len(summaries): raise ValueError("No omission in complete canonical combination")
    combined = metric.combine_sequences(sequences)
    coverage = {}
    for role in ("known", "novel"):
        totals = {k: sum(row["coverage"][role][k] for row in summaries)
                  for k in ("gt_tracks", "reliably_observed", "missing_or_unreliable")}
        totals["coverage"] = totals["reliably_observed"] / totals["gt_tracks"] if totals["gt_tracks"] else None
        coverage[role] = totals
    lengths = np.repeat(np.asarray(sorted(histogram), dtype=np.int64), [histogram[k] for k in sorted(histogram)])
    length = length_statistics(lengths)
    length.update(counts_come_from_full_frame_stream_not_only_annotated_frames=False,
                  scope="Full-Val annotated-cadence/projection lengths, not dense-frame lifetime")
    return {"canonical_tracking": {k: float(np.mean(combined[k])) for k in ("HOTA", "AssA", "DetA", "DetRe")},
            "canonical_raw_combined": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in combined.items()},
            "coverage": coverage, "purity": aggregate_purity([r["purity"] for r in summaries]),
            "annotated_projection_lengths": length, "per_video": summaries,
            "raw_prediction_rows": sum(r["raw_prediction_rows"] for r in summaries),
            "canonical_evaluated_prediction_rows": sum(r["canonical_evaluated_prediction_rows"] for r in summaries)}


def compare_reference(actual, expected, tolerance):
    difference = {k: abs(actual["canonical_tracking"][k] - expected["canonical_tracking"][k])
                  for k in ("HOTA", "AssA", "DetA", "DetRe")}
    exact = {k: actual[k] == expected[k] for k in ("coverage", "purity", "annotated_projection_lengths", "raw_prediction_rows", "canonical_evaluated_prediction_rows")}
    old = {v["video_id"]: v for v in expected["per_video"]}
    per_video_max = 0.; per_video_exact = set(old) == {v["video_id"] for v in actual["per_video"]}
    for row in actual["per_video"]:
        if row["video_id"] not in old:
            per_video_exact = False; continue
        reference = old[row["video_id"]]
        per_video_max = max(per_video_max, max(abs(row["canonical_tracking"][k] - reference["canonical_tracking"][k]) for k in difference))
        per_video_exact = per_video_exact and all(row[k] == reference[k] for k in ("images", "gt_rows", "raw_prediction_rows", "canonical_evaluated_prediction_rows", "coverage", "purity", "source_npz_bytes", "source_npz_sha256"))
    return {"pass": all(d <= tolerance for d in difference.values()) and all(exact.values()) and per_video_exact and per_video_max <= tolerance,
            "absolute_metric_differences": difference, "exact_full_fields": exact,
            "per_video_max_absolute_metric_difference": per_video_max, "per_video_nonmetric_fields_exact": per_video_exact}
