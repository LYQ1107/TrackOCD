#!/usr/bin/env python3
"""Read-only, per-video redundancy and GT coverage audit of frozen NPZ."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.baselines.pandas_bytetrack.diagnostics import (
    add_counts, atomic_json, box_iou, finish_recall, frame_rows, gt_arrays,
    load_evaluator, read_video, recall_counts, sha256, summary,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    r, out = args.asset_root, args.output_root
    tao, anns, roles = load_evaluator(args.annotation, r / "audit/tao_val_protocol.json")
    ids = sorted(v["id"] for v in tao["videos"])
    calibration = ids[:20]
    confirmation = sorted(sorted(ids[20:], key=lambda i: hashlib.sha256(
        f"pandas-dedup-confirm-v1:{i}".encode()).hexdigest())[:20])
    selection = {"calibration_rule": "first 20 ascending TAO Validation video IDs",
                 "calibration_ids": calibration,
                 "confirmation_rule": "first 20 SHA256(pandas-dedup-confirm-v1:<id>) ranks outside calibration, then ID sort",
                 "confirmation_ids": confirmation,
                 "full_ids": ids,
                 "selection_uses_metrics": False,
                 "variants": {"Dedup-A": {"nms_iou": .5, "max_detections": 50},
                              "Dedup-B": {"nms_iou": .7, "max_detections": 50},
                              "Dedup-C": {"nms_iou": .7, "max_detections": 100}},
                 "full_gate": {"min_candidate_reduction_fraction": .2,
                               "max_absolute_recall_loss": .02,
                               "min_absolute_hota_gain": .005,
                               "min_absolute_deta_gain": .002,
                               "require_evaluator_consistency": True},
                 "selection_rule": "among recall-eligible calibration variants, highest HOTA; ties by listed A/B/C order"}
    out.mkdir(parents=True, exist_ok=True)
    sel_path = out / "selection.json"
    if sel_path.exists() and json.loads(sel_path.read_text()) != selection:
        raise ValueError("refusing to change preregistered selection")
    atomic_json(sel_path, selection)
    counts = {"original": [], **{str(t): [] for t in [.01, .1, .5, .6]},
              "unique_exact_geometry": [], "unique_geometry_nms_099": []}
    overlaps = {str(t): {"pairs": 0, "candidates_with_partner": 0,
                        "cross_prototype_pairs": 0, "equal_foreground_pairs": 0,
                        "cross_prototype_equal_foreground_pairs": 0} for t in [.9, .95, .99]}
    exact = {"duplicate_rows_beyond_first": 0, "pairs": 0,
             "cross_prototype_pairs": 0, "equal_foreground_pairs": 0}
    recall = {"original_pre_floor": {}, **{str(t): {} for t in [.01, .1, .5, .6]}}
    hashes, per_video = [], []
    total_frames = total_rows = 0
    start = time.time()
    for vid in calibration:
        src = r / f"outputs/score_fix_full/anonymous/full/video_{vid:04d}.npz"
        orig = r / f"outputs/anonymous/full/video_{vid:04d}.npz"
        before_hash = sha256(src)
        d, o = read_video(src), read_video(orig)
        if not np.array_equal(d["frame_index"], o["frame_index"]):
            raise ValueError("original/frozen timeline mismatch")
        # Original historical files may have sentinel image IDs. Current cache
        # canonical IDs determine evaluator joins; no GT enters any adapter.
        vrows = 0
        for pos, image_id in enumerate(d["image_id"]):
            a, b, boxes = frame_rows(d, pos)
            _, _, old_boxes = frame_rows(o, pos)
            fg, proto = d["foreground_scores"][a:b], d["prototype_id"][a:b]
            n = len(boxes)
            total_frames += 1; total_rows += n; vrows += n
            counts["original"].append(len(old_boxes))
            for t in [.01, .1, .5, .6]:
                counts[str(t)].append(int((fg >= t).sum()))
            u = len(np.unique(boxes, axis=0))
            counts["unique_exact_geometry"].append(u)
            exact["duplicate_rows_beyond_first"] += n-u
            # Standard TorchVision NMS only estimates near-independent geometry.
            keep = nms(torch.from_numpy(boxes), torch.from_numpy(fg), .99)
            counts["unique_geometry_nms_099"].append(len(keep))
            iou = box_iou(boxes, boxes)
            rr, cc = np.triu_indices(n, 1)
            cross = proto[rr] != proto[cc]
            eq_fg = fg[rr] == fg[cc]
            eq_box = np.all(boxes[rr] == boxes[cc], axis=1)
            exact["pairs"] += int(eq_box.sum())
            exact["cross_prototype_pairs"] += int((eq_box & cross).sum())
            exact["equal_foreground_pairs"] += int((eq_box & eq_fg).sum())
            for t, stat in overlaps.items():
                mask = iou[rr, cc] >= float(t)
                stat["pairs"] += int(mask.sum())
                stat["candidates_with_partner"] += len(np.unique(np.r_[rr[mask], cc[mask]]))
                stat["cross_prototype_pairs"] += int((mask & cross).sum())
                stat["equal_foreground_pairs"] += int((mask & eq_fg).sum())
                stat["cross_prototype_equal_foreground_pairs"] += int((mask & cross & eq_fg).sum())
            if int(image_id) >= 0:
                gt, gtroles = gt_arrays(anns[int(image_id)], roles)
                add_counts(recall["original_pre_floor"], recall_counts(gt, gtroles, old_boxes))
                for t in [.01, .1, .5, .6]:
                    add_counts(recall[str(t)], recall_counts(gt, gtroles, boxes[fg >= t]))
        if sha256(src) != before_hash:
            raise ValueError("frozen input changed")
        hashes.append({"video_id": vid, "path": str(src), "sha256": before_hash,
                       "original_path": str(orig), "original_sha256": sha256(orig)})
        per_video.append({"video_id": vid, "frames": len(d["frame_index"]), "rows": vrows})
        print(json.dumps(per_video[-1]), flush=True)
    for stat in overlaps.values():
        stat["candidate_fraction"] = stat["candidates_with_partner"] / total_rows
        for key in ["cross_prototype_pairs", "equal_foreground_pairs", "cross_prototype_equal_foreground_pairs"]:
            stat[key+"_fraction"] = stat[key] / max(stat["pairs"], 1)
    result = {"status": "PASS", "selection": selection, "frames": total_frames,
              "detections": total_rows, "counts_per_frame": {k: summary(v) for k,v in counts.items()},
              "overlap": overlaps, "exact_geometry": exact,
              "proposal_id_available": False,
              "origin_evidence": "Exact geometry plus equal foreground is an observable proxy; cached original proposal IDs are unavailable. Source duplicates class-agnostic regression across prototype classes and performs class-wise NMS.",
              "proposal_recall_iou05": {k: finish_recall(v) for k,v in recall.items()},
              "source_hashes": hashes, "per_video": per_video,
              "elapsed_seconds": time.time()-start}
    atomic_json(out / "audit.json", result)
    print(json.dumps({"status": "PASS", "frames": total_frames, "rows": total_rows,
                      "exact_duplicate_rows": exact["duplicate_rows_beyond_first"],
                      "overlap": overlaps}), flush=True)


if __name__ == "__main__":
    main()
