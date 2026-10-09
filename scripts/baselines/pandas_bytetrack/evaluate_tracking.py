#!/usr/bin/env python3
"""Evaluator-only cached-candidate recall and canonical TAO-OW export.

GT only enters this command after tracking. A subset uses the exact original
videos/images/annotations/tracks and unchanged category/flag metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.baselines.pandas_bytetrack.diagnostics import (
    add_counts, atomic_json, box_iou, finish_recall, frame_rows, gt_arrays,
    load_evaluator, read_video, recall_counts, sha256, summary,
)


def run_trackeval(args):
    command = [sys.executable,str(Path(__file__).parent / "frozen_runtime/run_tao_ow_trackeval.py"),
               "--trackeval-root",str(args.trackeval_root),"--gt-folder",str(args.output_root / "gt"),
               "--trackers-folder",str(args.output_root),"--tracker","bytetrack",
               "--output",str(args.output_root / "trackeval_summary.json")]
    with (args.output_root / "trackeval.log").open("w") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((args.output_root / "trackeval_summary.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--annotation", type=Path, required=True)
    p.add_argument("--tracks-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--subset", choices=["calibration", "confirmation", "full"], required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--trackeval-root", type=Path, required=True)
    args = p.parse_args()
    selection = json.loads(args.selection.read_text())
    ids = selection[args.subset+"_ids"]
    pred_path = args.output_root / "bytetrack/data/pred.json"
    if pred_path.exists():
        marker = args.output_root / "diagnostics.json"
        if not marker.exists():
            raise RuntimeError("prediction lacks completed diagnostics; audit incomplete output before resume")
        done = json.loads(marker.read_text())
        if (done["video_ids"] != ids or done["variant"] != args.variant
            or done["annotation_sha256"] != sha256(args.annotation)
            or done["prediction_sha256"] != sha256(pred_path)
            or done["gt_export_sha256"] != sha256(args.output_root / "gt/validation.json")):
            raise ValueError("evaluation resume contract mismatch")
        for source in done["source_hashes"]:
            name = f"video_{source['video_id']:04d}.npz"
            if (source["detections_sha256"] != sha256(args.asset_root / "outputs/score_fix_full/anonymous/full" / name)
                or source["tracks_sha256"] != sha256(args.tracks_root / name)):
                raise ValueError("evaluation source changed")
        summary_path = args.output_root / "trackeval_summary.json"
        result = json.loads(summary_path.read_text()) if summary_path.exists() else run_trackeval(args)
        if result["status"] != "PASS":
            raise ValueError("existing TrackEval did not pass")
        print(json.dumps({"status":"PASS","resumed":True,"metrics":result["metrics"]}),flush=True)
        return
    tao, anns, roles = load_evaluator(args.annotation, args.asset_root / "audit/tao_val_protocol.json")
    expected_images = {int(i["id"]): int(i["video_id"]) for i in tao["images"] if i["video_id"] in ids}
    recall = {k:{} for k in ["before", "after", "association_before", "association_after", "birth_before", "birth_after", "track_output"]}
    counts_before, counts_after, lengths = [], [], []
    losses = Counter(); loss_examples = []
    seen_images = set(); total_frames = total_rows = 0
    voting = defaultdict(Counter)
    semantic_diag = Counter()
    mapping = json.loads((args.asset_root / "outputs/detection/pandas_hungarian_mapping.json").read_text())["prototype_to_tao_category_id"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_root / "bytetrack/data"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "pred.json"
    tmp = pred_path.with_suffix(".json.tmp")
    first = True
    sources = []
    with tmp.open("w") as f:
        f.write("[")
        for vid in ids:
            src = args.asset_root / f"outputs/score_fix_full/anonymous/full/video_{vid:04d}.npz"
            tp = args.tracks_root / src.name
            d, t = read_video(src), read_video(tp)
            if not np.array_equal(d["frame_index"],t["frame_index"]) or not np.array_equal(d["image_id"],t["image_id"]):
                raise ValueError(f"source/track timeline mismatch for {vid}")
            if "source_frame_offsets" in t and not np.array_equal(d["frame_offsets"],t["source_frame_offsets"]):
                raise ValueError("source offsets changed")
            _, video_lengths = np.unique(t["track_id"], return_counts=True)
            lengths.extend(video_lengths.tolist())
            total_frames += len(d["frame_index"])
            for pos,image_id in enumerate(d["image_id"]):
                a,b,boxes = frame_rows(d,pos)
                fg = d["foreground_scores"][a:b]
                if "candidate_original_detection_index" in t:
                    ca,cb = t["candidate_frame_offsets"][pos:pos+2]
                    indices = t["candidate_original_detection_index"][int(ca):int(cb)]
                    if np.any((indices < a) | (indices >= b)):
                        raise ValueError("candidate index outside its source frame")
                    keep = indices-a
                else:
                    keep = np.arange(b-a)
                counts_before.append(b-a); counts_after.append(len(keep))
                ta,tb,track_boxes = frame_rows(t,pos)
                if len(np.unique(t["track_id"][ta:tb])) != tb-ta:
                    raise ValueError("duplicate physical IDs within a frame")
                if int(image_id) < 0:
                    continue
                if expected_images.get(int(image_id)) != vid or int(image_id) in seen_images:
                    raise ValueError("canonical image/video join failed")
                seen_images.add(int(image_id))
                gt,gtroles = gt_arrays(anns[int(image_id)],roles)
                after = boxes[keep]
                before_iou,after_iou = box_iou(gt,boxes),box_iou(gt,after)
                hit_before = before_iou.max(axis=1)>=.5 if len(boxes) else np.zeros(len(gt),bool)
                hit_after = after_iou.max(axis=1)>=.5 if len(after) else np.zeros(len(gt),bool)
                lost = hit_before & ~hit_after
                losses["covered_gt_lost"] += int(lost.sum())
                gt_overlap = box_iou(gt,gt)
                np.fill_diagonal(gt_overlap,0)
                for j in np.where(lost)[0]:
                    role = gtroles[j]; losses["lost_"+role] += 1
                    nearby = bool(np.any(gt_overlap[j] >= .5))
                    losses["lost_with_other_gt_iou05"] += int(nearby)
                    if len(loss_examples) < 20:
                        loss_examples.append({"image_id":int(image_id),"annotation_id":anns[int(image_id)][j]["id"],
                                              "role":str(role),"nearby_distinct_gt_iou05":nearby,
                                              "best_iou_before":float(before_iou[j].max()),
                                              "best_iou_after":float(after_iou[j].max()) if len(after) else 0.})
                for k,candidates in [("before",boxes),("after",after),
                    ("association_before",boxes[fg>.1]),("association_after",after[fg[keep]>.1]),
                    ("birth_before",boxes[fg>=.6]),("birth_after",after[fg[keep]>=.6]),
                    ("track_output",track_boxes)]:
                    add_counts(recall[k],recall_counts(gt,gtroles,candidates))
                if args.variant == "Original-FG-0":
                    for j in np.where(hit_before & (gtroles == "novel"))[0]:
                        candidates = np.where(before_iou[j]>=.5)[0]
                        idx = candidates[np.argmax(d["scores"][a+candidates])]
                        proto = int(d["prototype_id"][a+idx]); cat = int(anns[int(image_id)][j]["category_id"])
                        voting[proto][cat] += 1
                        semantic_diag["covered_novel_gt"] += 1
                        semantic_diag["assigned_novel_prototype"] += int(proto >= 80)
                        mapped = (mapping[proto] if 0 <= proto < len(mapping) else None) if isinstance(mapping,list) else mapping.get(str(proto))
                        semantic_diag["mapped_correct_covered_novel"] += int(mapped is not None and int(mapped)==cat)
                for row in range(ta,tb):
                    x1,y1,x2,y2 = map(float,t["boxes"][row])
                    record = {"image_id":int(image_id),"video_id":vid,"category_id":1,
                              "track_id":int(t["track_id"][row]),
                              "bbox":[x1,y1,max(0.,x2-x1),max(0.,y2-y1)],"score":float(t["score"][row])}
                    if not first: f.write(",")
                    json.dump(record,f,separators=(",",":"),allow_nan=False)
                    first=False; total_rows+=1
            sources.append({"video_id":vid,"detections_sha256":sha256(src),"tracks_sha256":sha256(tp)})
            print(json.dumps({"video_id":vid,"exported_rows_so_far":total_rows}),flush=True)
        f.write("]\n")
    if seen_images != set(expected_images):
        raise ValueError(f"annotated frame coverage mismatch: {len(seen_images)} != {len(expected_images)}")
    os.replace(tmp,pred_path)
    gt_dir = args.output_root / "gt"
    gt_dir.mkdir(parents=True,exist_ok=True)
    gt_path = gt_dir / "validation.json"
    if args.subset == "full":
        gt_path.symlink_to(args.annotation.resolve())
    else:
        sub = dict(tao)
        for key in ["videos","images","annotations","tracks"]:
            sub[key] = [v for v in tao[key] if (v["id"] if key == "videos" else v["video_id"]) in ids]
        atomic_json(gt_path,sub)
    purity = sum(max(v.values()) for v in voting.values()) / max(semantic_diag["covered_novel_gt"],1)
    diag = {"status":"PASS","variant":args.variant,"subset":args.subset,"video_ids":ids,
            "videos":len(ids),"frames":total_frames,"annotated_frames":len(seen_images),
            "candidate_counts_before":summary(counts_before),"candidate_counts_after":summary(counts_after),
            "retained_fraction":sum(counts_after)/max(sum(counts_before),1),
            "proposal_recall_iou05":{k:finish_recall(v) for k,v in recall.items()},
            "physical_tracks":len(lengths),"track_lengths":summary(lengths),
            "track_rows_full_stream":int(sum(lengths)),"track_rows_exported":total_rows,
            "lost_covered_gt":dict(losses),"loss_examples":loss_examples,
            "evaluator_only_semantic_diagnostic":{
                **dict(semantic_diag),"cluster_purity_on_covered_novel":purity if voting else None,
                "mapped_accuracy_on_covered_novel":semantic_diag["mapped_correct_covered_novel"]/max(semantic_diag["covered_novel_gt"],1) if voting else None,
                "selection":"highest semantic score among IoU>=0.5 candidates per GT; evaluator-only, optimistic mapping fit on TAO Val",
                "anonymous_novel_clusters_observed":len(voting)},
            "source_hashes":sources,"prediction_sha256":sha256(pred_path),
            "annotation_sha256":sha256(args.annotation),"gt_export_sha256":sha256(gt_path),
            "unmatched_policy":"spatial one-to-one diagnostic; may include unannotated real objects"}
    atomic_json(args.output_root / "diagnostics.json",diag)
    # Use the captured canonical wrapper without changing evaluator semantics.
    result = run_trackeval(args)
    print(json.dumps({"status":"PASS","variant":args.variant,"subset":args.subset,
                      "metrics":result["metrics"]}),flush=True)


if __name__ == "__main__":
    main()
