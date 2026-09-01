"""Evaluate ORBIT-D1 on matched predicted tracks (matched-only diagnostic)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.track_matching import load_gt_tracks, load_pred_tracks, match_tracks
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.outputs import emit
from src.orbit.evaluate import load_model, run_stream, build_known
from src.orbit.protocol import load_frame_features, load_train_labels, load_mean_features
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    model, ck = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device="cuda")
    train_feats = load_frame_features("train_known_mean")
    train_labels = load_train_labels()
    protos, radii = build_known(model, train_feats, train_labels, set(train_labels.values()), "cuda")
    gt_vid, gt_anns = load_gt_tracks()
    pred_vid, pred_anns, pred_rows = load_pred_tracks()
    # rebuild GT frame anns from per-track records (upstream by_video_anns
    # builder only keeps the last frame per track id)
    gt_anns = defaultdict(dict)
    for vid, tracks in gt_vid.items():
        for tid, rec in tracks.items():
            gt_anns[vid][tid] = {
                fid: box for fid, box in zip(rec["frame_ids"], rec["boxes_xyxy"])
            }
    matches = match_tracks(gt_anns, pred_anns, threshold=0.5)
    gt_private = {}
    with open(ROOT / "data/tao_ow_ocd_v1/private/val_gt_track_labels.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                gt_private[r["sample_id"]] = r
    pred_to_gt = {}
    for vid, gt_tid, pred_tid, iou in matches:
        gt_sid = f"{vid}_{gt_tid}"
        if gt_sid in gt_private:
            pred_sid = f"P{vid}_{pred_tid}"
            pred_to_gt[pred_sid] = gt_private[gt_sid]
    pred_feats = load_frame_features("pred_tracks_mean")
    rows = [r for r in pred_rows if r["sample_id"] in pred_feats and r["sample_id"] in pred_to_gt]
    gt_rows = [
        {"sample_id": r["sample_id"],
         "ground_truth_category_id": pred_to_gt[r["sample_id"]]["ground_truth_category_id"],
         "protocol_role": "supported_known" if pred_to_gt[r["sample_id"]]["is_known"] else "novel"}
        for r in rows
    ]
    preds, _ = run_stream(model, rows, pred_feats, protos, radii, "cuda", mode="joint")
    ev = TrackOCDEvaluator(gt_rows)
    res = ev.evaluate(preds)
    # TrackOCD-Ref on the same matched predicted tracks
    tr_labels = load_train_labels()
    tr_mean = load_mean_features("train_known_mean")
    ref_protos = {}
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in tr_labels.items():
        if sid in tr_mean:
            sums[c] += tr_mean[sid]
            counts[c] += 1
    for c in counts:
        ref_protos[c] = sums[c] / counts[c]
        ref_protos[c] = ref_protos[c] / (np.linalg.norm(ref_protos[c]) + 1e-12)
    pred_mean = load_mean_features("pred_tracks_mean")
    mem = B2Memory(ref_protos, threshold=0.45)
    ref_preds = []
    for i, r in enumerate(rows):
        vid, kind = mem.predict_one(pred_mean[r["sample_id"]], r["sample_id"], i)
        ref_preds.append(emit(r["sample_id"], i, kind,
                              vid if kind == "known" else None,
                              vid if kind == "novel" else None))
    ref_res = ev.evaluate(ref_preds)
    out = ROOT / "outputs/orbit/predicted_track_orbit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"orbit": res, "reference": ref_res,
                               "prediction_log": preds}, indent=1, default=str))
    print(json.dumps({k: res[k] for k in
          ["num_samples", "overall_known_acc", "route_aware_novel_acc",
           "conditional_novel_acc", "novel_routing_recall", "novel_only_nmi",
           "novel_only_ari", "all_track_acc", "novel_count_abs_error"]}, indent=1))
    print("reference", {k: ref_res[k] for k in
          ["num_samples", "overall_known_acc", "route_aware_novel_acc",
           "conditional_novel_acc", "novel_routing_recall", "novel_only_nmi",
           "novel_only_ari", "all_track_acc", "novel_count_abs_error"]})


if __name__ == "__main__":
    main()
