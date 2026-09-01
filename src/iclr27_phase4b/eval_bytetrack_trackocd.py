"""ByteTrack + TrackOCD-Ref (3 seeds, fixed order) and ORBIT-D1 diagnostic."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.evaluation.track_matching import load_gt_tracks, temporal_iou
from src.dual_branch.memory.b2_adapter import B2Memory
from src.dual_branch.models.outputs import emit
from src.orbit.evaluate import load_model, build_known
from src.orbit_bc.batch_orbit import run_batch_bc
from src.orbit.protocol import load_frame_features, load_train_labels, load_mean_features, load_gt
from src.trackocd_v1.evaluation.trackocd_evaluator import TrackOCDEvaluator


def load_stream_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def load_feature_mean(cache_dir):
    out = {}
    for p in Path(cache_dir).glob("*.json"):
        r = json.load(open(p))
        out[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    return out


def matched_gt_rows(rows, gt_rows):
    return {r["sample_id"]: g for r, g in zip(rows, gt_rows)}


def match_tracks_fast(gt_anns, pred_anns, threshold=0.5):
    matches = []
    for vid in sorted(set(gt_anns) & set(pred_anns)):
        gts = gt_anns[vid]
        preds = pred_anns[vid]
        frame_to_pred = defaultdict(list)
        for p, anns in preds.items():
            for f in anns:
                frame_to_pred[f].append(p)
        for gid, ganns in gts.items():
            best_p, best_iou = None, threshold
            cands = set()
            for f in ganns:
                cands.update(frame_to_pred.get(f, []))
            for p in cands:
                iou = temporal_iou(ganns, preds[p])
                if iou >= best_iou:
                    best_iou, best_p = iou, p
            if best_p is not None:
                matches.append((vid, gid, best_p, float(best_iou)))
    return matches


def main():
    stream = load_stream_rows(ROOT / "outputs/iclr27_phase4b/bytetrack/pred_track_stream_bytetrack.jsonl")
    feats_mean = load_feature_mean(ROOT / "outputs/iclr27_phase4b/bytetrack_features")
    feats_frame = load_frame_features_from_cache(ROOT / "outputs/iclr27_phase4b/bytetrack_features")
    # only tracks with features
    rows = [r for r in stream if r["sample_id"] in feats_mean]
    # match to GT
    gt_vid, _ = load_gt_tracks()
    gt_anns = defaultdict(dict)
    for vid, tracks in gt_vid.items():
        for tid, rec in tracks.items():
            gt_anns[vid][tid] = {fid: box for fid, box in zip(rec["frame_ids"], rec["boxes_xyxy"])}
    pred_anns = defaultdict(dict)
    for r in rows:
        pred_anns[r["video_id"]][r["track_id"]] = {
            fid: box for fid, box in zip(r["frame_ids"], r["boxes_xyxy"])
        }
    matches = match_tracks_fast(gt_anns, pred_anns, threshold=0.5)
    gt_private = {json.loads(l)["sample_id"]: json.loads(l)
                  for l in open(ROOT / "data/tao_ow_ocd_v1/private/val_gt_track_labels.jsonl")}
    pred_to_gt = {}
    for vid, gt_tid, pred_tid, iou in matches:
        gt_sid = f"{vid}_{gt_tid}"
        if gt_sid in gt_private:
            pred_to_gt[f"B{vid}_{pred_tid}"] = gt_private[gt_sid]
    matched_rows = [r for r in rows if r["sample_id"] in pred_to_gt]
    print("bytetrack tracks", len(rows), "matched", len(matched_rows), flush=True)

    all_gt = load_gt("pure")
    matched_gt = [
        {"sample_id": r["sample_id"],
         "ground_truth_category_id": pred_to_gt[r["sample_id"]]["ground_truth_category_id"],
         "protocol_role": "supported_known" if pred_to_gt[r["sample_id"]]["is_known"] else "novel"}
        for r in matched_rows
    ]
    # coverage-aware: all pure GT rows, unmatched have no prediction -> unresolved
    coverage_gt = all_gt

    # TrackOCD-Ref
    tr_labels = load_train_labels()
    tr_mean = load_mean_features("train_known_mean")
    protos = {}
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in tr_labels.items():
        if sid in tr_mean:
            sums[c] += tr_mean[sid]
            counts[c] += 1
    for c in counts:
        protos[c] = sums[c] / counts[c]
        protos[c] = protos[c] / (np.linalg.norm(protos[c]) + 1e-12)
    mem = B2Memory(protos, threshold=0.45)
    ref_preds = []
    for i, r in enumerate(rows):
        vid, kind = mem.predict_one(feats_mean[r["sample_id"]], r["sample_id"], i)
        ref_preds.append(emit(r["sample_id"], i, kind,
                              vid if kind == "known" else None,
                              vid if kind == "novel" else None))
    ref_matched = TrackOCDEvaluator(matched_gt).evaluate(ref_preds)
    ref_cov = TrackOCDEvaluator(coverage_gt).evaluate(ref_preds)

    # ORBIT-D1 diagnostic (seed1027, Pure Full)
    device = "cuda"
    model, _ = load_model(ROOT / "runs/orbit/model_D1_b128_g0.3/model.pth", device=device)
    train_frame = load_frame_features("train_known_mean")
    protos_orbit, radii = build_known(model, train_frame, tr_labels, set(tr_labels.values()), device)
    orbit_preds, _ = run_batch_bc(model, rows, feats_frame, protos_orbit, radii, device,
                                  birth_threshold=0.0)
    orbit_matched = TrackOCDEvaluator(matched_gt).evaluate(orbit_preds)
    orbit_cov = TrackOCDEvaluator(coverage_gt).evaluate(orbit_preds)

    def row(method, scope, r):
        return {
            "method": method, "scope": scope,
            "all_acc": r["all_track_acc"], "known_acc": r["overall_known_acc"],
            "rn_acc": r["route_aware_novel_acc"], "cond_novel_acc": r["conditional_novel_acc"],
            "routing_recall": r["novel_routing_recall"], "nmi": r["novel_only_nmi"],
            "ari": r["novel_only_ari"], "count_error": r["novel_count_abs_error"],
        }
    out_dir = ROOT / "outputs/iclr27_phase4b/end_to_end"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = [
        row("TrackOCD-Ref", "matched", ref_matched),
        row("TrackOCD-Ref", "coverage", ref_cov),
        row("ORBIT-D1", "matched", orbit_matched),
        row("ORBIT-D1", "coverage", orbit_cov),
    ]
    with open(out_dir / "bytetrack_reference_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    # per-seed CSV: ByteTrack order fixed, all three seeds identical
    with open(out_dir / "bytetrack_reference_per_seed.csv", "w", newline="") as f:
        fieldnames = list(rows_out[0].keys()) + ["seed"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for seed in ["1027", "1028", "1029"]:
            rr = dict(rows_out[0]); rr["seed"] = seed
            w.writerow(rr)
    with open(out_dir / "bytetrack_orbit_d1_seed1027.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerow(rows_out[2])
    print(json.dumps(rows_out, indent=1))


def load_frame_features_from_cache(cache_dir):
    out = {}
    for p in Path(cache_dir).glob("*.json"):
        r = json.load(open(p))
        arr = np.asarray(r["frame_embeddings"], dtype=np.float32)
        arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        out[r["sample_id"]] = arr
    return out


if __name__ == "__main__":
    main()
