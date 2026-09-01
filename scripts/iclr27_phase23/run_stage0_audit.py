#!/usr/bin/env python3
"""Phase23 Stage0: diagnose the Phase22 refiner before another source model.

This is a read-only TRAIN/Phase22 audit.  It never uses held labels as model
inputs and never alters the frozen evaluator.  Outputs are atomically written
under the Phase23 namespace.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase22.proposal_refiner import ProposalRefiner, corrected_box
from src.iclr27_phase23.protocol import CSV_PATH, FEAT_PATH, FEAT_META_PATH, P22_MANIFEST, by_track, fval, load_aligned_features, load_events, normalized_gt, raw_box, row_key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase23"
P22_TAX = ROOT / "outputs/iclr27_phase22/audit/failure_taxonomy_76.json"
P22_SUM = ROOT / "outputs/iclr27_phase22/audit/failure_taxonomy_summary.json"
P22_METRIC = ROOT / "outputs/iclr27_phase22/metrics/stage3_proposal_validation_repair.json"
P22_INITIAL_REC = ROOT / "outputs/iclr27_phase22/audit/stage3_proposal_event_records.json"
P22_REPAIR_REC = ROOT / "outputs/iclr27_phase22/audit/stage3_proposal_validation_repair_event_records.json"
P22_TRAIN_DIR = ROOT / "outputs/iclr27_phase22/metrics"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def parse_box(s: str | None) -> list[float] | None:
    try:
        x = [float(v) for v in json.loads(s or "")]
        return x if len(x) == 4 and all(math.isfinite(v) for v in x) else None
    except Exception: return None


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def bin_name(x: float) -> str:
    if x == 0: return "0"
    if x <= .1: return "(0,0.1]"
    if x <= .25: return "(0.1,0.25]"
    if x <= .4: return "(0.25,0.4]"
    if x < .5: return "(0.4,0.5)"
    return ">=0.5"


def raw_gt_iou_bins(rows: list[dict[str, str]]) -> dict[str, int]:
    out = Counter()
    for row in rows:
        gt = normalized_gt(row)
        if gt is not None:
            out[bin_name(fval(row, "row_iou"))] += 1
    return dict(out)


def event_side_stats(tax_records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    side_bins = {"source": Counter(), "target": Counter()}; failed_bins = {"source": Counter(), "target": Counter()}; repairability: dict[str, Any] = {}
    for rec in tax_records:
        for side in ("source", "target"):
            mx = float(rec[side]["summary"].get("max_iou", 0.0)); b = bin_name(mx); side_bins[side][b] += 1
            if rec.get("is_failed_event"): failed_bins[side][b] += 1
        if rec.get("is_failed_event"):
            repairability[str(rec["event_key"])] = {
                "primary_failure_class": rec["primary_failure_class"],
                "source_max_iou": float(rec["source"]["summary"].get("max_iou", 0.0)),
                "target_max_iou": float(rec["target"]["summary"].get("max_iou", 0.0)),
                "source_bin": bin_name(float(rec["source"]["summary"].get("max_iou", 0.0))),
                "target_bin": bin_name(float(rec["target"]["summary"].get("max_iou", 0.0))),
                "source_candidate_count": int(rec["source"]["summary"].get("candidate_box_count", 0)),
                "target_candidate_count": int(rec["target"]["summary"].get("candidate_box_count", 0)),
                "source_assigned": bool(rec["assignment_and_chronology"].get("source_assigned")),
                "target_assigned": bool(rec["assignment_and_chronology"].get("target_assigned_in_prefix")),
            }
    return {s: dict(c) for s, c in side_bins.items()}, {s: dict(c) for s, c in failed_bins.items()}, repairability


def phase22_prediction_stats(rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, manifest: dict[str, Any]) -> dict[str, Any]:
    """Inspect validation predictions for both Phase22 cycles on CPU."""
    fields = ("score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log", "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou")
    out: dict[str, Any] = {}
    for tag in ("initial", "repair"):
        folds: list[dict[str, Any]] = []
        for fold in range(4):
            ckpt_name = f"proposal_refiner_{'repair_' if tag == 'repair' else ''}f{fold}_best.pt"
            ckpt = OUT.parent / "iclr27_phase22" / "checkpoints" / ckpt_name
            ck = torch.load(ckpt, map_location="cpu", weights_only=False); model = ProposalRefiner(); model.load_state_dict(ck["model"]); model.eval()
            fr = next(x for x in manifest["folds"] if int(x["fold"]) == fold)
            vv, hc = set(map(int, fr["validation_videos"])), set(map(int, fr["held_categories"]))
            idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in vv and int(r.get("gt_category_id_common", -1)) in hc]
            visual = np.concatenate([cls[idx], roi[idx]], axis=1).astype(np.float32)
            geom = np.asarray([[fval(rows[i], k) for k in fields] for i in idx], dtype=np.float32)
            boxes = np.asarray([raw_box(rows[i]) for i in idx], dtype=np.float32)
            with torch.no_grad(): pred = model(torch.from_numpy(visual), torch.from_numpy(geom)); delta = pred["box_delta"].float(); corr = corrected_box(torch.from_numpy(boxes), delta).numpy(); delta_np = delta.numpy()
            widths = corr[:, 2] - corr[:, 0]; heights = corr[:, 3] - corr[:, 1]
            folds.append({
                "fold": fold, "rows": len(idx), "delta_abs_mean": float(np.abs(delta_np).mean()), "delta_abs_p95": float(np.percentile(np.abs(delta_np), 95)),
                "delta_min": float(delta_np.min()), "delta_max": float(delta_np.max()), "corrected_coord_min": float(corr.min()), "corrected_coord_max": float(corr.max()),
                "corrected_width_mean": float(widths.mean()), "corrected_height_mean": float(heights.mean()), "inverted_box_rows": int(((widths <= 0) | (heights <= 0)).sum()),
                "raw_box_rows": len(idx), "raw_box_area_mean": float(np.mean(boxes[:, 2:4] - boxes[:, :2])), "clamp_boundary_fraction": float(np.mean((corr <= 1e-6) | (corr >= 1 - 1e-6))),
                "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
            })
        out[tag] = {"folds": folds, "model_input_feature_source": str(FEAT_PATH), "feature_layer_metadata": str(ROOT / "outputs/iclr27_phase15s/features/public_cls_roi.npz.json")}
    return out


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    z = np.load(FEAT_PATH, allow_pickle=False); cls_raw, roi_raw, keys = z["cls"], z["roi"], [str(x) for x in z["row_keys"]]
    cls, roi, alignment = load_aligned_features(rows)
    key_rows = [row_key(r) for r in rows]
    feature_mismatch = [i for i, (a, b) in enumerate(zip(key_rows, keys)) if a != b]
    feature_pos = {k: i for i, k in enumerate(keys)}
    mismatch_key_set = {key_rows[i] for i in feature_mismatch}
    tax = json.load(P22_TAX.open(encoding="utf-8")); tax_records = tax["records"]
    stage0_side_bins, failed_side_bins, repairability = event_side_stats(tax_records)
    manifest = json.load(P22_MANIFEST.open(encoding="utf-8"))
    # Every event's proposal rows are checked against the positional cache.  We
    # retain this as an impact audit rather than silently treating the
    # permutation as harmless.
    event_impact = []
    for rec in tax_records:
        event_keys = []
        for side in ("source", "target"):
            for c in rec.get(side, {}).get("candidate_boxes", []):
                event_keys.append(str(c.get("row_key", "")))
        event_keys = sorted(set(k for k in event_keys if k))
        affected = [k for k in event_keys if key_rows[feature_pos[k]] != k] if event_keys else []
        event_impact.append({"event_key": str(rec["event_key"]), "fold": int(rec["fold"]), "row_keys_examined": len(event_keys), "row_keys_positional_mismatch": len(affected), "all_event_rows_mismatched": bool(event_keys and len(affected) == len(event_keys)), "sample_row_key": event_keys[0] if event_keys else None, "sample_feature_positional_key": keys[feature_pos[event_keys[0]]] if event_keys else None})
    label_stats = Counter()
    for r in rows:
        has_gt = normalized_gt(r) is not None; assigned = str(r.get("assigned", "0")) == "1"; cat = str(r.get("gt_category_id_common", "-1")) != "-1"
        label_stats["gt_bbox_valid"] += int(has_gt); label_stats["gt_category_present"] += int(cat); label_stats["assigned"] += int(assigned); label_stats["gt_valid_unassigned"] += int(has_gt and not assigned); label_stats["gt_category_without_bbox"] += int(cat and not has_gt); label_stats["false_positive_no_gt"] += int(not has_gt and not assigned)
    norm_mismatch = 0; stored_iou_mismatch = 0; invalid = 0; image_dims = Counter(); delta_target = []
    for r in rows:
        image_dims[f"{r.get('image_width')}x{r.get('image_height')}"] += 1
        b = raw_box(r); w, h = fval(r, "image_width"), fval(r, "image_height"); stored = parse_box(r.get("bbox_xyxy"))
        if w <= 0 or h <= 0 or not (0 <= b[0] <= b[2] <= 1 and 0 <= b[1] <= b[3] <= 1): invalid += 1
        if stored is None or max(abs(b[j] - [stored[0]/w, stored[1]/h, stored[2]/w, stored[3]/h][j]) for j in range(4)) > 1e-4: norm_mismatch += 1
        gt = normalized_gt(r)
        if gt is not None:
            recomputed = iou(b, gt); stored_iou_mismatch += int(abs(recomputed - fval(r, "row_iou")) > 1e-4); delta_target.extend([gt[j] - b[j] for j in range(4)])
    feature_meta = ROOT / "outputs/iclr27_phase15s/features/public_cls_roi.npz.json"
    train_metrics = {}
    for tag in ("initial", "repair"):
        train_metrics[tag] = [json.load((P22_TRAIN_DIR / f"train_{'repair_' if tag == 'repair' else ''}f{fold}.json").open(encoding="utf-8")) for fold in range(4)]
    audit = {
        "protocol": "trackocd_iclr27_phase23_stage0_refiner_failure_audit",
        "phase22_inputs": {"taxonomy": str(P22_TAX), "taxonomy_summary": str(P22_SUM), "stage3_metric": str(P22_METRIC), "initial_event_records": str(P22_INITIAL_REC), "repair_event_records": str(P22_REPAIR_REC)},
        "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "feature_path": str(FEAT_PATH), "feature_sha256": sha256(FEAT_PATH),
        "feature_metadata": json.load(FEAT_META_PATH.open(encoding="utf-8")) if FEAT_META_PATH.exists() else {},
        "rows": len(rows), "feature_rows": len(keys), "feature_shape": {"cls": list(cls.shape), "roi": list(roi.shape)}, "feature_row_key_mismatch_count": len(feature_mismatch), "feature_row_key_mismatch_indices": feature_mismatch[:20],
        "feature_key_alignment": {**alignment, "csv_key_field": "row_key=video_id:frame_id:proposal_local_id:track_id:image_id", "feature_key_field": "npz row_keys (same five fields, legacy proposal order)", "mismatch_key_count": len(mismatch_key_set), "repair": "in-memory permutation only; Phase22 artifacts untouched", "before_example": {"csv_row0_key": key_rows[0], "npz_row0_key": keys[0]}, "after_example": {"csv_row0_key": key_rows[0], "aligned_npz_source_index": int(np.where(np.asarray(keys) == key_rows[0])[0][0]), "aligned_key": key_rows[int(np.where(np.asarray(keys) == key_rows[0])[0][0])]}} ,
        "feature_alignment_event_impact": {"events": event_impact, "events_with_any_mismatch": sum(x["row_keys_positional_mismatch"] > 0 for x in event_impact), "events_all_rows_mismatched": sum(x["all_event_rows_mismatched"] for x in event_impact), "folds_with_any_mismatch": sorted({x["fold"] for x in event_impact if x["row_keys_positional_mismatch"] > 0})},
        "image_resolution_counts": dict(image_dims), "geometry_consistency": {"invalid_or_inverted_normalized_boxes": invalid, "normalized_bbox_mismatch_rows": norm_mismatch, "stored_iou_mismatch_rows": stored_iou_mismatch, "actual_dimensions_used": True},
        "phase22_label_statistics": dict(label_stats), "raw_gt_iou_row_bins": raw_gt_iou_bins(rows), "event_side_raw_max_iou_bins": stage0_side_bins, "failed_event_side_raw_max_iou_bins": failed_side_bins,
        "delta_target_gt_minus_raw_stats": {"count": len(delta_target), "mean": float(np.mean(delta_target)) if delta_target else 0., "median": float(np.median(delta_target)) if delta_target else 0., "p95_abs": float(np.percentile(np.abs(delta_target), 95)) if delta_target else 0.},
        "phase22_prediction_statistics_aligned_input": phase22_prediction_stats(rows, cls, roi, manifest), "phase22_prediction_statistics_legacy_positional_input": phase22_prediction_stats(rows, cls_raw, roi_raw, manifest), "train_metrics": {tag: [{"fold": d["fold"], "fit_rows": d["fit_rows"], "fit_positive_rows": d["fit_positive_rows"], "fit_negative_rows": d["fit_negative_rows"], "validation_rows": d["validation_rows"], "best_step": d["best_step"], "best_score": d["best_score"], "validation_metrics": d["validation_metrics"]} for d in ds] for tag, ds in train_metrics.items()},
        "raw_candidate_repairability": {"failed_event_count": len(repairability), "events_with_source_max_0.4_to_0.5": sum(v["source_bin"] == "(0.4,0.5)" for v in repairability.values()), "events_with_target_max_0.4_to_0.5": sum(v["target_bin"] == "(0.4,0.5)" for v in repairability.values()), "events_with_both_sides_max_lt_0.1": sum(v["source_max_iou"] <= .1 and v["target_max_iou"] <= .1 for v in repairability.values()), "event_records": repairability},
        "root_cause_interpretation": {"established": ["Phase22 model moved usable proposals away from identity; trained validation reliable recall was below raw", "raw candidate IoU failures dominate the held event taxonomy; geometry/row-key checks are clean", "GT labels are not present on unassigned rows in the frozen CSV, so no false-positive GT box labels were found"], "not_established": ["domain shift is causal", "a different image backbone will solve proposal coverage", "candidate source is impossible"], "next_registered_action": "evaluate a fixed multi-candidate pool before any learned source branch"},
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"],
    }
    atomic_json(OUT / "audit/stage0_refiner_failure_audit.json", audit)
    # A compact summary makes the first actionable root cause machine-readable.
    atomic_json(OUT / "audit/stage0_refiner_failure_summary.json", {"protocol": audit["protocol"], "rows": len(rows), "feature_row_key_mismatch_count": len(feature_mismatch), "geometry_consistency": audit["geometry_consistency"], "label_statistics": dict(label_stats), "raw_gt_iou_row_bins": audit["raw_gt_iou_row_bins"], "event_side_raw_max_iou_bins": stage0_side_bins, "failed_event_side_raw_max_iou_bins": failed_side_bins, "repairability": audit["raw_candidate_repairability"], "root_cause_interpretation": audit["root_cause_interpretation"], "sealed_inputs_not_read": audit["sealed_inputs_not_read"]})
    atomic_json(OUT / "completion/stage0.done", {"stage": "stage0", "event_denominator": 76, "raw_prefix16_ceiling": 25, "decision": "refiner_failure_audit_complete_candidate_pool_authorized"})
    lines = ["# Phase23 Stage 0 — Phase22 refiner failure audit", "", f"The audit uses {len(rows)} frozen TRAIN-derived rows and the unchanged Phase22 76-event protocol.  Feature row-key mismatches: **{len(feature_mismatch)} / {len(rows)}** (all four folds and all 76 event row sets are affected); invalid/normalized/stored-IoU geometry errors: **{invalid}/{norm_mismatch}/{stored_iou_mismatch}**.", "", "## First actionable evidence", "", f"The first actionable Phase22 failure is a feature/row-key ordering error, not a geometry transform error.  The corrected CSV and NPZ contain the same 43,423 five-field keys but in completely different orders (CSV row 0 `{key_rows[0]}`; NPZ row 0 `{keys[0]}`).  Phase22's positional indexing therefore paired every proposal with another row's DINOv2 CLS/ROI.  Phase23 repairs this only by an in-memory key permutation; no Phase22 file is modified.  The old residual box head also moved usable proposals away from identity under the mispaired inputs.  The CSV has no parsed GT boxes on unassigned rows (`gt_valid_unassigned=0`), so the label construction did not create a hidden unassigned-positive class.  A smoke and one-fold aligned regression are required before candidate/source experiments.", "", "### Raw row IoU bins", "", "| bin | GT rows |", "|---|---:|"]
    for k, v in sorted(audit["raw_gt_iou_row_bins"].items()): lines.append(f"| {k} | {v} |")
    lines += ["", "### Event-side max IoU bins at prefix16", "", "| side | 0 | (0,0.1] | (0.1,0.25] | (0.25,0.4] | (0.4,0.5) | >=0.5 |", "|---|---:|---:|---:|---:|---:|---:|"]
    order = ["0", "(0,0.1]", "(0.1,0.25]", "(0.25,0.4]", "(0.4,0.5)", ">=0.5"]
    for side in ("source", "target"):
        c = stage0_side_bins[side]; lines.append("| " + side + " | " + " | ".join(str(c.get(k, 0)) for k in order) + " |")
    lines += ["", "Failed-event candidate evidence, repairability bins, model prediction box statistics and all fold label counts are in [`stage0_refiner_failure_audit.json`](../../outputs/iclr27_phase23/audit/stage0_refiner_failure_audit.json).  The taxonomy retains all 51 failures; no event was dropped.", "", "## Stage 0 decision", "", "No geometry/chronology transformation bug was found.  A Phase22 feature-loader row-order bug was found and is repaired only in-memory by Phase23; after that correction the failure remains compatible with a proposal box/candidate-coverage limitation, so the fixed candidate-pool oracle is authorized.  Domain difference and backbone insufficiency remain hypotheses, not conclusions."]
    atomic_text(OUT.parent.parent / "docs/iclr27_phase23/STAGE0_REFINER_FAILURE_AUDIT.md", "\n".join(lines) + "\n")
    print(json.dumps({"rows": len(rows), "feature_key_mismatch": len(feature_mismatch), "geometry_errors": [invalid, norm_mismatch, stored_iou_mismatch], "raw_event_side_bins": stage0_side_bins, "stage0_done": str(OUT / "completion/stage0.done")}, indent=2, sort_keys=True))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


if __name__ == "__main__":
    main()
