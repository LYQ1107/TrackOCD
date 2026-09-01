"""Paired crop metrics and preregistered proposal-sensitivity decision."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
VIEWS = ["GT_TIGHT", "GT_CTX10", "PROPOSAL_RAW", "PROPOSAL_CTX10", "PROPOSAL_TEMPORAL", "JITTER_MILD", "JITTER_MEDIUM", "JITTER_SEVERE", "MULTISCALE_ROI_0", "MULTISCALE_ROI_10", "MULTISCALE_ROI_25"]


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)


def _retrieval(feat: np.ndarray, cat: np.ndarray, vid: np.ndarray) -> dict[str, Any]:
    n = len(feat); ranks = []; aps = []; ys = []; ss = []
    # One BLAS matrix product avoids the accidental O(n^2*d) Python loop that
    # made the first diagnostic attempt unnecessarily long.
    sim_all = np.asarray(feat, dtype=np.float32) @ np.asarray(feat, dtype=np.float32).T
    for i in range(n):
        valid = np.where(vid != vid[i])[0]
        if not len(valid): continue
        sims = sim_all[i, valid]; order = np.argsort(-sims); labels = (cat[valid] == cat[i]).astype(np.int32)
        # If a category has no cross-video partner, it is not an opportunity
        # and is excluded from retrieval denominators (reported separately).
        if not labels.any(): continue
        ranks.append((int(labels[order[:1]].any()), int(labels[order[:5]].any())))
        aps.append(float(average_precision_score(labels, sims)))
        ys.extend(labels.tolist()); ss.extend(sims.tolist())
    if not ranks: return {"queries_with_cross_video_positive": 0}
    out = {"queries_with_cross_video_positive": len(ranks), "r_at_1": float(np.mean([x[0] for x in ranks])), "r_at_5": float(np.mean([x[1] for x in ranks])), "mAP": float(np.mean(aps)), "same_pairs": int(sum(ys)), "different_pairs": int(len(ys) - sum(ys))}
    if len(set(ys)) > 1:
        out["roc_auc"] = float(roc_auc_score(ys, ss)); out["pr_auc"] = float(average_precision_score(ys, ss))
    return out


def _bootstrap_effect(rows: list[dict[str, Any]], a: np.ndarray, b: np.ndarray, n: int = 1000) -> dict[str, float]:
    by: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows): by[int(r["video_id"])].append(i)
    vids = sorted(by); rng = np.random.default_rng(1717); draws = []
    for _ in range(n):
        chosen = rng.choice(vids, size=len(vids), replace=True)
        idx = [j for v in chosen for j in by[int(v)]]
        draws.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    return {"mean": float(np.mean(draws)), "low95": float(np.quantile(draws, .025)), "high95": float(np.quantile(draws, .975)), "videos": len(vids), "resamples": n}


def _same_occurrence(z: np.ndarray) -> dict[str, float]:
    gt = z[:, 0]
    return {v: float(np.mean(cos(gt, z[:, j]))) for j, v in enumerate(VIEWS)}


def _metrics_for(z: np.ndarray, cats: np.ndarray, vids: np.ndarray) -> dict[str, Any]:
    out = {"same_occurrence_cosine_to_gt": _same_occurrence(z), "retrieval": {}}
    for j, v in enumerate(VIEWS): out["retrieval"][v] = _retrieval(z[:, j], cats, vids)
    # Quality-stratified same-occurrence and retrieval are reported by caller.
    return out


def _open_set(z: np.ndarray, roles: np.ndarray, cats: np.ndarray) -> dict[str, Any]:
    known = roles == "supported_known"; novel = roles == "novel"; gt = z[:, 0]
    # Similarity to the leave-one-out known gallery is an open-set score.
    gallery = np.where(known)[0]; scores = []; labels = []
    for i in np.where(known | novel)[0]:
        g = gallery[gallery != i]
        if not len(g): continue
        scores.append(float(np.max(z[i, 2] @ z[g, 2].T))); labels.append(int(known[i]))
    out = {"known_rows": int(known.sum()), "novel_rows": int(novel.sum()), "score_definition": "max proposal-raw cosine to leave-one-out known gallery"}
    if len(set(labels)) > 1:
        out["known_vs_novel_roc_auc"] = float(roc_auc_score(labels, scores)); out["known_vs_novel_pr_auc"] = float(average_precision_score(labels, scores))
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    z = np.load(args.public, allow_pickle=False); d2, d3 = z["dinov2"], z["dinov3"]
    cats, vids, roles, iou = z["category_id"].astype(int), z["video_id"].astype(int), z["gt_role"].astype(str), z["row_iou"].astype(float)
    rows = [{"video_id": int(v), "category_id": int(c)} for v, c in zip(vids, cats)]
    dev = np.load(args.devplus, allow_pickle=False); dev_c, dev_v, dev_r = dev["category_id"].astype(int), dev["video_id"].astype(int), dev["gt_role"].astype(str)
    metrics = {"DINOv2_ROI_paired224": _metrics_for(d2, cats, vids), "DINOv3_dense": _metrics_for(d3, cats, vids),
               "DINOv2_ROI_paired224_open_set": _open_set(d2, roles, cats), "DINOv3_dense_open_set": _open_set(d3, roles, cats)}
    quality = {}
    for name, feat in [("DINOv2_ROI_paired224", d2), ("DINOv3_dense", d3)]:
        quality[name] = {}
        for lo, hi in zip([0, .25, .5, .75], [.25, .5, .75, 1.01]):
            mask = (iou >= lo) & (iou < hi)
            quality[name][f"row_iou_{lo:g}_{min(hi,1):g}"] = {"rows": int(mask.sum()), "same_occurrence_gt_to_raw": float(np.mean(cos(feat[mask, 0], feat[mask, 2]))) if mask.any() else None,
                                                                  "raw_to_temporal": float(np.mean(cos(feat[mask, 2], feat[mask, 4]))) if mask.any() else None}
    # The preregistered G1 effect is GT-to-raw drop in known macro cosine and
    # cross-video R@1.  Macro is category-balanced over available public rows.
    known = roles == "supported_known"; effect = {}
    for name, feat in [("DINOv2_ROI_paired224", d2), ("DINOv3_dense", d3)]:
        cats_known = sorted(set(cats[known])); gt_by, raw_by = [], []
        for c in cats_known:
            m = known & (cats == c)
            gt_by.append(float(np.mean(cos(feat[m, 0], feat[m, 0])))); raw_by.append(float(np.mean(cos(feat[m, 0], feat[m, 2]))))
        # For GT-to-raw cosine, identical GT-to-GT is exactly 1; this is the
        # paired quality drop used by the registered decision.
        gt_raw = 1.0 - float(np.mean(raw_by)) if raw_by else None
        rgt = _retrieval(feat[:, 0], cats, vids); rraw = _retrieval(feat[:, 2], cats, vids)
        rdrop = (rgt.get("r_at_1", 0.0) - rraw.get("r_at_1", 0.0)) if rgt.get("r_at_1") is not None else None
        known_rows = [{"video_id": int(vids[i])} for i in np.where(known)[0]]
        # Bootstrap the per-row GT-to-raw quality drop; category/video gates
        # are reported explicitly even when the effect is not significant.
        vals_a = np.ones(int(known.sum())); vals_b = cos(feat[known, 0], feat[known, 2])
        effect[name] = {"known_category_macro_gt_to_raw_drop": gt_raw, "cross_video_r1_gt": rgt.get("r_at_1"), "cross_video_r1_raw": rraw.get("r_at_1"), "cross_video_r1_drop": rdrop,
                        "known_categories": len(cats_known), "known_videos": len(set(vids[known])), "video_bootstrap_gt_to_raw_drop": _bootstrap_effect(known_rows, vals_a, vals_b)}
    # DEV+ is diagnostic only.  A 20-row public-safe crop subset is enough to
    # establish direction; no DEV+ label enters training or selection.
    dev_metrics = {}
    for name, feat in [("DINOv2_ROI_paired224", dev["dinov2"]), ("DINOv3_dense", dev["dinov3"])]:
        valid = np.isfinite(feat).all(axis=(1, 2)); f = feat[valid]
        dev_metrics[name] = {"rows": int(valid.sum()), "gt_to_raw_cosine": float(np.mean(cos(f[:, 0], f[:, 2]))) if len(f) else None,
                             "raw_to_temporal_cosine": float(np.mean(cos(f[:, 2], f[:, 4]))) if len(f) else None,
                             "diagnostic_only_devplus_gt": True}
    # G1 requires a positive absolute GT-to-raw *retrieval* drop or >=.10
    # known macro drop, video-bootstrap CI excluding zero, >=3 categories and
    # >=2 videos, plus same-direction DEV+ diagnostic.  The small diagnostic
    # is deliberately not promoted to a DEV+ model-selection result.
    g1 = {}
    for name, e in effect.items():
        ci = e["video_bootstrap_gt_to_raw_drop"]; devdir = dev_metrics[name]["gt_to_raw_cosine"] is not None and dev_metrics[name]["gt_to_raw_cosine"] < 1.0
        g1[name] = {"proposal_sensitivity_gate": bool((e["known_category_macro_gt_to_raw_drop"] is not None and e["known_category_macro_gt_to_raw_drop"] >= .10 or (e["cross_video_r1_drop"] is not None and e["cross_video_r1_drop"] >= .10)) and ci["low95"] > 0 and e["known_categories"] >= 3 and e["known_videos"] >= 2 and devdir),
                    "devplus_same_direction": devdir, "required_abs_drop": .10, "bootstrap_ci": ci}
    result = {"protocol": "trackocd_iclr27_phase17_paired_crop_diagnostic", "public_rows": len(cats), "devplus_rows": len(dev_c), "views": VIEWS,
              "paired_metrics": metrics, "quality_strata": quality, "effect": effect, "devplus_diagnostic": dev_metrics, "G1_gate": g1,
              "dino2_resolution_note": "paired DINOv2 uses 224px for tractability; historical 518px proposal cache remains separate",
              "gt_crops_public_diagnostic_only": True, "diagnostic_only_devplus_gt": True, "future_frames_used": False,
              "physical_id_as_feature": False, "q1_label_used": False}
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = args.out.with_suffix(args.out.suffix + ".tmp"); tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, args.out); print(json.dumps(result, indent=2, sort_keys=True)); return result


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--public", type=Path, default=ROOT / "outputs/iclr27_phase17/paired/public_paired_features.npz"); ap.add_argument("--devplus", type=Path, default=ROOT / "outputs/iclr27_phase17/paired/devplus_paired_features.npz"); ap.add_argument("--out", type=Path, default=ROOT / "outputs/iclr27_phase17/eval/paired_crop_diagnostic.json"); args = ap.parse_args(); run(args)


if __name__ == "__main__": main()
