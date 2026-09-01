"""Full-population frozen-feature paired/temporal baseline after training."""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase17r.evaluation.evaluate_candidate import l2, retrieval, track_items

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    rows = list(csv.DictReader((ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv").open()))
    n = len(rows); assigned = np.asarray([int(r["assigned"]) == 1 for r in rows]); roles = np.asarray([0 if r["gt_role_common"] == "fp" else 1 if r["gt_role_common"] == "supported_known" else 2 for r in rows]); cats = np.asarray([int(r["gt_category_id_common"]) for r in rows])
    data = {"rows": rows, "assigned": assigned, "roles": roles, "cats": cats}
    d3 = np.load(ROOT / "outputs/iclr27_phase17r/features/full_public_dinov3.npz", allow_pickle=False)
    d2z = np.load(ROOT / "data/iclr27_phase17r/sources/public_dinov2_cls_roi.npz", allow_pickle=False)
    lookup = {str(k): i for i, k in enumerate(d2z["row_keys"].astype(str))}; order = np.asarray([lookup[r["row_key"]] for r in rows])
    representations = {
        "DINOv2_ROI_518": d2z["roi"][order].astype(np.float32),
        "DINOv2_CLS_518": d2z["cls"][order].astype(np.float32),
        "DINOv3_PROPOSAL_RAW": d3["features"][:, 0].astype(np.float32),
        "DINOv3_PROPOSAL_CTX10": d3["features"][:, 1].astype(np.float32),
        "DINOv3_PROPOSAL_CTX25": d3["features"][:, 2].astype(np.float32),
        "DINOv3_PROPOSAL_CAUSAL_SMOOTHED": d3["features"][:, 3].astype(np.float32),
        "DINOv3_GT_TIGHT_TEACHER_ONLY": d3["features"][:, 4].astype(np.float32),
        "DINOv3_GT_CTX10_TEACHER_ONLY": d3["features"][:, 5].astype(np.float32)
    }
    all_idx = np.arange(n); metrics = {}
    for name, feat in representations.items():
        metrics[name] = {"supported_known": retrieval(track_items(all_idx, data, feat, role_value=1)),
                         "novel": retrieval(track_items(all_idx, data, feat, role_value=2))}
    raw, smooth, gt = representations["DINOv3_PROPOSAL_RAW"], representations["DINOv3_PROPOSAL_CAUSAL_SMOOTHED"], representations["DINOv3_GT_TIGHT_TEACHER_ONLY"]
    valid = assigned; iou = np.asarray([float(r["row_iou"]) for r in rows])
    strata = {}
    for name, mask in {"zero_iou": valid & (iou == 0), "low_iou": valid & (iou > 0) & (iou < .5), "high_iou": valid & (iou >= .5)}.items():
        strata[name] = {"rows": int(mask.sum()),
                        "gt_to_raw_cosine": float(np.mean(np.sum(l2(gt[mask]) * l2(raw[mask]), axis=1))) if mask.any() else None,
                        "raw_to_temporal_cosine": float(np.mean(np.sum(l2(raw[mask]) * l2(smooth[mask]), axis=1))) if mask.any() else None,
                        "raw_temporal_different_rows": int((np.linalg.norm(raw[mask] - smooth[mask], axis=1) > 1e-4).sum())}
    effect = {}
    for role_name in ("supported_known", "novel"):
        role_value = 1 if role_name == "supported_known" else 2
        raw_m = metrics["DINOv3_PROPOSAL_RAW"][role_name]; gt_m = metrics["DINOv3_GT_TIGHT_TEACHER_ONLY"][role_name]
        effect[role_name] = {"gt_r_at_1": gt_m.get("r_at_1"), "raw_r_at_1": raw_m.get("r_at_1"),
                             "gt_to_raw_r_at_1_drop": (gt_m.get("r_at_1") - raw_m.get("r_at_1")) if gt_m.get("r_at_1") is not None and raw_m.get("r_at_1") is not None else None,
                             "population": "all assigned public tracks across frozen roles; diagnostic only"}
    value = {"protocol": "trackocd_iclr27_phase17r_full_paired_baseline", "rows": n, "assigned_rows": int(assigned.sum()),
             "metrics": metrics, "quality_strata": strata, "paired_effect": effect,
             "full_population_not_phase17_sample": True, "gt_teacher_deployed": False,
             "future_frames_used": False, "q1_labels_used": False}
    out = ROOT / "outputs/iclr27_phase17r/eval/full_paired_baseline.json"; tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); os.replace(tmp, out)
    print(json.dumps({"rows": n, "assigned": int(assigned.sum()), "effect": effect, "strata": strata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
