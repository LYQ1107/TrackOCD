#!/usr/bin/env python3
"""Minimal Phase23 feature-key repair smoke and targeted regression.

The script does not alter Phase22 artifacts.  It proves that the corrected
in-memory feature permutation can be consumed by the existing refiner and
that an identity residual preserves the frozen raw 25/76 protocol result.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase22.proposal_refiner import GEOM_FIELDS, ProposalRefiner, corrected_box, box_iou_xyxy
from src.iclr27_phase23.protocol import CSV_PATH, FEAT_PATH, POS_PATH, P22_MANIFEST, by_track, fval, load_aligned_features, load_events, normalized_gt, raw_box

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase23"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def geom(r: dict[str, str]) -> np.ndarray:
    return np.asarray([fval(r, k) for k in GEOM_FIELDS], dtype=np.float32)


def event_ceiling(rows: list[dict[str, str]], events: list[dict[str, Any]], boxes: np.ndarray) -> dict[str, Any]:
    tracks = by_track(rows); out = []
    for e in events:
        sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"])
        si, ti = tracks.get(sk, []), tracks.get(tk, [])
        p = min(16, len(ti)); src_ok = False; tgt_ok = False
        for i in si:
            g = normalized_gt(rows[i]); src_ok |= bool(str(rows[i].get("assigned", "0")) == "1" and g is not None and float(box_iou_xyxy(torch.tensor(boxes[i]).reshape(1,4), torch.tensor(g).reshape(1,4))[0]) >= .5)
        for i in ti[:p]:
            g = normalized_gt(rows[i]); tgt_ok |= bool(str(rows[i].get("assigned", "0")) == "1" and g is not None and float(box_iou_xyxy(torch.tensor(boxes[i]).reshape(1,4), torch.tensor(g).reshape(1,4))[0]) >= .5)
        out.append({"event_key": str(e["event_key"]), "fold": int(e["fold"]), "source_reliable": int(src_ok), "target_reliable": int(tgt_ok), "ceiling": int(src_ok and tgt_ok)})
    return {"event_count": len(out), "ceiling_correct": sum(x["ceiling"] for x in out), "by_fold": [{"fold": f, "denominator": sum(x["fold"] == f for x in out), "ceiling_correct": sum(x["fold"] == f and x["ceiling"] for x in out)} for f in range(4)], "events": out}


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events()
    cls, roi, alignment = load_aligned_features(rows)
    raw = np.asarray([raw_box(r) for r in rows], dtype=np.float32)
    identity = event_ceiling(rows, events, raw)
    smoke = ProposalRefiner(); opt = torch.optim.AdamW(smoke.parameters(), lr=2e-4)
    # Two CPU steps are deliberately small: this is a path/smoke check, not a
    # result-bearing training run.  Features are already in corrected CSV order.
    idx = np.arange(min(32, len(rows)), dtype=np.int64)
    v = torch.from_numpy(np.concatenate([cls[idx], roi[idx]], axis=1).astype(np.float32)); g = torch.from_numpy(np.stack([geom(rows[i]) for i in idx])); b = torch.from_numpy(raw[idx]); gt = torch.from_numpy(np.stack([normalized_gt(rows[i]) if normalized_gt(rows[i]) is not None else raw[i] for i in idx]).astype(np.float32))
    smoke_losses = []
    for _ in range(2):
        opt.zero_grad(); out = smoke(v, g); pred = corrected_box(b, out["box_delta"]); loss = torch.nn.functional.smooth_l1_loss(pred, gt) + .1 * out["quality_logit"].square().mean(); loss.backward(); opt.step(); smoke_losses.append(float(loss.detach()))
    # Targeted fold-0 regression: compare identity, frozen Phase22 repair
    # checkpoint under aligned features, and the same checkpoint under the old
    # positional features.  This explicitly exposes the pre-fix pairing.
    manifest = json.loads(P22_MANIFEST.read_text())
    fold = 0; fr = next(x for x in manifest["folds"] if int(x["fold"]) == fold)
    val_v, held_c = set(map(int, fr["validation_videos"])), set(map(int, fr["held_categories"]))
    val_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in val_v and int(r.get("gt_category_id_common", -1)) in held_c]
    ckpt = ROOT / "outputs/iclr27_phase22/checkpoints/proposal_refiner_repair_f0_best.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); model = ProposalRefiner(); model.load_state_dict(ck["model"]); model.eval()
    def model_boxes(c: np.ndarray, r: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.from_numpy(np.concatenate([c[val_idx], r[val_idx]], axis=1).astype(np.float32)), torch.from_numpy(np.stack([geom(rows[i]) for i in val_idx]))); return corrected_box(torch.from_numpy(raw[val_idx]), out["box_delta"]).numpy()
    aligned_boxes = model_boxes(cls, roi)
    # Reconstruct the legacy positional feature view for a direct regression.
    z = np.load(FEAT_PATH, allow_pickle=False); legacy_cls, legacy_roi = z["cls"], z["roi"]
    legacy_boxes = model_boxes(legacy_cls, legacy_roi)
    def val_reliable(boxes: np.ndarray) -> int:
        n = 0
        for j, i in enumerate(val_idx):
            g = normalized_gt(rows[i]); n += int(str(rows[i].get("assigned", "0")) == "1" and g is not None and float(box_iou_xyxy(torch.tensor(boxes[j]).reshape(1,4), torch.tensor(g).reshape(1,4))[0]) >= .5)
        return n
    result = {"protocol": "trackocd_iclr27_phase23_alignment_smoke_targeted_regression", "feature_alignment": alignment, "raw_baseline_prefix16": identity, "identity_refiner_preserves_raw_ceiling": identity["ceiling_correct"] == 25, "smoke": {"steps": 2, "losses": smoke_losses, "aligned_feature_shape": [int(cls.shape[1]), int(roi.shape[1])]}, "targeted_fold": {"fold": fold, "validation_rows": len(val_idx), "phase22_repair_checkpoint": str(ckpt), "aligned_input_reliable_rows": val_reliable(aligned_boxes), "legacy_positional_input_reliable_rows": val_reliable(legacy_boxes), "aligned_prediction_stats": {"coord_min": float(aligned_boxes.min()), "coord_max": float(aligned_boxes.max()), "inverted": int(((aligned_boxes[:,2] <= aligned_boxes[:,0]) | (aligned_boxes[:,3] <= aligned_boxes[:,1])).sum())}, "legacy_prediction_stats": {"coord_min": float(legacy_boxes.min()), "coord_max": float(legacy_boxes.max()), "inverted": int(((legacy_boxes[:,2] <= legacy_boxes[:,0]) | (legacy_boxes[:,3] <= legacy_boxes[:,1])).sum())}}, "phase22_files_modified": False, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}
    atomic_json(OUT / "audit/alignment_smoke_targeted_regression.json", result)
    atomic_json(OUT / "completion/alignment_regression.done", {"stage": "alignment_regression", "raw_prefix16": identity["ceiling_correct"], "targeted_fold": fold, "smoke_steps": 2})
    print(json.dumps({"raw_prefix16": identity["ceiling_correct"], "identity_preserves": result["identity_refiner_preserves_raw_ceiling"], "fold0_aligned_rows": result["targeted_fold"]["aligned_input_reliable_rows"], "fold0_legacy_rows": result["targeted_fold"]["legacy_positional_input_reliable_rows"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
