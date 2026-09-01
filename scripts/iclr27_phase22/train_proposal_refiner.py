#!/usr/bin/env python3
"""Train one independent Phase22 class-agnostic proposal refiner.

The script is intentionally self-contained so each fold can run on one GPU
under a bounded supervisor.  It uses only TRAIN rows selected by the
Phase22 video/category-disjoint manifest.  Checkpoints and markers are
atomic/resumable; no event evaluator or public/Q1 file is opened here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.iclr27_phase22.proposal_refiner import GEOM_FIELDS, ProposalRefiner, box_iou_xyxy, corrected_box, state_dict_metadata

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
FEAT_PATH = ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz"
MANIFEST = ROOT / "outputs/iclr27_phase22/manifests/fold_manifest.json"
OUT = ROOT / "outputs/iclr27_phase22"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        torch.save(value, tmp)
        with open(tmp, "rb") as f: os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default)); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default


def parse_box(value: str | None) -> list[float] | None:
    try:
        x = [float(v) for v in json.loads(value or "")]
        return x if len(x) == 4 and all(math.isfinite(v) for v in x) else None
    except Exception: return None


def normalized_gt(row: dict[str, str]) -> np.ndarray | None:
    b = parse_box(row.get("gt_bbox_xyxy")); w = fval(row, "image_width"); h = fval(row, "image_height")
    if b is None or w <= 0 or h <= 0: return None
    return np.asarray([b[0]/w, b[1]/h, b[2]/w, b[3]/h], dtype=np.float32)


def row_key(row: dict[str, str]) -> str:
    return str(row.get("row_key", ""))


def make_geom(row: dict[str, str]) -> np.ndarray:
    return np.asarray([fval(row, k) for k in GEOM_FIELDS], dtype=np.float32)


def reliable(row: dict[str, str]) -> bool:
    return str(row.get("assigned", "0")) == "1" and fval(row, "row_iou") >= .5


def load_arrays() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, dict[str, int]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f: rows = list(csv.DictReader(f))
    z = np.load(FEAT_PATH, allow_pickle=False)
    cls, roi, keys = z["cls"], z["roi"], [str(x) for x in z["row_keys"]]
    if len(rows) != len(keys): raise RuntimeError(f"feature/row count mismatch {len(rows)} != {len(keys)}")
    key_to_idx = {k: i for i, k in enumerate(keys)}
    if len(key_to_idx) != len(keys): raise RuntimeError("duplicate feature row keys")
    missing = [row_key(r) for r in rows if row_key(r) not in key_to_idx]
    if missing: raise RuntimeError(f"missing feature keys: {len(missing)}")
    return rows, cls, roi, key_to_idx


def build_split(rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, key_to_idx: dict[str, int], row_indices: list[int]) -> dict[str, torch.Tensor]:
    visual = np.concatenate([cls[row_indices], roi[row_indices]], axis=1).astype(np.float32, copy=False)
    geom = np.stack([make_geom(rows[i]) for i in row_indices]).astype(np.float32)
    box = np.stack([[fval(rows[i], k) for k in ("box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm")] for i in row_indices]).astype(np.float32)
    gt = np.zeros((len(row_indices), 4), dtype=np.float32); gt_valid = np.zeros(len(row_indices), dtype=np.float32)
    q = np.zeros(len(row_indices), dtype=np.float32); assigned = np.zeros(len(row_indices), dtype=np.float32)
    for j, i in enumerate(row_indices):
        g = normalized_gt(rows[i])
        if g is not None: gt[j] = g; gt_valid[j] = 1.0
        q[j] = float(reliable(rows[i])); assigned[j] = float(str(rows[i].get("assigned", "0")) == "1")
    return {"visual": torch.from_numpy(visual), "geom": torch.from_numpy(geom), "box": torch.from_numpy(box), "gt": torch.from_numpy(gt), "gt_valid": torch.from_numpy(gt_valid), "quality": torch.from_numpy(q), "assigned": torch.from_numpy(assigned)}


def evaluate(model: ProposalRefiner, data: dict[str, torch.Tensor], device: torch.device, amp_dtype: torch.dtype | None) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        v, g = data["visual"].to(device), data["geom"].to(device)
        if amp_dtype is None:
            out = model(v, g)
        else:
            with torch.autocast(device_type="cuda", dtype=amp_dtype): out = model(v, g)
        corrected = corrected_box(data["box"].to(device), out["box_delta"])
        iou = box_iou_xyxy(corrected.float(), data["gt"].to(device).float()).detach().cpu().numpy()
        prob = torch.sigmoid(out["quality_logit"].float()).detach().cpu().numpy()
    valid = data["gt_valid"].numpy() > .5; assigned = data["assigned"].numpy() > .5; y = data["quality"].numpy().astype(np.int64)
    rel = valid & assigned
    try: ap = float(average_precision_score(y, prob))
    except ValueError: ap = None
    try: auc = float(roc_auc_score(y, prob))
    except ValueError: auc = None
    return {"rows": int(len(iou)), "gt_rows": int(valid.sum()), "raw_reliable_rows": int((data["quality"].numpy() > .5).sum()), "refined_reliable_rows": int((iou >= .5).sum()), "assigned_refined_reliable_rows": int((iou >= .5).astype(np.int64)[assigned].sum()), "reliable_recall_iou50": float(((iou >= .5) & rel).sum() / max(rel.sum(), 1)), "quality_ap": ap, "quality_auc": auc, "iou_mean_gt_rows": float(iou[valid].mean()) if valid.any() else 0.0, "iou_median_gt_rows": float(np.median(iou[valid])) if valid.any() else 0.0, "corrected_iou_values": iou.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--batch-size", type=int, default=256); ap.add_argument("--checkpoint-every", type=int, default=500); ap.add_argument("--seed", type=int, default=20260828); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="", help="independent repair tag for artifacts"); args = ap.parse_args()
    if args.fold not in range(4): raise ValueError(args.fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    seed = int(args.seed) + int(args.fold); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed_all(seed)
    rows, cls, roi, key_to_idx = load_arrays(); manifest = json.loads(MANIFEST.read_text()); fr = next(x for x in manifest["folds"] if int(x["fold"]) == args.fold)
    fit_v, val_v = set(map(int, fr["fit_videos"])), set(map(int, fr["validation_videos"])); fit_c, held_c = set(map(int, fr["fit_categories"])), set(map(int, fr["held_categories"]))
    fit_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in fit_v and (int(r.get("gt_category_id_common", -1)) in fit_c or int(r.get("gt_category_id_common", -1)) < 0)]
    val_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in val_v and (int(r.get("gt_category_id_common", -1)) in held_c)]
    if not fit_idx or not val_idx: raise RuntimeError(f"empty split fold={args.fold} fit={len(fit_idx)} val={len(val_idx)}")
    train = build_split(rows, cls, roi, key_to_idx, fit_idx); val = build_split(rows, cls, roi, key_to_idx, val_idx)
    pos = np.flatnonzero(train["gt_valid"].numpy() > .5); neg = np.flatnonzero(train["gt_valid"].numpy() < .5)
    if len(pos) < 2 or len(neg) < 2: raise RuntimeError(f"insufficient balanced fit positives/negatives {len(pos)}/{len(neg)}")
    smoke_steps = 2 if args.smoke else int(args.steps); prefix = (str(args.tag).strip() + "_") if str(args.tag).strip() else ""; run_name = f"{prefix}smoke_f{args.fold}" if args.smoke else f"{prefix}f{args.fold}"
    marker = OUT / "completion" / f"{run_name}.launched"; done = OUT / "completion" / f"{run_name}.done"; log_path = OUT / "logs" / f"train_{run_name}.jsonl"; ckpt_dir = OUT / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    if done.exists() and not args.resume: print(json.dumps({"fold": args.fold, "status": "already_done", "marker": str(done)})); return
    if marker.exists() and not args.resume: raise RuntimeError(f"{marker} exists; refusing blind relaunch")
    marker.write_text(json.dumps({"fold": args.fold, "pid": os.getpid(), "started": time.time(), "device": str(device)}) + "\n", encoding="utf-8")
    model = ProposalRefiner(); model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    start = 0; best_score = -1.0; best_step = 0; history: list[dict[str, Any]] = []
    latest = ckpt_dir / f"proposal_refiner_{run_name}_latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False); model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"]); start = int(ck.get("global_step", 0)); best_score = float(ck.get("best_score", -1.0)); best_step = int(ck.get("best_step", 0));
    amp_dtype: torch.dtype | None = torch.bfloat16 if device.type == "cuda" else None; amp_mode = "bf16" if amp_dtype is not None else "fp32"
    pos_rng = np.random.default_rng(seed + 11); neg_rng = np.random.default_rng(seed + 17); t0 = time.time(); log_path.parent.mkdir(parents=True, exist_ok=True)
    # Validation is performed at each registered checkpoint cadence; no held
    # event result is read by this training process.
    for step in range(start + 1, smoke_steps + 1):
        model.train(); half = max(1, int(args.batch_size) // 2); pi = pos_rng.choice(pos, size=half, replace=len(pos) < half); ni = neg_rng.choice(neg, size=half, replace=len(neg) < half); bi = np.concatenate([pi, ni]); np.random.default_rng(seed + step).shuffle(bi); idx = torch.from_numpy(bi.astype(np.int64))
        v, g, b, gt = train["visual"][idx].to(device), train["geom"][idx].to(device), train["box"][idx].to(device), train["gt"][idx].to(device); valid = train["gt_valid"][idx].to(device) > .5; q = train["quality"][idx].to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp_dtype is None:
            out = model(v, g); pred = corrected_box(b, out["box_delta"]); loss_box = torch.nn.functional.smooth_l1_loss(pred[valid], gt[valid]) if valid.any() else pred.sum() * 0.; loss_quality = torch.nn.functional.binary_cross_entropy_with_logits(out["quality_logit"], q); loss = loss_box + .5 * loss_quality
        else:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(v, g); pred = corrected_box(b, out["box_delta"]); loss_box = torch.nn.functional.smooth_l1_loss(pred[valid], gt[valid]) if valid.any() else pred.sum() * 0.; loss_quality = torch.nn.functional.binary_cross_entropy_with_logits(out["quality_logit"], q); loss = loss_box + .5 * loss_quality
        if not torch.isfinite(loss):
            amp_dtype = None; amp_mode = "fp32_fallback"; optimizer.zero_grad(set_to_none=True); out = model(v, g); pred = corrected_box(b, out["box_delta"]); loss_box = torch.nn.functional.smooth_l1_loss(pred[valid], gt[valid]) if valid.any() else pred.sum() * 0.; loss_quality = torch.nn.functional.binary_cross_entropy_with_logits(out["quality_logit"], q); loss = loss_box + .5 * loss_quality
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        rec: dict[str, Any] = {"step": step, "loss": float(loss.detach().cpu()), "loss_box": float(loss_box.detach().cpu()), "loss_quality": float(loss_quality.detach().cpu()), "amp": amp_mode}
        if step == smoke_steps or step % int(args.checkpoint_every) == 0:
            val_metrics = evaluate(model, val, device, amp_dtype if amp_mode == "bf16" else None); rec["validation"] = {k: v for k, v in val_metrics.items() if k != "corrected_iou_values"}; score = float(val_metrics["reliable_recall_iou50"] + .05 * (val_metrics["quality_ap"] or 0.0));
            payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "optimizer": optimizer.state_dict(), "global_step": step, "best_score": best_score, "best_step": best_step, "fold": args.fold, "seed": seed, "metadata": state_dict_metadata(model), "protocol": "trackocd_iclr27_phase22_class_agnostic_proposal_refiner", "source_csv_sha256": sha256(CSV_PATH), "feature_path": str(FEAT_PATH), "amp": amp_mode, "fit_rows": len(fit_idx), "validation_rows": len(val_idx)}
            atomic_torch(latest, payload)
            ckpt = ckpt_dir / f"proposal_refiner_{run_name}_step{step:05d}.pt"; atomic_torch(ckpt, payload)
            if score > best_score:
                best_score, best_step = score, step; payload["best_score"], payload["best_step"] = best_score, best_step; atomic_torch(ckpt_dir / f"proposal_refiner_{run_name}_best.pt", payload)
            rec["validation_score"] = score; rec["best_score"] = best_score; rec["checkpoint"] = str(ckpt); rec["elapsed_s"] = time.time() - t0; history.append(rec)
            with log_path.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
    final_metrics = evaluate(model, val, device, amp_dtype if amp_mode == "bf16" else None)
    result = {"protocol": "trackocd_iclr27_phase22_proposal_refiner_training", "fold": args.fold, "seed": seed, "steps": smoke_steps, "smoke": bool(args.smoke), "device": str(device), "amp": amp_mode, "fit_rows": len(fit_idx), "fit_positive_rows": int(len(pos)), "fit_negative_rows": int(len(neg)), "validation_rows": len(val_idx), "validation_held_categories": sorted(held_c), "validation_metrics": {k: v for k, v in final_metrics.items() if k != "corrected_iou_values"}, "best_score": best_score, "best_step": best_step, "history": history, "checkpoint_latest": str(latest), "checkpoint_best": str(ckpt_dir / f"proposal_refiner_{run_name}_best.pt"), "marker": str(marker), "done": str(done), "input_fields": state_dict_metadata(model), "forbidden_model_input_fields": ["gt_bbox_xyxy", "row_iou", "gt_category_id_common", "physical_id", "semantic_id", "future_frame"], "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}
    atomic_json(OUT / "metrics" / f"train_{run_name}.json", result); done.write_text(json.dumps({"fold": args.fold, "steps": smoke_steps, "checkpoint": str(ckpt_dir / f"proposal_refiner_{run_name}_best.pt"), "validation_reliable_recall_iou50": final_metrics["reliable_recall_iou50"]}, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fold": args.fold, "steps": smoke_steps, "fit_rows": len(fit_idx), "val_rows": len(val_idx), "val_recall": final_metrics["reliable_recall_iou50"], "best_step": best_step, "amp": amp_mode, "done": str(done)}, indent=2))


if __name__ == "__main__": main()
