"""Frame-grouped full Phase17R DINOv3 crop extraction.

Each worker owns one explicit shard/GPU. A frame is decoded once and all row
views from that frame are cropped before the next decode. Outputs are atomic
and keyed by the global corrected-row index.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.iclr27_phase17.representation.dinov3_local import LocalDinoV3

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data/iclr27_phase17r/sources/tao_train_frames"
VIEWS = ("PROPOSAL_RAW", "PROPOSAL_CTX10", "PROPOSAL_CTX25",
         "PROPOSAL_CAUSAL_SMOOTHED", "GT_TIGHT", "GT_CTX10")
MEAN = np.asarray([.485, .456, .406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([.229, .224, .225], dtype=np.float32).reshape(1, 1, 3)


def box(v: Any) -> list[float] | None:
    if v in (None, "", "None"):
        return None
    if isinstance(v, str):
        v = json.loads(v)
    return [float(x) for x in v]


def crop_tensor(image: Image.Image, bb: list[float], context: float, size: int = 256) -> torch.Tensor:
    width, height = image.size
    x1, y1, x2, y2 = bb
    bw, bh = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
    nw, nh = bw * (1.0 + 2.0 * context), bh * (1.0 + 2.0 * context)
    x1, y1 = max(0.0, cx - nw * .5), max(0.0, cy - nh * .5)
    x2, y2 = min(float(width), cx + nw * .5), min(float(height), cy + nh * .5)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        x1, y1, x2, y2 = 0.0, 0.0, min(2.0, float(width)), min(2.0, float(height))
    patch = image.crop((int(x1), int(y1), int(x2), int(y2))).resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    arr = (arr / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp, **arrays)
    generated = Path(str(tmp) + ".npz")
    if tmp.exists():
        generated = tmp
    os.replace(generated, path)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def run(args: argparse.Namespace) -> dict:
    all_rows = list(csv.DictReader(args.rows.open()))
    by_image: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(all_rows):
        by_image[int(row["image_id"])].append((index, row))
    image_ids = sorted(by_image)
    selected_ids = [im for pos, im in enumerate(image_ids) if pos % args.num_shards == args.shard_id]
    if args.limit_frames and len(selected_ids) > args.limit_frames:
        take = np.linspace(0, len(selected_ids) - 1, args.limit_frames, dtype=int)
        selected_ids = [selected_ids[int(i)] for i in take]
    items = [item for im in selected_ids for item in by_image[im]]
    local_of_global = {global_i: local_i for local_i, (global_i, _) in enumerate(items)}
    features = np.zeros((len(items), len(VIEWS), 768), dtype=np.float16)
    teacher_mask = np.zeros(len(items), dtype=np.uint8)

    device = "cuda:" + str(args.device)
    model = LocalDinoV3(device=device, feature_mode="dense")
    pending: list[torch.Tensor] = []
    destinations: list[tuple[int, int]] = []
    crop_count = 0
    failures = []
    start = time.time()

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending).pin_memory().to(device, non_blocking=True)
        # The local torch-1.12/CUDA-11.6 EVA rotary path is non-finite under
        # autocast (targeted FP32/AMP smoke recorded in the resource ledger).
        # Frozen feature extraction therefore stays FP32; output shards remain
        # float16 after unit-norm numerical validation.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            out = model.embed(batch).float().cpu().numpy()
        for feat, (li, vi) in zip(out, destinations):
            features[li, vi] = feat.astype(np.float16)
        pending.clear(); destinations.clear()

    for image_id in selected_ids:
        frame_items = by_image[image_id]
        rel = frame_items[0][1]["image_path"]
        try:
            with Image.open(FRAMES / rel) as im:
                image = im.convert("RGB")
                for global_i, row in frame_items:
                    li = local_of_global[global_i]
                    raw = box(row["bbox_xyxy"])
                    smooth = box(row["causal_smoothed_bbox_xyxy"])
                    gt = box(row.get("gt_bbox_xyxy"))
                    specs = [(raw, 0.0), (raw, .10), (raw, .25), (smooth, 0.0)]
                    if gt is not None and int(row["assigned"]):
                        specs.extend([(gt, 0.0), (gt, .10)])
                        teacher_mask[li] = 1
                    for vi, (bb, ctx) in enumerate(specs):
                        pending.append(crop_tensor(image, bb, ctx))
                        destinations.append((li, vi)); crop_count += 1
                        if len(pending) >= args.batch:
                            flush()
        except Exception as exc:
            failures.append({"image_id": image_id, "path": rel, "error": repr(exc)})
            if len(failures) >= 3:
                break
    flush()
    torch.cuda.synchronize(int(args.device))
    wall = time.time() - start
    if failures:
        raise RuntimeError("frame extraction failures: " + repr(failures[:3]))
    valid = np.linalg.norm(features.astype(np.float32), axis=-1)
    proposal_norms = valid[:, :4].reshape(-1)
    teacher_norms = valid[teacher_mask.astype(bool), 4:].reshape(-1)
    if not np.isfinite(features).all() or len(proposal_norms) == 0:
        raise RuntimeError("non-finite or empty features")
    if float(np.min(proposal_norms)) < .95 or float(np.max(proposal_norms)) > 1.05:
        raise RuntimeError("proposal feature normalization failed")
    if len(teacher_norms) and (float(np.min(teacher_norms)) < .95 or float(np.max(teacher_norms)) > 1.05):
        raise RuntimeError("teacher feature normalization failed")
    atomic_npz(args.out, features=features,
               global_index=np.asarray([i for i, _ in items], dtype=np.int64),
               row_keys=np.asarray([r["row_key"] for _, r in items]), teacher_mask=teacher_mask)
    meta = {
        "protocol": "trackocd_iclr27_phase17r_full_dinov3_shard",
        "shard_id": args.shard_id, "num_shards": args.num_shards, "device": int(args.device),
        "frames": len(selected_ids), "rows": len(items), "crops": crop_count,
        "wall_seconds": wall, "rows_per_second": len(items) / max(wall, 1e-9),
        "crops_per_second": crop_count / max(wall, 1e-9), "feature_shape": list(features.shape),
        "dtype": str(features.dtype), "views": list(VIEWS), "teacher_rows": int(teacher_mask.sum()),
        "proposal_norm_mean": float(proposal_norms.mean()),
        "teacher_norm_mean": float(teacher_norms.mean()) if len(teacher_norms) else None,
        "failures": failures, "frame_decode_once": True, "model": model.metadata(),
        "future_frames_used": False, "gt_deployment_input": False,
        "physical_id_semantic_feature": False
    }
    atomic_json(args.meta, meta)
    args.done.parent.mkdir(parents=True, exist_ok=True)
    args.done.write_text(json.dumps({"out": str(args.out), "rows": len(items), "wall_seconds": wall}))
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv")
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--done", type=Path, required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
