"""Extract frozen DINOv2/DINOv3 features for the preregistered paired views."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.iclr27_phase17.representation.dinov3_local import LocalDinoV3

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data/iclr27_phase17/sources/tao_train_frames"
VIEWS = ["GT_TIGHT", "GT_CTX10", "PROPOSAL_RAW", "PROPOSAL_CTX10", "PROPOSAL_TEMPORAL",
         "JITTER_MILD", "JITTER_MEDIUM", "JITTER_SEVERE", "MULTISCALE_ROI_0",
         "MULTISCALE_ROI_10", "MULTISCALE_ROI_25"]


def _box(v):
    return None if v in (None, "", "None") else [float(x) for x in (json.loads(v) if isinstance(v, str) else v)]


def _crop(img: Image.Image, box: list[float], context: float = 0.0) -> Image.Image:
    w, h = img.size; x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 2.0), max(y2 - y1, 2.0); cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    x1, y1, x2, y2 = max(0, cx - nw / 2), max(0, cy - nh / 2), min(w, cx + nw / 2), min(h, cy + nh / 2)
    if x2 <= x1 + 1 or y2 <= y1 + 1: x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


def _jitter(box: list[float], frac: float, idx: int) -> list[float]:
    x1, y1, x2, y2 = box; w, h = x2 - x1, y2 - y1
    # Fixed, zero-randomness perturbations make the diagnostic reproducible.
    signs = ((1, -1), (-1, 1), (1, 1), (-1, -1))
    sx, sy = signs[idx % len(signs)]
    cx, cy = (x1 + x2) / 2 + sx * frac * w, (y1 + y2) / 2 + sy * frac * h
    nw, nh = w * (1 + frac * (0.5 if idx % 2 else -0.5)), h * (1 + frac * (0.5 if idx % 3 else -0.5))
    return [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]


def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [dict(r) for r in csv.DictReader(path.open())]
    rows = [r for r in rows if int(float(r.get("assigned", 0))) and r.get("gt_bbox_xyxy") not in ("", None)]
    rows.sort(key=lambda r: (r.get("role17", ""), int(r["gt_category_id_common"]), int(r["video_id"]), int(r["gt_track_id"]), int(r["source_frame_index"]), int(r["proposal_local_id"])))
    # Three causal observations per physical/semantic occurrence; then a
    # deterministic round-robin ensures every role/category/video contributes.
    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r.get("role17", "devplus"), int(r["gt_category_id_common"]), int(r["video_id"]), int(r["gt_track_id"]))].append(r)
    selected: list[dict[str, Any]] = []
    for k in sorted(groups): selected.extend(groups[k][:3])
    if len(selected) > limit:
        # Keep group-level coverage with a stable stride rather than taking a
        # first-video prefix (which would bias cross-video retrieval).
        idx = np.linspace(0, len(selected) - 1, limit, dtype=int)
        selected = [selected[int(i)] for i in idx]
    return selected


def _public_paths() -> dict[int, str]:
    ann = json.loads((ROOT / "data/iclr27_phase17/sources/public_dsct_annotation.json").read_text())
    return {int(i["id"]): i["file_name"] for i in ann["images"]}


def _dev_paths() -> dict[tuple[int, int], str]:
    out = {}
    with (ROOT / "data/iclr27_phase17/sources/devplus_gt_tracks.jsonl").open() as f:
        for line in f:
            if not line.strip(): continue
            g = json.loads(line); v, t = int(g["video_id"]), int(g["track_id"])
            for fi, p in zip(g["frame_indices"], g["image_paths"]): out[(v, int(fi))] = p
    return out


def _view_boxes(r: dict[str, Any], history: dict[tuple[int, int], list[list[float]]]) -> dict[str, list[float]]:
    gt = _box(r["gt_bbox_xyxy"]); prop = _box(r["bbox_xyxy"]); key = (int(r["video_id"]), int(r["track_id"]))
    prior = history[key] + [prop]; temporal = np.median(np.asarray(prior[-3:], dtype=float), axis=0).tolist(); history[key].append(prop)
    return {"GT_TIGHT": gt, "GT_CTX10": gt, "PROPOSAL_RAW": prop, "PROPOSAL_CTX10": prop,
            "PROPOSAL_TEMPORAL": temporal, "JITTER_MILD": _jitter(prop, .05, int(r["proposal_local_id"])),
            "JITTER_MEDIUM": _jitter(prop, .10, int(r["proposal_local_id"])),
            "JITTER_SEVERE": _jitter(prop, .20, int(r["proposal_local_id"])),
            "MULTISCALE_ROI_0": prop, "MULTISCALE_ROI_10": prop, "MULTISCALE_ROI_25": prop}


def _tensor(crop: Image.Image, size: int) -> torch.Tensor:
    """Fast PIL->tensor path; torchvision.to_tensor became the CPU bottleneck."""
    arr = np.asarray(crop.resize((size, size), Image.BILINEAR), dtype=np.float32).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().div_(255.0)
    mean = torch.tensor([.485, .456, .406], dtype=t.dtype)[:, None, None]
    std = torch.tensor([.229, .224, .225], dtype=t.dtype)[:, None, None]
    return (t - mean) / std


def _extract(model, tensors: list[torch.Tensor], device: str) -> np.ndarray:
    if not tensors: return np.empty((0, 768), dtype=np.float32)
    with torch.no_grad(): return model(torch.stack(tensors).to(device)).detach().cpu().numpy().astype(np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(args.rows, args.limit)
    public = args.domain == "public"; paths = _public_paths() if public else _dev_paths()
    device = args.device
    # DINOv2 is loaded from its already-audited local torch hub cache.
    hub = Path("/home/lwr/.cache/torch/hub/facebookresearch_dinov2_main")
    d2 = torch.hub.load(str(hub), "dinov2_vitb14", source="local").eval().to(device)
    d3 = LocalDinoV3(device=device, feature_mode="dense")
    history: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    d2_out = np.zeros((len(rows), len(VIEWS), 768), np.float32); d3_out = np.zeros_like(d2_out)
    failures: list[dict[str, Any]] = []
    # Process view-by-view so model memory is bounded and causal history is
    # fixed before any features are generated.
    all_tensors2: list[torch.Tensor] = []; all_tensors3: list[torch.Tensor] = []; loc: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        try:
            if public: rel = paths[int(r["image_id"])]
            else: rel = paths[(int(r["video_id"]), int(r["source_frame_index"]))]
            with Image.open(FRAMES / rel) as im: image = im.convert("RGB")
            boxes = _view_boxes(r, history)
            for j, view in enumerate(VIEWS):
                box = boxes[view]; ctx = .10 if view == "GT_CTX10" or view == "PROPOSAL_CTX10" or view == "MULTISCALE_ROI_10" else (.25 if view == "MULTISCALE_ROI_25" else 0.0)
                # The historical DINOv2 cache used 518px crops.  The paired
                # audit uses the same frozen ViT-B/14 at 224px so the eleven
                # views remain computationally tractable; this resolution is
                # recorded explicitly and is never mixed with the 518px
                # historical controller features.
                crop = _crop(image, box, ctx); all_tensors2.append(_tensor(crop, 224)); all_tensors3.append(_tensor(crop, 256)); loc.append((i, j))
                if len(all_tensors2) >= args.batch:
                    b2 = _extract(lambda x: torch.nn.functional.normalize(d2(x), dim=-1), all_tensors2, device); b3 = _extract(d3.embed, all_tensors3, device)
                    for z, (ii, jj) in enumerate(loc): d2_out[ii, jj] = b2[z]; d3_out[ii, jj] = b3[z]
                    all_tensors2.clear(); all_tensors3.clear(); loc.clear()
        except Exception as exc:
            failures.append({"row": i, "row_key": r.get("row_key"), "error": repr(exc)})
    if all_tensors2:
        b2 = _extract(lambda x: torch.nn.functional.normalize(d2(x), dim=-1), all_tensors2, device); b3 = _extract(d3.embed, all_tensors3, device)
        for z, (ii, jj) in enumerate(loc): d2_out[ii, jj] = b2[z]; d3_out[ii, jj] = b3[z]
    if failures: raise RuntimeError(f"paired crop failures: {failures[:3]}")
    if not np.isfinite(d2_out).all() or not np.isfinite(d3_out).all(): raise RuntimeError("non-finite paired features")
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    np.savez_compressed(tmp, dinov2=d2_out, dinov3=d3_out, row_keys=np.asarray([r["row_key"] for r in rows]),
                        video_id=np.asarray([int(r["video_id"]) for r in rows]), category_id=np.asarray([int(r["gt_category_id_common"]) for r in rows]),
                        gt_role=np.asarray([r["gt_role_common"] for r in rows]), role17=np.asarray([r.get("role17", args.domain) for r in rows]),
                        row_iou=np.asarray([float(r["row_iou"]) for r in rows]))
    generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp; os.replace(generated, args.out)
    meta = {"domain": args.domain, "rows": len(rows), "views": VIEWS, "feature_shape": list(d2_out.shape), "failures": failures,
            "dino2_source": str(hub), "dino2_paired_input_resolution": 224, "historical_dino2_cache_resolution": 518,
            "dino3": d3.metadata(), "frames": str(FRAMES.resolve()),
            "diagnostic_only_devplus_gt": not public, "gt_crops_public_diagnostic_only": True,
            "future_frames_used": False, "physical_id_used_as_feature": False, "q1_label_used": False}
    mp = args.meta; mp.parent.mkdir(parents=True, exist_ok=True); mt = mp.with_suffix(mp.suffix + ".tmp"); mt.write_text(json.dumps(meta, indent=2)); os.replace(mt, mp)
    print(json.dumps(meta, indent=2)); return meta


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--domain", choices=["public", "devplus"], default="public"); ap.add_argument("--rows", type=Path, default=ROOT / "outputs/iclr27_phase17/csv/public_role_rows.csv"); ap.add_argument("--limit", type=int, default=5000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", type=Path, default=ROOT / "outputs/iclr27_phase17/paired/public_paired_features.npz"); ap.add_argument("--meta", type=Path, default=ROOT / "outputs/iclr27_phase17/paired/public_paired_features_meta.json"); args = ap.parse_args(); run(args)


if __name__ == "__main__": main()
