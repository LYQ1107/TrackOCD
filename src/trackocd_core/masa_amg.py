"""Separate frozen SAM-grid -> anonymous-box -> MASA candidate.

Image/math utilities are byte-identical Meta SAM source (Apache-2.0), pinned
by manifest. This owned box-only wrapper follows the no-crop AMG defaults,
omits mask RLE storage, rejects degenerate boxes with explicit counts, and
uses mask stability as bounded tracker quality, NOT foreground probability.
It is not the RPN/ROI route or an official MASA benchmark reproduction.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/trackocd_core/audit/sam_amg_candidate_source"
DEFAULTS = {"points_per_side": 32, "points_per_batch": 64, "pred_iou_thresh": .88,
            "stability_score_thresh": .95, "stability_score_offset": 1.,
            "box_nms_thresh": .7, "crop_n_layers": 0, "min_mask_region_area": 0}


def verified_sources():
    manifest = json.loads((ROOT / "configs/trackocd_core/sam_amg_source_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["bytes"] > 32768:
            raise ValueError("Unsafe SAM source allowlist")
        path = SOURCE / relative
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError("Unexpected SAM source link")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != row["bytes"] or blob != row["git_blob_sha1"]:
            raise ValueError("Pinned SAM source identity mismatch: " + row["path"])
        rows.append({**row, "sha256": hashlib.sha256(data).hexdigest()})
    return rows


def utility_modules():
    """Load only pure utilities; never SAM registry/builders/predictor APIs."""
    verified_sources()
    modules = []
    for name in ("amg", "transforms"):
        spec = importlib.util.spec_from_file_location("trackocd_pinned_sam_" + name,
                                                     SOURCE / "segment_anything/utils" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def require_defaults(options):
    if options != DEFAULTS:
        raise ValueError("No AMG default search, crop expansion or proposal cap changes")


def filter_masks(logits, predicted_iou, utility):
    """Official IoU/stability filters, then disclosed geometric validity only."""
    import torch
    if (logits.ndim != 3 or predicted_iou.shape != logits.shape[:1]
            or not torch.isfinite(logits).all() or not torch.isfinite(predicted_iou).all()):
        raise ValueError("Invalid decoder shape/nonfinite logits or raw quality; no repair")
    raw_count = len(predicted_iou)
    keep = predicted_iou > DEFAULTS["pred_iou_thresh"]
    logits, predicted_iou = logits[keep], predicted_iou[keep]
    after_iou = len(predicted_iou)
    stability = utility.calculate_stability_score(logits, 0., DEFAULTS["stability_score_offset"])
    undefined = int((~torch.isfinite(stability)).sum())
    keep = stability >= DEFAULTS["stability_score_thresh"]
    logits, predicted_iou, stability = logits[keep], predicted_iou[keep], stability[keep]
    boxes = utility.batched_mask_to_box(logits > 0.).float()
    valid = torch.all(boxes[:, 2:] > boxes[:, :2], dim=1)
    counts = {"raw_masks": raw_count, "after_predicted_iou": after_iou,
              "undefined_stability_excluded": undefined, "after_stability": len(boxes),
              "degenerate_boxes_excluded": int((~valid).sum()), "before_nms": int(valid.sum())}
    return boxes[valid], predicted_iou[valid], stability[valid], counts


def infer_current_frame(model, image_rgb, ordinal, device, guard=lambda: None):
    """One encoder/current image; fixed generic foreground points, past memo."""
    import numpy as np
    import torch
    from torchvision.ops import batched_nms
    from mmdet.structures import DetDataSample
    from mmengine.structures import InstanceData

    if (image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8
            or type(ordinal) is not int or ordinal < 0):
        raise ValueError("Current RGB uint8 image and nonnegative local ordinal required")
    amg, transforms = utility_modules()
    transform = transforms.ResizeLongestSide(1024)
    height, width = image_rgb.shape[:2]
    resized = transform.apply_image(image_rgb)  # Official PIL resize, not old native cv2 resize.
    new_height, new_width = resized.shape[:2]
    pixels = torch.as_tensor(resized.copy(), device=device).permute(2, 0, 1).float()
    pixels = model.detector.preprocess(pixels).unsqueeze(0)  # Normalize exactly once.
    sample = DetDataSample(metainfo={"ori_shape": (height, width), "img_shape": (new_height, new_width),
        "batch_input_shape": (1024, 1024), "pad_shape": (1024, 1024), "frame_id": ordinal,
        "scale_factor": (new_width / width, new_height / height)})
    totals = dict.fromkeys(("raw_masks", "after_predicted_iou", "undefined_stability_excluded",
                           "after_stability", "degenerate_boxes_excluded", "before_nms"), 0)
    boxes, qualities, stabilities, raw_min, raw_max = [], [], [], None, None
    with torch.inference_mode():
        base = model.detector.backbone.forward_base_multi_level(pixels)
        if model.detector.backbone.out_indices[-1] != len(model.detector.backbone.blocks) - 1:
            raise ValueError("Last multi-level feature is not SAM's final block")
        embedding = model.detector.backbone.forward_neck(base[-1])
        adapted = model.masa_adapter(base)
        points = amg.build_point_grid(DEFAULTS["points_per_side"]) * np.asarray([[width, height]])
        for start in range(0, len(points), DEFAULTS["points_per_batch"]):
            guard()
            coords = torch.as_tensor(transform.apply_coords(points[start:start + 64], (height, width)), device=device)
            labels = torch.ones((len(coords), 1), dtype=torch.int, device=device)  # Positive point, no category.
            sparse, dense = model.detector.prompt_encoder(points=(coords[:, None, :], labels), boxes=None, masks=None)
            logits, quality = model.detector.mask_decoder(image_embeddings=embedding,
                image_pe=model.detector.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense, multimask_output=True)
            logits = model.detector.postprocess_masks(logits, (new_height, new_width), (height, width)).flatten(0, 1)
            quality = quality.flatten(0, 1)
            low, high = float(quality.min()), float(quality.max())
            raw_min, raw_max = low if raw_min is None else min(raw_min, low), high if raw_max is None else max(raw_max, high)
            b, q, s, counts = filter_masks(logits, quality, amg)
            for key in totals:
                totals[key] += counts[key]
            boxes.append(b); qualities.append(q); stabilities.append(s)
            del logits, quality, sparse, dense
            guard()
        boxes, qualities, stabilities = torch.cat(boxes), torch.cat(qualities), torch.cat(stabilities)
        keep = batched_nms(boxes, qualities, torch.zeros_like(qualities), DEFAULTS["box_nms_thresh"])
        sample.pred_instances = InstanceData(bboxes=boxes[keep], scores=stabilities[keep],
                                             labels=torch.zeros(len(keep), dtype=torch.long, device=device))
        tracks = model.tracker.track(model=model, img=pixels, feats=adapted, data_sample=sample,
                                     rescale=True, with_segm=False)
    guard()
    return sample.pred_instances, tracks, {**totals, "after_nms": len(keep), "prompt_batches": 16,
        "raw_predicted_iou_min": raw_min, "raw_predicted_iou_max": raw_max,
        "quality_contract": "Mask stability in [0,1]; not class/foreground probability. Raw predicted IoU ranks NMS only.",
        "crop_layers": 0, "proposal_cap": None, "encoder_forwards": 1}
