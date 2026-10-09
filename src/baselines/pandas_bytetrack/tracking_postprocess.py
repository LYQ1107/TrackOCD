"""Tracking-only class-agnostic TorchVision NMS of frozen detections.

No GT, class text or category role is accepted. Metadata follows selected
indices without influencing NMS. Equal foreground scores use input index order.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import nms


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def prepare_tracking_detections(boxes_xyxy, foreground_scores, semantic_scores,
                                prototype_ids, *, nms_iou, max_detections):
    """Return aligned NumPy arrays and original frame-local row indices.

    NMS runs on the boxes' device when supplied as a Torch tensor, otherwise
    on CPU. Unique artificial *ordering ranks* resolve score ties consistently
    without modifying output foreground scores. These ranks preserve the order
    of every unequal score; only that order matters to greedy TorchVision NMS.
    """
    device = boxes_xyxy.device if isinstance(boxes_xyxy, torch.Tensor) else torch.device("cpu")
    boxes, fg, sem, proto = map(_numpy, (boxes_xyxy, foreground_scores, semantic_scores, prototype_ids))
    n = len(boxes)
    if boxes.shape != (n, 4) or any(v.shape != (n,) for v in [fg, sem, proto]):
        raise ValueError("detection arrays must be Nx4 and aligned 1D vectors")
    if not np.isfinite(boxes).all() or not np.isfinite(fg).all():
        raise ValueError("geometry and foreground scores must be finite")
    if np.any(boxes[:, 2:] < boxes[:, :2]) or np.any((fg < 0) | (fg > 1)):
        raise ValueError("invalid boxes or foreground scores outside [0, 1]")
    if nms_iou is not None and not 0 <= nms_iou <= 1:
        raise ValueError("NMS IoU must be within [0, 1]")
    if max_detections is not None and (not isinstance(max_detections, int) or max_detections < 0):
        raise ValueError("max_detections must be a nonnegative integer or None")
    order = np.lexsort((np.arange(n), -fg.astype(np.float64)))
    if n and nms_iou is not None:
        # CPU/GPU tie behavior of raw TorchVision scores is unspecified. Unique
        # ranks make this explicit while using its standard suppression kernel.
        ordered = torch.as_tensor(boxes[order], dtype=torch.float64, device=device)
        ranks = torch.arange(n, 0, -1, dtype=torch.float64, device=device)
        keep = nms(ordered, ranks, float(nms_iou)).cpu().numpy()
        order = order[keep]
    if max_detections is not None:
        order = order[:max_detections]
    return {"boxes": boxes[order].copy(), "foreground_scores": fg[order].copy(),
            "semantic_scores": sem[order].copy(), "prototype_ids": proto[order].copy(),
            "original_detection_index": order.astype(np.int64),
            "output_index": np.arange(len(order), dtype=np.int64)}


def track_video(data, tracker, *, nms_iou, max_detections):
    """Keep every frame, including empty frames, and advance ByteTrack once.

    Detection offsets necessarily change when rows are suppressed. The original
    values are retained verbatim as source_frame_offsets; new candidate and
    track offsets describe the respective compressed streams on the same frame
    timeline. Metadata attached to Kalman boxes uses nearest-IoU candidate and
    is explicitly an approximate audit association, not raw proposal lineage.
    """
    frame_index, image_id = data["frame_index"], data["image_id"]
    offsets = data["frame_offsets"]
    candidate_indices, candidate_offsets = [], [0]
    rows, scores, sem, proto, tids, original_indices, track_offsets = [], [], [], [], [], [], [0]
    counts = []
    from src.iclr27_phase3b.bytetrack import iou_matrix
    for pos in range(len(frame_index)):
        a, b = map(int, offsets[pos:pos+2])
        d = prepare_tracking_detections(
            data["boxes"][a:b], data["foreground_scores"][a:b],
            data["scores"][a:b], data["prototype_id"][a:b],
            nms_iou=nms_iou, max_detections=max_detections)
        counts.append(len(d["boxes"]))
        candidate_indices.extend((a+d["original_detection_index"]).tolist())
        candidate_offsets.append(len(candidate_indices))
        bt = np.column_stack([d["boxes"], d["foreground_scores"]]).astype(np.float32)
        tracked = tracker.update(bt.reshape(-1, 5))
        overlaps = iou_matrix(tracked[:, :4], d["boxes"]) if len(tracked) else np.empty((0, len(bt)))
        for j, row in enumerate(tracked):
            index = int(overlaps[j].argmax())
            rows.append(row[:4]); tids.append(int(row[4])); scores.append(row[5])
            sem.append(d["semantic_scores"][index]); proto.append(d["prototype_ids"][index])
            original_indices.append(a+int(d["original_detection_index"][index]))
        track_offsets.append(len(tids))
    result = {"frame_index": frame_index.copy(), "image_id": image_id.copy(),
              "source_frame_offsets": offsets.copy(),
              "frame_offsets": np.asarray(track_offsets, np.int64),
              "candidate_frame_offsets": np.asarray(candidate_offsets, np.int64),
              "candidate_original_detection_index": np.asarray(candidate_indices, np.int64),
              "original_detection_index": np.asarray(original_indices, np.int64),
              "boxes": np.asarray(rows, np.float32).reshape(-1, 4),
              "track_id": np.asarray(tids, np.int64),
              "score": np.asarray(scores, np.float32),
              "tracking_score": np.asarray(scores, np.float32),
              "semantic_score": np.asarray(sem, np.float32),
              "prototype_id": np.asarray(proto, np.int32)}
    _, lengths = np.unique(result["track_id"], return_counts=True)
    return result, np.asarray(counts), lengths
