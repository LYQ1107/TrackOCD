"""Evaluator-only diagnostics; never imported by the tracking adapter."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def summary(values):
    a = np.asarray(values)
    return {"count": len(a), "total": int(a.sum()),
            "mean": float(a.mean()) if len(a) else 0.,
            "median": float(np.median(a)) if len(a) else 0.,
            "p90": float(np.quantile(a, .9)) if len(a) else 0.,
            "max": int(a.max()) if len(a) else 0}


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                       - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    bb = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def load_evaluator(annotation, protocol):
    tao = json.loads(Path(annotation).read_text())
    roles = json.loads(Path(protocol).read_text())["role_mapping"]
    role_by_category = {}
    for role, key in [("base", "known_ids"), ("novel", "novel_ids"),
                      ("distractor", "distractor_ids")]:
        role_by_category.update({int(k): role for k in roles[key]})
    anns = defaultdict(list)
    for ann in tao["annotations"]:
        anns[int(ann["image_id"])].append(ann)
    return tao, anns, role_by_category


def gt_arrays(annotations, role_by_category):
    b, roles = [], []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        b.append([x, y, x+w, y+h])
        roles.append(role_by_category[int(ann["category_id"])])
    return np.asarray(b, dtype=np.float32).reshape(-1, 4), np.asarray(roles)


def recall_counts(gt_boxes, roles, predictions, threshold=.5):
    iou = box_iou(gt_boxes, predictions)
    covered = iou.max(axis=1) >= threshold if len(predictions) else np.zeros(len(gt_boxes), bool)
    result = {}
    for role in ["all", "base", "novel", "distractor"]:
        mask = np.ones(len(roles), bool) if role == "all" else roles == role
        result[role] = {"matched": int(np.sum(covered & mask)), "total": int(mask.sum())}
    # One-to-one spatial matching separates coverage from duplicate matches.
    matches = 0
    if iou.size:
        valid = iou >= threshold
        rr, cc = linear_sum_assignment(-(valid.astype(float) * (len(gt_boxes)+1) + iou))
        matches = int(valid[rr, cc].sum())
    result["one_to_one"] = {"matched": matches, "gt": len(gt_boxes),
                            "predictions": len(predictions),
                            "unmatched_predictions": len(predictions)-matches}
    return result


def add_counts(target, value):
    for group, fields in value.items():
        dst = target.setdefault(group, {})
        for k, v in fields.items():
            dst[k] = dst.get(k, 0) + v


def finish_recall(value):
    for role in ["all", "base", "novel", "distractor"]:
        if role in value:
            v = value[role]
            v["recall"] = v["matched"] / v["total"] if v["total"] else None
    return value


def read_video(path):
    # Only one compressed video is loaded at a time, never the full universe.
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    n = len(data["frame_index"])
    off = data["frame_offsets"]
    if len(off) != n+1 or off[0] != 0 or np.any(np.diff(off) < 0):
        raise ValueError(f"invalid offsets: {path}")
    if len(data["image_id"]) != n or not np.array_equal(data["frame_index"], np.arange(n)):
        raise ValueError(f"invalid timeline: {path}")
    for key in ["boxes", "scores", "prototype_id", "foreground_scores"]:
        if key in data and len(data[key]) != off[-1]:
            raise ValueError(f"row alignment failed: {path}, {key}")
    return data


def frame_rows(data, pos):
    a, b = map(int, data["frame_offsets"][pos:pos+2])
    return a, b, data["boxes"][a:b]
