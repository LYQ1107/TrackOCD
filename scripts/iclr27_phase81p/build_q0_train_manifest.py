#!/usr/bin/env python3
"""Build causal association shards from the *actual* frozen Q0 proposals.

The original Phase81P shards used tight TRAIN ground-truth boxes.  This route
keeps the Q0 proposal universe and score semantics identical to inference and
uses TRAIN annotations only to create post-hoc association targets.  Category,
physical-ID and semantic-ID fields are never serialized as model features.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")
Q0_JSON = ROOT / "outputs/iclr27_phase4q/q0_long/teta_results/tao_track.json"
EVENT_MANIFEST = ROOT / "outputs/iclr27_phase74s/manifests/model_events_v2.jsonl"
DATA_ROOT = Path("/data2/usr_for_deadline/trackocd_phase81p/data")
OUT_ROOT = ROOT / "outputs/iclr27_phase81p/manifests"
SEED = 8111
PAIR_DIM = 16
PAD_CANDIDATES = 9
HISTORY_HORIZON = 8


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix1, iy1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ab = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0


def event_videos() -> set[int]:
    out: set[int] = set()
    if not EVENT_MANIFEST.is_file():
        return out
    for line in EVENT_MANIFEST.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            out.update((int(e["source_video"]), int(e["target_video"])))
    return out


def load_train() -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], set[int], set[int]]:
    data = json.loads(TRAIN_JSON.read_text())
    images = {int(x["id"]): x for x in data.get("images", [])}
    gt_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for a in data.get("annotations", []):
        if int(a.get("iscrowd", 0)):
            continue
        b = [float(x) for x in a["bbox"]]
        gt_by_image[int(a["image_id"])].append(
            {"track": int(a.get("track_id", -1)), "category": int(a.get("category_id", -1)),
             "bbox": np.asarray([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32)}
        )
    event_vids = event_videos()
    videos = {int(im["video_id"]) for im in images.values()} - event_vids
    categories = {int(a.get("category_id", -1)) for a in data.get("annotations", []) if not int(a.get("iscrowd", 0))}
    return images, gt_by_image, videos, categories


def load_q0(images: dict[int, dict[str, Any]], allowed_videos: set[int], gt_by_image: dict[int, list[dict[str, Any]]]) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Read Q0 JSON once and assign TRAIN-only post-hoc GT targets."""
    import ijson

    out: dict[int, dict[int, list[dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    with Q0_JSON.open("rb") as f:
        for row in ijson.items(f, "item"):
            iid = int(row.get("image_id", -1)); vid = int(row.get("video_id", -1))
            if vid not in allowed_videos or iid not in images:
                continue
            b = [float(x) for x in row["bbox"]]
            box = np.asarray([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32)
            best_iou, best_track, best_category = 0.0, -1, -1
            for gt in gt_by_image.get(iid, []):
                score = box_iou(box, gt["bbox"])
                if score > best_iou:
                    best_iou, best_track, best_category = score, int(gt["track"]), int(gt["category"])
            im = images[iid]
            out[vid][iid].append({
                "bbox_xyxy": box,
                "base_score": float(row.get("score", 0.0)),
                "frame_id": int(im.get("frame_index", 0)),
                # Target metadata is consumed only while constructing labels.
                "_gt_track": best_track if best_iou >= 0.5 else -1,
                "_gt_category": best_category,
                "_gt_iou": best_iou,
            })
    return out


def make_examples(frame_map: dict[int, list[dict[str, Any]]], image_meta: dict[int, dict[str, Any]], limit: int, use_motion: bool, allowed_categories: set[int]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from src.iclr27_phase81p.association import pair_features

    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    stats = collections.Counter()
    frame_order = sorted(frame_map, key=lambda iid: (int(image_meta[iid].get("frame_index", 0)), iid))
    history: dict[int, dict[str, Any]] = {}
    for pos, iid in enumerate(frame_order):
        current = [d for d in frame_map[iid] if int(d.get("_gt_category", -1)) in allowed_categories]
        if not current:
            continue
        history = {k: v for k, v in history.items() if pos - int(v["pos"]) <= HISTORY_HORIZON}
        candidates = list(history.values())
        for det in current:
            gt_track = int(det["_gt_track"])
            prev = [h for h in candidates if int(h["gt_track"]) == gt_track] if gt_track >= 0 else []
            others = [h for h in candidates if h not in prev]
            box = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            others.sort(key=lambda h: abs(float(h["box"][0] - box[0])) + abs(float(h["box"][1] - box[1])))
            chosen = prev[:1] + others[: PAD_CANDIDATES - 1]
            feats: list[np.ndarray] = []
            target = PAD_CANDIDATES  # NEW class
            for j, h in enumerate(chosen):
                d = {"bbox_xyxy": box, "frame_id": int(det["frame_id"]), "base_score": float(det["base_score"])}
                t = {"last_bbox": h["box"], "appearance_ema": np.zeros(8, dtype=np.float32),
                     "last_frame": int(h["frame_id"]), "velocity": h.get("velocity", np.zeros(4, dtype=np.float32)),
                     "age": int(h["age"]), "miss_count": max(0, pos - int(h["pos"])), "score_ema": float(h["score"]),
                     "association_ema": 0.0, "hit_count": int(h["age"])}
                feats.append(pair_features(d, t, use_motion=use_motion))
                if gt_track >= 0 and int(h["gt_track"]) == gt_track and target == PAD_CANDIDATES:
                    target = j
            arr = np.zeros((PAD_CANDIDATES, PAIR_DIM), dtype=np.float32)
            if feats:
                arr[: len(feats)] = np.asarray(feats, dtype=np.float32)
            x_rows.append(arr); y_rows.append(int(target))
            stats["examples"] += 1
            stats["positive"] += int(target < PAD_CANDIDATES)
            stats["new"] += int(target == PAD_CANDIDATES)
            stats["hard_negatives"] += max(0, len(feats) - 1)
            stats["matched_iou_ge_0_5"] += int(float(det["_gt_iou"]) >= 0.5)
            if len(x_rows) >= limit:
                break
        # Update one causal history state per GT track using its highest-IoU Q0 row.
        best_for_track: dict[int, dict[str, Any]] = {}
        for det in current:
            g = int(det["_gt_track"])
            if g < 0:
                continue
            if g not in best_for_track or float(det["_gt_iou"]) > float(best_for_track[g]["_gt_iou"]):
                best_for_track[g] = det
        for g, det in best_for_track.items():
            old = history.get(g)
            if old is None:
                vel = np.zeros(4, dtype=np.float32); age = 1
            else:
                dt = float(max(1, int(det["frame_id"]) - int(old["frame_id"])))
                vel = 0.8 * np.asarray(old.get("velocity", np.zeros(4, dtype=np.float32))) + 0.2 * (np.asarray(det["bbox_xyxy"]) - np.asarray(old["box"])) / dt
                age = int(old["age"]) + 1
            history[g] = {"gt_track": g, "box": np.asarray(det["bbox_xyxy"], dtype=np.float32), "frame_id": int(det["frame_id"]), "pos": pos, "age": age, "velocity": vel, "score": float(det["base_score"]), "_gt_iou": float(det["_gt_iou"])}
        if len(x_rows) >= limit:
            break
    if not x_rows:
        return np.zeros((0, PAD_CANDIDATES, PAIR_DIM), np.float32), np.zeros((0,), np.int64), dict(stats)
    return np.stack(x_rows), np.asarray(y_rows, dtype=np.int64), dict(stats)


def build_fold(fold: int, q0: dict[int, dict[int, list[dict[str, Any]]]], images: dict[int, dict[str, Any]], categories: set[int], limit_fit: int, limit_val: int, data_dir: Path, use_motion: bool) -> dict[str, Any]:
    held_categories = {c for c in categories if c % 4 == fold}
    val_videos = {v for v in q0 if v % 4 == fold}
    fit_videos = set(q0) - val_videos
    def select(videos: set[int], allowed_categories: set[int], limit: int):
        merged: dict[int, list[dict[str, Any]]] = {}
        for v in sorted(videos):
            # Categories are TRAIN metadata only; enforce category-disjointness
            # by retaining a video when its annotations include the fold role.
            merged.update(q0[v])
        return make_examples(merged, images, limit, use_motion, allowed_categories)
    fit_x, fit_y, fit_stats = select(fit_videos, categories - held_categories, limit_fit)
    val_x, val_y, val_stats = select(val_videos, held_categories, limit_val)
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_dir / f"fold{fold}.npz", x=fit_x, y=fit_y)
    np.savez_compressed(data_dir / f"fold{fold}_val.npz", x=val_x, y=val_y)
    return {"fold": fold, "fit_videos": sorted(fit_videos), "val_videos": sorted(val_videos), "held_categories": sorted(held_categories), "fit": fit_stats, "val": val_stats, "fit_path": str(data_dir / f"fold{fold}.npz"), "val_path": str(data_dir / f"fold{fold}_val.npz")}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-tag", default="q0_aligned")
    ap.add_argument("--motion", action="store_true")
    ap.add_argument("--fit-limit", type=int, default=150000)
    ap.add_argument("--val-limit", type=int, default=60000)
    args = ap.parse_args()
    np.random.seed(SEED)
    images, gt_by_image, videos, categories = load_train()
    q0 = load_q0(images, videos, gt_by_image)
    folds = [build_fold(f, q0, images, categories, args.fit_limit, args.val_limit, DATA_ROOT / args.route_tag, args.motion) for f in range(4)]
    result = {"schema_version": "phase81p.q0_aligned_manifest.v1", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "seed": SEED, "route_tag": args.route_tag, "q0_json": str(Q0_JSON), "q0_json_sha256": sha256(Q0_JSON), "train_annotations": str(TRAIN_JSON), "train_annotations_sha256": sha256(TRAIN_JSON), "excluded_event_videos": sorted(event_videos()), "video_count": len(videos), "category_count": len(categories), "q0_videos_loaded": len(q0), "motion_features": bool(args.motion), "history_horizon": HISTORY_HORIZON, "candidate_width": PAD_CANDIDATES, "folds": folds, "data_dir": str(DATA_ROOT / args.route_tag), "inference_tensor_forbidden": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt"], "label_rule": "TRAIN-only max IoU >= 0.5 assigns a post-hoc GT track target; no label field is serialized in x"}
    atomic_json(OUT_ROOT / args.route_tag / "train_manifest.json", result)
    atomic_json(OUT_ROOT / args.route_tag / "supervision_inventory.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
