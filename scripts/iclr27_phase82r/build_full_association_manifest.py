#!/usr/bin/env python3
"""Build all-row causal association supervision from the legal TRAIN stream.

The arrays contain only current/history observations and an integer action
target (0=NEW, 1..M=existing candidate).  Category and track identifiers are
used transiently to construct TRAIN labels and audit statistics; they are not
serialized as model features.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
Q0_JSON = ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json"
TRAIN_JSON = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")
EVENT_MANIFEST = ROOT / "outputs/iclr27_phase74s/manifests/model_events_v2.jsonl"
APPEARANCE = ROOT / "outputs/iclr27_phase82r/features/q0_dinov2_corrected_r1.npz"
OUT_ROOT = ROOT / "outputs/iclr27_phase82r/manifests"
DATA_ROOT = Path("/data2/usr_for_deadline/trackocd_phase82r/full_assoc_data")
OBS_DIM = 48
K = 8
HORIZON = 16
MAX_CANDIDATES = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def event_videos() -> set[int]:
    out: set[int] = set()
    if EVENT_MANIFEST.exists():
        for line in EVENT_MANIFEST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.update((int(row["source_video"]), int(row["target_video"])))
    return out


def box_xyxy(row: dict[str, Any]) -> np.ndarray:
    b = row["bbox"]
    return np.asarray([float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])], dtype=np.float32)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def best_gt(row: dict[str, Any], gt_by_image: dict[int, list[dict[str, Any]]]) -> tuple[int, int, float]:
    box = box_xyxy(row)
    best = (0.0, -1, -1)
    for gt in gt_by_image.get(int(row["image_id"]), []):
        value = box_iou(box, gt["bbox"])
        if value > best[0]:
            best = (value, int(gt["track"]), int(gt["category"]))
    return (best[1], best[2], float(best[0])) if best[0] >= 0.5 else (-1, best[2], float(best[0]))


def candidate_sort(cur: dict[str, Any], states: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    current = cur["obs"]
    cc, cwh, capp = current[4:6], current[6:8], current[-32:]
    out: list[tuple[float, tuple[int, int], dict[str, Any]]] = []
    for state in states:
        last = state["obs"][-1]
        gap = max(1, step - int(state["step"]))
        pred_c = last[4:6] + last[12:14] * gap
        motion = float(np.linalg.norm(cc - pred_c)) + 0.5 * float(np.linalg.norm(cwh - last[6:8]))
        visual = 1.0 - float(np.dot(capp, last[-32:]) / max(float(np.linalg.norm(capp) * np.linalg.norm(last[-32:])), 1e-8))
        rank = motion + 0.35 * visual + 0.01 * float(gap)
        out.append((rank, (int(state["video"]), int(state["track"])), state))
    out.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in out[:MAX_CANDIDATES]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appearance", type=Path, default=APPEARANCE)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--max-videos", type=int, default=0, help="bounded smoke subset; 0 uses all legal TRAIN videos")
    args = ap.parse_args()
    base = importlib.import_module("scripts.iclr27_phase82p.build_residual_manifest")
    base.APPEARANCE = args.appearance
    rows = json.loads(Q0_JSON.read_text(encoding="utf-8"))
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in train["images"]}
    gt_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for ann in train.get("annotations", []):
        if int(ann.get("iscrowd", 0)):
            continue
        b = ann["bbox"]
        gt_by_image[int(ann["image_id"])].append({
            "track": int(ann.get("track_id", -1)), "category": int(ann.get("category_id", -1)),
            "bbox": np.asarray([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32),
        })
    feat = np.asarray(np.load(args.appearance, allow_pickle=False)["features"], dtype=np.float32)
    if feat.shape != (len(rows), 768):
        raise RuntimeError(f"appearance shape {feat.shape} != {(len(rows), 768)}")
    excluded = event_videos()
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        vid = int(row["video_id"])
        if vid not in excluded and int(row["image_id"]) in images:
            grouped[vid].append((idx, row))
    # Fixed fold assignment is inherited from Phase82R residual manifest; this
    # is a deterministic video-disjoint TRAIN split, not a held-event split.
    videos = sorted(grouped)
    if args.max_videos:
        videos = videos[: args.max_videos]
        grouped = {v: grouped[v] for v in videos}
    fold_sets = [{v for i, v in enumerate(videos) if i % 4 == f} for f in range(4)]
    all_examples: list[dict[str, Any]] = []
    video_stats: dict[str, dict[str, Any]] = {}
    for video in videos:
        by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
        for idx, row in grouped[video]:
            by_frame[int(images[int(row["image_id"])].get("frame_index", 0))].append((idx, row))
        frame_steps = {frame: step for step, frame in enumerate(sorted(by_frame))}
        states: dict[tuple[int, int], dict[str, Any]] = {}
        examples = 0; existing = 0; new = 0; candidates = 0; tracks = set(); cats = set()
        for frame in sorted(by_frame):
            step = frame_steps[frame]
            frame_rows = by_frame[frame]
            current: list[dict[str, Any]] = []
            # Deduplicate only for state update; every original Q0 row remains
            # an association training example.
            for idx, row in frame_rows:
                key = (video, int(row["track_id"]))
                b = box_xyxy(row)
                gt_track, gt_cat, gt_iou = best_gt(row, gt_by_image)
                prev = states.get(key)
                obs = base.observation({**row, "bbox_xyxy": b}, images[int(row["image_id"])], feat[idx], prev, step, len(prev["obs"]) if prev else 0)
                current.append({"idx": idx, "row": row, "key": key, "box": b, "gt_track": gt_track, "gt_cat": gt_cat, "gt_iou": gt_iou, "obs": obs})
                tracks.add(key); cats.add(gt_cat)
            for cur in current:
                prior = [s for s in states.values() if 0 < step - int(s["step"]) <= HORIZON]
                chosen = candidate_sort(cur, prior, step)
                target = 0
                histories: list[np.ndarray] = []
                masks: list[bool] = []
                for j, state in enumerate(chosen, start=1):
                    seq = state["obs"][-K:]
                    pad = np.zeros((K, OBS_DIM), dtype=np.float32)
                    pad[-len(seq):] = np.asarray(seq, dtype=np.float32)
                    histories.append(pad); masks.append(True)
                    if target == 0 and cur["gt_track"] >= 0 and state.get("gt_track", -1) == cur["gt_track"]:
                        target = j
                while len(histories) < MAX_CANDIDATES:
                    histories.append(np.zeros((K, OBS_DIM), dtype=np.float32)); masks.append(False)
                all_examples.append({"video": video, "step": step, "current": cur["obs"], "history": np.stack(histories), "candidate_mask": np.asarray(masks, dtype=np.bool_), "target": target, "gt_track": cur["gt_track"], "gt_category": cur["gt_cat"], "gt_iou": cur["gt_iou"], "candidate_tracks": [int(s["track"]) for s in chosen]})
                examples += 1; existing += int(target > 0); new += int(target == 0); candidates += len(chosen)
            # Update each physical Q0 track once per observed frame.
            best_rows: dict[tuple[int, int], dict[str, Any]] = {}
            for cur in current:
                if cur["key"] not in best_rows or float(cur["row"].get("score", 0.0)) > float(best_rows[cur["key"]]["row"].get("score", 0.0)):
                    best_rows[cur["key"]] = cur
            for key, cur in best_rows.items():
                prev = states.get(key)
                obs = base.observation({**cur["row"], "bbox_xyxy": cur["box"]}, images[int(cur["row"]["image_id"])], feat[cur["idx"]], prev, step, len(prev["obs"]) if prev else 0)
                if prev is None:
                    states[key] = {"video": video, "track": key[1], "step": step, "box": cur["box"], "obs": [obs], "gt_track": cur["gt_track"]}
                else:
                    prev["obs"].append(obs); prev["obs"] = prev["obs"][-K:]; prev["step"] = step; prev["box"] = cur["box"]; prev["gt_track"] = cur["gt_track"]
        video_stats[str(video)] = {"examples": examples, "existing": existing, "new": new, "mean_candidates": candidates / max(1, examples), "tracks": len(tracks), "categories_with_gt": len([x for x in cats if x >= 0])}
    def arrays(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not items:
            return {"current": np.zeros((0, OBS_DIM), np.float32), "history": np.zeros((0, MAX_CANDIDATES, K, OBS_DIM), np.float16), "candidate_mask": np.zeros((0, MAX_CANDIDATES), bool), "target": np.zeros((0,), np.int64)}
        # float16 halves disk while trainer converts to float32 on device.
        return {"current": np.stack([x["current"] for x in items]).astype(np.float16), "history": np.stack([x["history"] for x in items]).astype(np.float16), "candidate_mask": np.stack([x["candidate_mask"] for x in items]).astype(bool), "target": np.asarray([x["target"] for x in items], dtype=np.int64)}
    args.data_root.mkdir(parents=True, exist_ok=True)
    folds = []
    for f in range(4):
        val_v = fold_sets[f]; fit_v = set(videos) - val_v
        fit = [x for x in all_examples if int(x["video"]) in fit_v]; val = [x for x in all_examples if int(x["video"]) in val_v]
        fit_path, val_path = args.data_root / f"fold{f}.npz", args.data_root / f"fold{f}_val.npz"
        for path, data in ((fit_path, arrays(fit)), (val_path, arrays(val))):
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, **data); os.replace(tmp, path)
        folds.append({"fold": f, "fit_videos": sorted(fit_v), "val_videos": sorted(val_v), "fit_examples": len(fit), "val_examples": len(val), "fit_existing": sum(x["target"] > 0 for x in fit), "val_existing": sum(x["target"] > 0 for x in val), "fit_path": str(fit_path), "val_path": str(val_path), "fit_sha256": sha256(fit_path), "val_sha256": sha256(val_path)})
    manifest = {"schema_version": "trackocd.phase82r.full_association_manifest.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": 8261, "q0_path": str(Q0_JSON), "q0_sha256": sha256(Q0_JSON), "train_annotations": str(TRAIN_JSON), "train_annotations_sha256": sha256(TRAIN_JSON), "appearance_path": str(args.appearance), "appearance_sha256": sha256(args.appearance), "event_videos_excluded": sorted(excluded), "videos": videos, "video_count": len(videos), "examples": len(all_examples), "observation_dim": OBS_DIM, "history_length": K, "horizon_observed_steps": HORIZON, "max_candidates": MAX_CANDIDATES, "folds": folds, "video_stats": video_stats, "inference_tensor_fields": ["normalized_box", "center", "size", "score", "causal_velocity", "age", "fixed_DINOv2_projection"], "label_fields_not_in_tensor": ["gt_track", "gt_category", "gt_iou", "candidate_tracks"], "forbidden_inference_fields": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text", "DEV+", "Q1", "public_new", "sealed"], "causal_contract": "per-video observed-frame order; candidate states strictly prior; no cross-video state; every Q0 row preserved as one action example", "split_contract": "same deterministic video-disjoint folds as Phase82R residual; event videos excluded; category only metadata audit"}
    atomic_json(args.out_root / "full_association_manifest.json", manifest); atomic_json(args.out_root / "full_association_supervision_inventory.json", manifest)
    print(json.dumps({"examples": len(all_examples), "videos": len(videos), "folds": folds}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
