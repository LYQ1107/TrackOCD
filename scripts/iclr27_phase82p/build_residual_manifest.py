#!/usr/bin/env python3
"""Build causal Q0 birth/fragment training examples with per-video reset.

All GT fields in this file are labels/metadata used to construct ``target``;
only numerical causal observation vectors are serialized as model inputs.
The history map is recreated for every video, matching runtime chronology.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")
Q0_JSON = ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json"
EVENT_MANIFEST = ROOT / "outputs/iclr27_phase74s/manifests/model_events_v2.jsonl"
APPEARANCE = ROOT / "outputs/iclr27_phase82p/features/q0_dinov2.npz"
OUT_ROOT = ROOT / "outputs/iclr27_phase82p/manifests"
DATA_ROOT = Path("/data2/usr_for_deadline/trackocd_phase82p/data")
SEED = 8201
K = 8
HORIZON = 16
MAX_CANDIDATES = 16
APP_DIM = 32
OBS_DIM = 49


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
    for line in EVENT_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.update((int(row["source_video"]), int(row["target_video"])))
    return out


def load_inputs() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], np.ndarray, set[int]]:
    rows = json.loads(Q0_JSON.read_text(encoding="utf-8"))
    train = json.loads(TRAIN_JSON.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in train["images"]}
    gt_by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for ann in train.get("annotations", []):
        if int(ann.get("iscrowd", 0)):
            continue
        b = [float(v) for v in ann["bbox"]]
        gt_by_image[int(ann["image_id"])].append({
            "track": int(ann.get("track_id", -1)), "category": int(ann.get("category_id", -1)),
            "bbox": np.asarray([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32),
        })
    z = np.load(APPEARANCE, allow_pickle=False)
    features = np.asarray(z["features"], dtype=np.float32)
    if features.shape != (len(rows), 768):
        raise RuntimeError(f"appearance shape {features.shape} does not match Q0 rows {len(rows)}")
    allowed = {int(row["video_id"]) for row in rows} - event_videos()
    return rows, images, gt_by_image, features, allowed


def best_gt(row: dict[str, Any], gt_by_image: dict[int, list[dict[str, Any]]]) -> tuple[int, int, float]:
    box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
    best = (0.0, -1, -1)
    for gt in gt_by_image.get(int(row["image_id"]), []):
        score = box_iou(box, gt["bbox"])
        if score > best[0]:
            best = (score, int(gt["track"]), int(gt["category"]))
    return best[1] if best[0] >= 0.5 else -1, best[2], float(best[0])


def norm_box(box: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iw, ih = max(1.0, width), max(1.0, height)
    b = np.asarray([box[0] / iw, box[1] / ih, box[2] / iw, box[3] / ih], dtype=np.float32)
    c = np.asarray([(box[0] + box[2]) * 0.5 / iw, (box[1] + box[3]) * 0.5 / ih], dtype=np.float32)
    wh = np.asarray([(box[2] - box[0]) / iw, (box[3] - box[1]) / ih], dtype=np.float32)
    return b, c, wh


def observation(row: dict[str, Any], image: dict[str, Any], feature: np.ndarray, previous: dict[str, Any] | None, frame: int, history_len: int) -> np.ndarray:
    box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
    b, c, wh = norm_box(box, float(image.get("width", 640)), float(image.get("height", 480)))
    if previous is None:
        vel = np.zeros(4, dtype=np.float32); gap = 0.0; age = 1.0; hit = 1.0
    else:
        dtf = max(1.0, float(frame - previous["frame"]))
        pb = np.asarray(previous["box"], dtype=np.float32)
        _, pc, pwh = norm_box(pb, float(image.get("width", 640)), float(image.get("height", 480)))
        vel = np.concatenate(((c - pc) / dtf, (wh - pwh) / dtf)).astype(np.float32)
        gap = min(HORIZON, float(frame - previous["frame"])) / HORIZON
        age = min(32, int(previous.get("age", 1)) + 1) / 32.0
        hit = min(32, int(previous.get("age", 1)) + 1) / max(1.0, min(32, int(previous.get("age", 1)) + 1))
    app = np.asarray(feature, dtype=np.float32)
    app /= max(float(np.linalg.norm(app)), 1e-8)
    # 8 bbox/center/size + score/gap/age/hit + 4 velocity + 32 fixed DINOv2
    out = np.concatenate((b, c, wh, np.asarray([
        float(row.get("base_score", row.get("score", 0.0))), gap, age, hit,
    ], dtype=np.float32), vel, app), axis=0)
    if out.shape != (OBS_DIM,) or not np.isfinite(out).all():
        raise RuntimeError(f"invalid observation shape/value {out.shape}")
    return out.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-8 else 0.0


def candidate_order(current: dict[str, Any], states: list[dict[str, Any]], frame: int) -> list[dict[str, Any]]:
    cur = current["obs"][-1]
    out = []
    cur_c = cur[4:6]
    cur_wh = cur[6:8]
    cur_app = cur[-APP_DIM:]
    for state in states:
        last = state["obs"][-1]
        gap = max(1, frame - int(state["frame"]))
        pred_c = last[4:6] + last[12:14] * gap
        motion = float(np.linalg.norm(cur_c - pred_c)) + 0.5 * float(np.linalg.norm(cur_wh - last[6:8]))
        visual = 1.0 - cosine(cur_app, last[-APP_DIM:])
        # Lower is better.  Fixed weights are registered, not tuned per event.
        rank = motion + 0.35 * visual + 0.01 * float(gap)
        out.append((rank, state))
    out.sort(key=lambda x: (x[0], int(x[1]["track"])))
    return [state for _, state in out[:MAX_CANDIDATES]]


def process_video(video: int, rows: list[tuple[int, dict[str, Any]]], images: dict[int, dict[str, Any]], gt_by_image: dict[int, list[dict[str, Any]]], features: np.ndarray) -> tuple[list[dict[str, Any]], collections.Counter]:
    by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for idx, row in rows:
        image = images[int(row["image_id"])]
        by_frame[int(image.get("frame_index", 0))].append((idx, row))
    states: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    examples: list[dict[str, Any]] = []
    stats = collections.Counter()
    # This map is intentionally created inside process_video: no cross-video history.
    for frame in sorted(by_frame):
        frame_rows = by_frame[frame]
        current_states: list[dict[str, Any]] = []
        for idx, row in frame_rows:
            image = images[int(row["image_id"])]
            box = np.asarray([float(row["bbox"][0]), float(row["bbox"][1]), float(row["bbox"][0] + row["bbox"][2]), float(row["bbox"][1] + row["bbox"][3])], dtype=np.float32)
            gt_track, gt_category, gt_iou = best_gt({**row, "bbox_xyxy": box}, gt_by_image)
            current_states.append({"idx": idx, "row": row, "box": box, "gt_track": gt_track, "gt_category": gt_category, "gt_iou": gt_iou, "track": int(row["track_id"]), "frame": frame})
        # Build birth examples before updating this frame's history.
        for cur in current_states:
            if cur["track"] in seen:
                continue
            image = images[int(cur["row"]["image_id"])]
            current_obs = observation({**cur["row"], "bbox_xyxy": cur["box"]}, image, features[cur["idx"]], None, frame, 0)
            cur_for_order = {"obs": [current_obs]}
            dormant = [s for s in states.values() if int(s["track"]) != cur["track"] and frame > int(s["frame"]) and frame - int(s["frame"]) <= HORIZON]
            chosen = candidate_order(cur_for_order, dormant, frame)
            target = 0
            histories: list[np.ndarray] = []
            mask: list[bool] = []
            for j, state in enumerate(chosen, start=1):
                seq = state["obs"][-K:]
                pad = np.zeros((K, OBS_DIM), dtype=np.float32)
                pad[-len(seq):] = np.asarray(seq, dtype=np.float32)
                histories.append(pad); mask.append(True)
                if target == 0 and cur["gt_track"] >= 0 and int(state.get("gt_track", -1)) == int(cur["gt_track"]):
                    target = j
            while len(histories) < MAX_CANDIDATES:
                histories.append(np.zeros((K, OBS_DIM), dtype=np.float32)); mask.append(False)
            if target > 0:
                stats["positive_reconnect"] += 1
            else:
                stats["keep_q0"] += 1
            stats["birth_examples"] += 1
            stats["candidate_total"] += len(chosen)
            stats["candidate_nonempty"] += int(bool(chosen))
            stats["gt_iou_ge_05"] += int(cur["gt_iou"] >= 0.5)
            stats["history_observations"] += sum(len(s["obs"]) for s in chosen)
            examples.append({
                "video_id": video, "frame_id": frame, "q0_track_id": cur["track"],
                "current": current_obs, "history": np.stack(histories),
                "candidate_mask": np.asarray(mask, dtype=np.bool_), "target": target,
                "target_gt_track_label": int(cur["gt_track"]), "target_gt_iou_label": float(cur["gt_iou"]),
                "candidate_track_labels": [int(s.get("gt_track", -1)) for s in chosen],
            })
        # Update states after all birth decisions, retaining the highest-score row
        # per Q0 track in this frame and only causal observations.
        best_rows: dict[int, dict[str, Any]] = {}
        for cur in current_states:
            if cur["track"] not in best_rows or float(cur["row"].get("score", 0.0)) > float(best_rows[cur["track"]]["row"].get("score", 0.0)):
                best_rows[cur["track"]] = cur
        for track, cur in best_rows.items():
            image = images[int(cur["row"]["image_id"])]
            prev = states.get(track)
            obs = observation({**cur["row"], "bbox_xyxy": cur["box"]}, image, features[cur["idx"]], prev, frame, len(prev["obs"]) if prev else 0)
            if prev is None:
                states[track] = {"track": track, "frame": frame, "box": cur["box"], "obs": [obs], "gt_track": cur["gt_track"], "age": 1}
            else:
                prev["obs"].append(obs); prev["obs"] = prev["obs"][-K:]; prev["frame"] = frame; prev["box"] = cur["box"]; prev["gt_track"] = cur["gt_track"]; prev["age"] = int(prev.get("age", 1)) + 1
        seen.update(best_rows)
        stats["frames"] += 1
    return examples, stats


def save_fold(fold: int, examples: list[dict[str, Any]], fit_videos: set[int], val_videos: set[int]) -> dict[str, Any]:
    fit = [e for e in examples if int(e["video_id"]) in fit_videos]
    val = [e for e in examples if int(e["video_id"]) in val_videos]
    # fixed arrays contain no category/ID fields; metadata labels are kept only
    # in the manifest JSON for post-hoc audit and target construction.
    def arrays(items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not items:
            return {"current": np.zeros((0, OBS_DIM), np.float32), "history": np.zeros((0, MAX_CANDIDATES, K, OBS_DIM), np.float32), "candidate_mask": np.zeros((0, MAX_CANDIDATES), bool), "target": np.zeros((0,), np.int64)}
        return {"current": np.stack([x["current"] for x in items]).astype(np.float32), "history": np.stack([x["history"] for x in items]).astype(np.float32), "candidate_mask": np.stack([x["candidate_mask"] for x in items]).astype(bool), "target": np.asarray([x["target"] for x in items], dtype=np.int64)}
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    fit_a, val_a = arrays(fit), arrays(val)
    fit_path = DATA_ROOT / f"fold{fold}.npz"; val_path = DATA_ROOT / f"fold{fold}_val.npz"
    for path, data in ((fit_path, fit_a), (val_path, val_a)):
        tmp = Path(str(path) + f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, **data); os.replace(tmp, path)
    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"examples": len(items), "positive_reconnect": sum(int(x["target"] > 0) for x in items), "keep_q0": sum(int(x["target"] == 0) for x in items), "videos": sorted({int(x["video_id"]) for x in items}), "mean_history_length": float(np.mean([sum(np.any(h, axis=1).astype(int)) for x in items for h in x["history"] if np.any(h)])) if any(np.any(x["history"]) for x in items) else 0.0}
    return {"fold": fold, "fit": stats(fit), "val": stats(val), "fit_path": str(fit_path), "val_path": str(val_path), "fit_sha256": sha256(fit_path), "val_sha256": sha256(val_path)}


def main() -> None:
    global APPEARANCE
    ap = argparse.ArgumentParser(); ap.add_argument("--appearance", type=Path, default=APPEARANCE); args = ap.parse_args()
    APPEARANCE = args.appearance
    rows, images, gt_by_image, features, allowed = load_inputs()
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        if int(row["video_id"]) in allowed and int(row["image_id"]) in images:
            b = [float(v) for v in row["bbox"]]; grouped[int(row["video_id"])].append((idx, row))
    all_examples: list[dict[str, Any]] = []; video_stats: dict[str, Any] = {}
    for video in sorted(grouped):
        ex, st = process_video(video, grouped[video], images, gt_by_image, features); all_examples.extend(ex); video_stats[str(video)] = dict(st)
    vids = sorted(grouped); fold_sets = [{v for i, v in enumerate(vids) if i % 4 == f} for f in range(4)]
    folds = [save_fold(f, all_examples, set(vids) - fold_sets[f], fold_sets[f]) for f in range(4)]
    manifest = {
        "schema_version": "trackocd.phase82p.residual_manifest.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "seed": SEED,
        "q0_path": str(Q0_JSON), "q0_sha256": sha256(Q0_JSON), "train_annotations": str(TRAIN_JSON), "train_annotations_sha256": sha256(TRAIN_JSON), "appearance_path": str(APPEARANCE), "appearance_sha256": sha256(APPEARANCE),
        "event_videos_excluded": sorted(event_videos()), "videos": vids, "video_count": len(vids), "q0_rows_used": sum(len(v) for v in grouped.values()), "examples": len(all_examples),
        "history_length": K, "horizon_frames": HORIZON, "max_candidates": MAX_CANDIDATES, "observation_dim": OBS_DIM, "appearance_dim": APP_DIM,
        "video_stats": video_stats, "folds": folds,
        "inference_tensor_fields": ["normalized_bbox", "center", "size", "base_score", "frame_gap", "age", "hit_ratio", "causal_velocity", "DINOv2_crop_embedding_32_fixed_projection"],
        "forbidden_inference_fields": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text"],
        "label_fields_not_in_tensor": ["target_gt_track_label", "target_gt_iou_label", "candidate_track_labels"],
        "history_contract": "per-video reset; chronological frame loop; observations only at or before current birth; no cross-video state",
        "candidate_contract": "dormant/lost Q0 fragments <=16 frames, deterministic causal motion+visual+recency order, max16",
        "split_contract": "event videos excluded; four deterministic video-disjoint folds; category used only for metadata audit",
    }
    atomic_json(OUT_ROOT / "residual_train_manifest.json", manifest)
    atomic_json(OUT_ROOT / "supervision_inventory.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["examples", "video_count", "q0_rows_used", "folds"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
