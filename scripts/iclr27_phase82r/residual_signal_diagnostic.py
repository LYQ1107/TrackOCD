#!/usr/bin/env python3
"""Read-only diagnostics for the Phase82P residual training signal.

The script reconstructs the same per-video causal candidate ordering as the
Phase82P manifest while retaining raw 768-D appearance for diagnostics. GT
track/category fields are used only to identify TRAIN targets and are never
placed in the model-input tensors.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
b = importlib.import_module("scripts.iclr27_phase82p.build_residual_manifest")  # read-only helper

OUT = ROOT / "outputs/iclr27_phase82r/audit/residual_signal_diagnostic.json"
APP = ROOT / "outputs/iclr27_phase82p/features/q0_dinov2.npz"


def qstats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.quantile(a, 0.10)), "p90": float(np.quantile(a, 0.90)), "min": float(a.min()), "max": float(a.max())}


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    if not scores_pos or not scores_neg:
        return 0.0
    p = np.asarray(scores_pos, dtype=np.float64)[:, None]
    n = np.asarray(scores_neg, dtype=np.float64)[None, :]
    return float(((p > n).mean() + 0.5 * (p == n).mean()))


def cosine(a: np.ndarray, z: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(z))
    return float(np.dot(a, z) / den) if den > 1e-8 else 0.0


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--appearance", type=Path, default=APP)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    # The Phase82P loader is reused read-only, but its appearance path is
    # overridden explicitly so this route can audit a corrected cache without
    # mutating the old module or output namespace.
    b.APPEARANCE = args.appearance
    rows, images, gt_by_image, features, allowed = b.load_inputs()
    event_vids = b.event_videos()
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        if int(row["video_id"]) in allowed and int(row["image_id"]) in images:
            grouped[int(row["video_id"])].append((idx, row))
    records: list[dict[str, Any]] = []
    by_fold: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    raw_pos_cos: list[float] = []; raw_neg_cos: list[float] = []
    geom_pos: dict[str, list[float]] = collections.defaultdict(list); geom_neg: dict[str, list[float]] = collections.defaultdict(list)
    history_lengths: list[float] = []; gaps: list[float] = []; candidate_counts: list[float] = []; margins: list[float] = []
    positive_total = 0; candidate_missing = 0
    vids = sorted(grouped); fold_by_video = {v: i % 4 for i, v in enumerate(vids)}
    for video in vids:
        by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
        for idx, row in grouped[video]:
            by_frame[int(images[int(row["image_id"])].get("frame_index", 0))].append((idx, row))
        frame_steps = {frame: step for step, frame in enumerate(sorted(by_frame))}
        states: dict[int, dict[str, Any]] = {}; seen: set[int] = set()
        for frame in sorted(by_frame):
            step = frame_steps[frame]
            current: list[dict[str, Any]] = []
            for idx, row in by_frame[frame]:
                box = np.asarray([float(row["bbox"][0]), float(row["bbox"][1]), float(row["bbox"][0] + row["bbox"][2]), float(row["bbox"][1] + row["bbox"][3])], dtype=np.float32)
                gt_track, gt_category, gt_iou = b.best_gt({**row, "bbox_xyxy": box}, gt_by_image)
                current.append({"idx": idx, "row": row, "box": box, "gt_track": gt_track, "gt_category": gt_category, "gt_iou": gt_iou, "track": int(row["track_id"]), "frame": step})
            for cur in current:
                if cur["track"] in seen:
                    continue
                image = images[int(cur["row"]["image_id"])]
                cur_obs = b.observation({**cur["row"], "bbox_xyxy": cur["box"]}, image, features[cur["idx"]], None, step, 0)
                dormant = [s for s in states.values() if int(s["track"]) != cur["track"] and step > int(s["frame"]) and step - int(s["frame"]) <= b.HORIZON]
                chosen = b.candidate_order({"obs": [cur_obs]}, dormant, step)
                positive = [s for s in chosen if int(s.get("gt_track", -1)) >= 0 and int(s.get("gt_track", -1)) == int(cur["gt_track"]) and int(cur["gt_track"]) >= 0]
                is_positive = bool(cur["gt_track"] >= 0 and positive)
                if not is_positive:
                    for s in chosen:
                        if int(s.get("gt_track", -1)) == int(cur["gt_track"]) and int(cur["gt_track"]) >= 0:
                            is_positive = True; positive = [s]; break
                if not is_positive:
                    # Only examples with a known same-GT fragment are positives;
                    # all other births are natural KEEP examples.
                    pass
                if positive:
                    positive_total += 1
                    by_fold[fold_by_video[video]]["positive"] += 1
                    p = positive[0]
                    pos_app = np.asarray(p["app"], dtype=np.float32)
                    neg_states = [s for s in chosen if s is not p]
                    pos_cos = cosine(features[cur["idx"]], pos_app)
                    neg_cos = [cosine(features[cur["idx"]], np.asarray(s["app"], dtype=np.float32)) for s in neg_states]
                    raw_pos_cos.append(pos_cos); raw_neg_cos.extend(neg_cos)
                    last = np.asarray(p["box"], dtype=np.float32); cb = cur["box"]
                    iw, ih = float(image.get("width", 640)), float(image.get("height", 480))
                    pc = np.asarray([(last[0] + last[2]) * .5 / iw, (last[1] + last[3]) * .5 / ih])
                    cc = np.asarray([(cb[0] + cb[2]) * .5 / iw, (cb[1] + cb[3]) * .5 / ih])
                    pwh = np.asarray([(last[2] - last[0]) / iw, (last[3] - last[1]) / ih]); cwh = np.asarray([(cb[2] - cb[0]) / iw, (cb[3] - cb[1]) / ih])
                    gap = float(step - int(p["frame"])); pred_c = pc + np.asarray(p["velocity"][:2]) * gap
                    geom_pos["center_distance"].append(float(np.linalg.norm(cc - pred_c))); geom_pos["scale_residual"].append(float(np.linalg.norm(cwh - pwh))); geom_pos["gap"].append(gap)
                    neg_vals = []
                    for s in neg_states:
                        sl = np.asarray(s["box"], dtype=np.float32); sc = np.asarray([(sl[0] + sl[2]) * .5 / iw, (sl[1] + sl[3]) * .5 / ih]); swh = np.asarray([(sl[2] - sl[0]) / iw, (sl[3] - sl[1]) / ih]); pred = sc + np.asarray(s["velocity"][:2]) * gap
                        neg_vals.append(float(np.linalg.norm(cc - pred))); geom_neg["center_distance"].append(neg_vals[-1]); geom_neg["scale_residual"].append(float(np.linalg.norm(cwh - swh))); geom_neg["gap"].append(gap)
                    history_lengths.append(float(len(p["obs"]))); gaps.append(gap); candidate_counts.append(float(len(chosen)))
                    # Positive rank under the deterministic candidate order.
                    rank = 1 + next(i for i, s in enumerate(chosen) if s is p)
                    candidate_missing += int(rank > b.MAX_CANDIDATES)
                    margins.append(float(pos_cos - max(neg_cos, default=-1.0)))
                    by_fold[fold_by_video[video]]["rank1"] += int(rank <= 1); by_fold[fold_by_video[video]]["rank4"] += int(rank <= 4); by_fold[fold_by_video[video]]["rank8"] += int(rank <= 8); by_fold[fold_by_video[video]]["rank16"] += int(rank <= 16)
                    records.append({"video_id": video, "frame_id": frame, "causal_step": step, "q0_track_id": int(cur["track"]), "gt_track_label": int(cur["gt_track"]), "candidate_count": len(chosen), "positive_rank": rank, "history_length": len(p["obs"]), "gap": gap, "positive_cosine": pos_cos, "hard_negative_cosine": max(neg_cos, default=-1.0), "margin": margins[-1], "center_distance_positive": geom_pos["center_distance"][-1], "center_distance_hard_negative": min(neg_vals, default=0.0)})
                # Candidate states retain their latest raw appearance and
                # velocity, while no ID enters an observation tensor.
            best: dict[int, dict[str, Any]] = {}
            for cur in current:
                if cur["track"] not in best or float(cur["row"].get("score", 0.0)) > float(best[cur["track"]]["row"].get("score", 0.0)):
                    best[cur["track"]] = cur
            for track, cur in best.items():
                prev = states.get(track); obs = b.observation({**cur["row"], "bbox_xyxy": cur["box"]}, images[int(cur["row"]["image_id"])], features[cur["idx"]], prev, step, len(prev["obs"]) if prev else 0)
                velocity = obs[12:16].copy()
                if prev is None:
                    states[track] = {"track": track, "frame": step, "box": cur["box"], "obs": [obs], "gt_track": cur["gt_track"], "age": 1, "app": features[cur["idx"]].copy(), "velocity": velocity}
                else:
                    prev["obs"].append(obs); prev["obs"] = prev["obs"][-b.K:]; prev["frame"] = step; prev["box"] = cur["box"]; prev["gt_track"] = cur["gt_track"]; prev["age"] += 1; prev["app"] = features[cur["idx"]].copy(); prev["velocity"] = velocity
            seen.update(best)
    out = {
        "schema_version": "trackocd.phase82r.residual_signal_diagnostic.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "q0_rows": len(rows), "non_event_videos": len(vids), "positive_reconnect_examples": positive_total, "candidate_missing_positive": candidate_missing,
        "candidate_recall": {"top1": float(sum(c["rank1"] for c in by_fold.values()) / max(1, positive_total)), "top4": float(sum(c["rank4"] for c in by_fold.values()) / max(1, positive_total)), "top8": float(sum(c["rank8"] for c in by_fold.values()) / max(1, positive_total)), "top16": float(sum(c["rank16"] for c in by_fold.values()) / max(1, positive_total))},
        "appearance_768d": {"positive": qstats(raw_pos_cos), "hard_negative": qstats(raw_neg_cos), "pair_auc": auc(raw_pos_cos, raw_neg_cos)},
        "geometry_motion": {k: {"positive": qstats(v), "hard_negative": qstats(geom_neg.get(k, [])), "positive_minus_negative_mean": float(np.mean(v) - np.mean(geom_neg.get(k, [0.0]))) if v else 0.0} for k, v in geom_pos.items()},
        "history": {"length": qstats(history_lengths), "gap": qstats(gaps), "candidate_count": qstats(candidate_counts), "positive_margin": qstats(margins)},
        "folds": {str(f): dict(c) for f, c in sorted(by_fold.items())},
        "records_count": len(records),
        "records_sample": records[:200],
        "event_videos_excluded": sorted(event_vids),
        "forbidden_inference_fields": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text"],
        "public_dev_q1_sealed_accessed": False,
    }
    out["appearance_path"] = str(args.appearance)
    out["appearance_sha256"] = hashlib.sha256(args.appearance.read_bytes()).hexdigest()
    atomic(args.out, out)
    print(json.dumps({k: out[k] for k in ["positive_reconnect_examples", "candidate_recall", "appearance_768d", "history"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
