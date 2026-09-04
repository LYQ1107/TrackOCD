#!/usr/bin/env python3
"""Causal physical re-association over the complete Phase83 Q0 stream.

This is intentionally a physical-only diagnostic.  It keeps every native Q0
row and base score, but creates a new canonical physical root when a birth
fragment is causally reconnected to a dormant fragment.  No GT, category,
text, future row, or semantic/physical identifier is used as a model input.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
OUT = ROOT / "outputs/iclr27_phase84"
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-8 else 0.0


def box_geometry(box: list[float], iw: float, ih: float) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(box, dtype=np.float32)
    return np.asarray([(b[0] + b[2]) * .5 / iw, (b[1] + b[3]) * .5 / ih]), np.asarray([(b[2] - b[0]) / iw, (b[3] - b[1]) / ih])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full_temporal_r1")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--accept-score", type=float, default=0.5)
    ap.add_argument("--max-gap", type=int, default=16)
    args = ap.parse_args()
    if not NATIVE.is_file() or not FEATURES.is_file(): raise FileNotFoundError("Phase83 native Q0 lineage/features missing")
    native = [json.loads(line) for line in NATIVE.open(encoding="utf-8") if line.strip()]
    features = np.asarray(np.load(FEATURES, allow_pickle=False)["features"], dtype=np.float32)
    if features.shape != (len(native), 768): raise RuntimeError(f"native feature shape {features.shape} != {(len(native), 768)}")
    by_video: dict[int, dict[tuple[int, int], list[tuple[int, dict[str, Any]]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for idx, row in enumerate(native):
        by_video[int(row["video_id"])][(int(row.get("frame_id", 0)), int(row.get("image_id", -1)))].append((idx, row))
    videos = sorted(by_video)
    if args.max_videos:
        if not 1 <= args.max_videos <= len(videos): raise ValueError("--max-videos out of range")
        videos = videos[:args.max_videos]
    output_rows: list[dict[str, Any]] = []; unions: list[dict[str, Any]] = []; stats = collections.Counter(); dim_cache: dict[str, tuple[int, int]] = {}
    for video in videos:
        frames = sorted(by_video[video]); steps = {key: i for i, key in enumerate(frames)}
        parent: dict[int, int] = {}; states: dict[int, dict[str, Any]] = {}

        def root(tid: int) -> int:
            parent.setdefault(tid, tid)
            while parent[tid] != tid:
                parent[tid] = parent.get(parent[tid], parent[tid]); tid = parent[tid]
            return tid

        for frame_key in frames:
            frame_id, image_id = frame_key; step = steps[frame_key]; frame_rows = by_video[video][frame_key]
            path_key = str(frame_rows[0][1].get("file_path", "")); size = dim_cache.get(path_key)
            if size is None:
                try:
                    from PIL import Image
                    with Image.open(FRAMES / path_key) as im: size = (int(im.width), int(im.height))
                except Exception: size = (640, 480); stats["missing_dimensions"] += 1
                dim_cache[path_key] = size
            iw, ih = float(size[0]), float(size[1]); claimed: set[int] = set(); decisions: dict[int, dict[str, Any]] = {}
            current_roots = {root(int(r["physical_track_id"])) for _, r in frame_rows if r.get("bbox_xyxy") is not None and str(r.get("lifecycle", "")) == "continuation"}
            for native_idx, row in frame_rows:
                if row.get("bbox_xyxy") is None or str(row.get("lifecycle", "")) != "birth": continue
                original = int(row["physical_track_id"]); parent.setdefault(original, original); box = row["bbox_xyxy"]; cc, cwh = box_geometry(box, iw, ih); candidates: list[tuple[float, int, int]] = []
                for state in states.values():
                    state_root = root(int(state["track"]))
                    if state.get("status", "active") != "dormant" or state_root in current_roots:
                        stats["active_candidates_skipped"] += 1; continue
                    gap = step - int(state["step"])
                    if gap <= 0 or gap > args.max_gap: continue
                    sb = state["box"]; sc, swh = box_geometry(sb, iw, ih); pred = sc + np.asarray(state.get("velocity", np.zeros(2)), dtype=np.float32) * gap
                    score = cosine(features[native_idx], np.asarray(state["app"], dtype=np.float32)) - float(np.linalg.norm(cc - pred)) - .5 * float(np.linalg.norm(cwh - swh)) - .01 * gap
                    candidates.append((score, state_root, gap))
                candidates.sort(key=lambda x: (-x[0], x[1])); best = candidates[0] if candidates else None
                if best is not None and best[0] >= args.accept_score and best[1] not in claimed and best[1] != root(original):
                    old_root = root(original); new_root = root(best[1]); parent[old_root] = new_root; claimed.add(new_root); action = "RECONNECT"
                    event = {"video_id": video, "frame_id": frame_id, "image_id": image_id, "step": step, "child_original_physical_track_id": original, "parent_original_physical_track_id": int(best[1]), "parent_canonical_physical_track_id": int(new_root), "score": float(best[0]), "gap": int(best[2]), "reason": "dormant_causal_appearance_motion_geometry", "max_gap": args.max_gap, "accept_score": args.accept_score}
                    unions.append(event); stats["reconnect_decisions"] += 1
                else:
                    action = "KEEP_Q0"; stats["keep_decisions"] += 1
                    if best is not None and best[0] >= args.accept_score and best[1] in claimed: stats["same_frame_collision_fallback"] += 1
                decisions[original] = {"action": action, "candidate_original_track_id": int(best[1]) if best else None, "assignment_score": float(best[0]) if best else None, "candidate_gap": int(best[2]) if best else None, "candidate_count": len(candidates)}
            # Preserve all rows and prevent two bbox detections sharing one
            # canonical ID in a frame.  The lower-score fragment is detached
            # only from the newly created union; no row is removed.
            by_canonical: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
            for _, row in frame_rows:
                if row.get("bbox_xyxy") is not None: by_canonical[root(int(row["physical_track_id"]))].append((int(row["physical_track_id"]), row))
            for canonical_id, group in by_canonical.items():
                if len(group) <= 1: continue
                root_rows = [x for x in group if x[0] == canonical_id]; keep = canonical_id if root_rows else max(group, key=lambda x: (float(x[1].get("base_score", 0.0)), -x[0]))[0]
                for raw_id, _ in group:
                    if raw_id != keep and raw_id != root(raw_id): parent.pop(raw_id, None); stats["lineage_collision_fallback"] += 1
            for native_idx, row in frame_rows:
                original = int(row["physical_track_id"]); canonical = root(original); out = dict(row)
                out.update({"phase84_row_uid": f"{video}:{image_id}:{frame_id}:{native_idx}:{int(row.get('candidate_rank', 0))}", "original_physical_track_id": original, "physical_track_id": canonical, "phase84_canonical_physical_track_id": canonical, "phase84_parent_assignment_action": decisions.get(original, {}).get("action", "Q0")})
                if original in decisions: out.update({"phase84_assignment_score": decisions[original]["assignment_score"], "phase84_assignment_gap": decisions[original]["candidate_gap"], "phase84_assignment_candidate_count": decisions[original]["candidate_count"], "phase84_candidate_original_track_id": decisions[original]["candidate_original_track_id"]})
                output_rows.append(out)
            # Update causal state after decisions, using only observed rows.
            best_rows: dict[int, tuple[int, dict[str, Any]]] = {}
            for native_idx, row in frame_rows:
                if row.get("bbox_xyxy") is None: continue
                canonical = root(int(row["physical_track_id"]))
                if canonical not in best_rows or float(row.get("base_score", 0.0)) > float(best_rows[canonical][1].get("base_score", 0.0)): best_rows[canonical] = (native_idx, row)
            for canonical, (native_idx, row) in best_rows.items():
                box = row["bbox_xyxy"]; center, _ = box_geometry(box, iw, ih); prev = states.get(canonical); velocity = np.zeros(2, dtype=np.float32) if prev is None else center - prev["center"]
                status = "dormant" if str(row.get("lifecycle", "")) == "termination" else "active"; states[canonical] = {"track": canonical, "step": step, "box": box, "center": center, "velocity": velocity, "app": features[native_idx].copy(), "status": status}
            for _, row in frame_rows:
                if str(row.get("lifecycle", "")) == "termination" and row.get("bbox_xyxy") is None:
                    state = states.get(root(int(row["physical_track_id"])))
                    if state is not None: state["status"] = "dormant"
        stats["videos"] += 1
    lineage = OUT / "physical/full_temporal_lineage.jsonl"; union_path = OUT / "physical/union_events.jsonl"
    atomic_jsonl(lineage, output_rows); atomic_jsonl(union_path, unions)
    summary = {"schema_version": "trackocd.phase84.full_temporal_physical.v1", "phase": "Phase84 A84P", "tag": args.tag, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "native_path": str(NATIVE.resolve()), "native_sha256": sha256(NATIVE), "native_features": str(FEATURES.resolve()), "native_features_sha256": sha256(FEATURES), "rows": len(output_rows), "videos": len(videos), "lineage": str(lineage.resolve()), "lineage_sha256": sha256(lineage), "union_events": str(union_path.resolve()), "union_events_sha256": sha256(union_path), "accept_score": args.accept_score, "max_gap": args.max_gap, "causal_temporal_appearance": True, "dormant_only_candidates": True, "observed_step_timing": True, "same_frame_collision_safe": True, "q0_rows_preserved": True, "stats": dict(stats), "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "physical/full_temporal_summary.json", summary); atomic_json(OUT / "status.json", {"phase": "Phase84", "route": "A84P", "status": "A84P_COMPLETE", "summary": str((OUT / "physical/full_temporal_summary.json").resolve()), "public_dev_q1_sealed_accessed": False})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
