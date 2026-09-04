#!/usr/bin/env python3
"""Fixed, parameter-free causal assignment using corrected raw appearance.

This is the registered final Phase82R escalation after residual/selective
replay.  It never changes proposal boxes or non-birth rows.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
APPEARANCE = ROOT / "outputs/iclr27_phase82r/features/native_dinov2_corrected_r1.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"); os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b)); return float(np.dot(a, b) / den) if den > 1e-8 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--tag", default="full_causal_r1"); ap.add_argument("--max-videos", type=int); args = ap.parse_args()
    native = [json.loads(line) for line in NATIVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    feats = np.asarray(np.load(APPEARANCE, allow_pickle=False)["features"], dtype=np.float32)
    if feats.shape != (len(native), 768): raise RuntimeError(f"appearance shape {feats.shape} != {(len(native), 768)}")
    by_video: dict[int, dict[tuple[int, int], list[tuple[int, dict[str, Any]]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for idx, row in enumerate(native): by_video[int(row["video_id"])][(int(row.get("frame_id", 0)), int(row.get("image_id", -1)))].append((idx, row))
    videos = sorted(by_video); videos = videos[: args.max_videos] if args.max_videos else videos
    out_rows: list[dict[str, Any]] = []; stats = collections.Counter(); dim_cache: dict[str, tuple[int, int]] = {}
    for video in videos:
        frames = sorted(by_video[video]); steps = {key: i for i, key in enumerate(frames)}; states: dict[int, dict[str, Any]] = {}; parent: dict[int, int] = {}
        def root(tid: int) -> int:
            while parent.get(tid, tid) != tid:
                parent[tid] = parent.get(parent[tid], parent[tid]); tid = parent[tid]
            return tid
        for key in frames:
            frame_id, image_id = key; step = steps[key]; frame_rows = by_video[video][key]; path_key = str(frame_rows[0][1].get("file_path", "")); size = dim_cache.get(path_key)
            if size is None:
                try:
                    from PIL import Image
                    with Image.open(FRAMES / path_key) as im: size = (int(im.width), int(im.height))
                except Exception: size = (640, 480)
                dim_cache[path_key] = size
            iw, ih = float(size[0]), float(size[1]); decisions: dict[int, dict[str, Any]] = {}; claimed: set[int] = set()
            for idx, row in frame_rows:
                if row.get("bbox_xyxy") is None or str(row.get("lifecycle", "")) != "birth": continue
                original = int(row["physical_track_id"]); box = np.asarray(row["bbox_xyxy"], dtype=np.float32); cc = np.asarray([(box[0] + box[2]) * .5 / iw, (box[1] + box[3]) * .5 / ih]); cwh = np.asarray([(box[2] - box[0]) / iw, (box[3] - box[1]) / ih]); candidates=[]
                for st in states.values():
                    # Only fragments that have explicitly terminated/lost are eligible.
                    # Active Q0 tracks must never be used as reconnect targets.
                    if str(st.get("status", "active")) != "dormant":
                        stats["active_candidates_skipped"] += 1
                        continue
                    gap = step - int(st["step"])
                    if gap <= 0 or gap > 16: continue
                    sb = st["box"]; sc = np.asarray([(sb[0] + sb[2]) * .5 / iw, (sb[1] + sb[3]) * .5 / ih]); swh = np.asarray([(sb[2] - sb[0]) / iw, (sb[3] - sb[1]) / ih]); vel = np.asarray(st.get("velocity", np.zeros(2)), dtype=np.float32); pred = sc + vel * gap
                    score = cosine(feats[idx], st["app"]) - float(np.linalg.norm(cc - pred)) - 0.5 * float(np.linalg.norm(cwh - swh)) - 0.01 * gap
                    candidates.append((score, int(st["track"]), gap))
                candidates.sort(key=lambda x: (-x[0], x[1])); best = candidates[0] if candidates else None
                if best is not None and best[0] >= 0.5 and root(best[1]) not in claimed:
                    parent[original] = root(best[1]); action = "RECONNECT"; stats["reconnect_decisions"] += 1; chosen = best
                else:
                    if best is not None and best[0] >= 0.5 and root(best[1]) in claimed: stats["same_frame_collision_fallback"] += 1
                    action = "KEEP_Q0"; stats["keep_decisions"] += 1; chosen = None
                decisions[original] = {"action": action, "candidate_original_track_id": int(chosen[1]) if chosen else None, "assignment_score": float(chosen[0]) if chosen else None, "candidate_gap": int(chosen[2]) if chosen else None, "candidate_count": len(candidates)}
                if chosen is not None: claimed.add(root(chosen[1])); stats["dormant_candidates_used"] += 1
            for idx, row in frame_rows:
                original = int(row["physical_track_id"]); out = dict(row); out["original_physical_track_id"] = original; out["physical_track_id"] = root(original)
                if original in decisions: out.update({"full_assignment_action": decisions[original]["action"], "full_assignment_candidate_original_track_id": decisions[original]["candidate_original_track_id"], "full_assignment_score": decisions[original]["assignment_score"], "full_assignment_candidate_gap": decisions[original]["candidate_gap"], "full_assignment_candidate_count": decisions[original]["candidate_count"]})
                out_rows.append(out)
            best_rows: dict[int, tuple[int, dict[str, Any], np.ndarray]] = {}
            for idx, row in frame_rows:
                raw_tid = int(row["physical_track_id"])
                if row.get("bbox_xyxy") is None:
                    # A termination row can be metadata-only.  Mark the last
                    # observed fragment dormant without inventing a box/feature.
                    if str(row.get("lifecycle", "")) == "termination":
                        st = states.get(root(raw_tid))
                        if st is not None:
                            st["status"] = "dormant"; st["step"] = step; stats["metadata_termination_rows"] += 1
                    continue
                tid = int(row["physical_track_id"]); box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
                if tid not in best_rows or float(row.get("base_score", 0.0)) > float(best_rows[tid][1].get("base_score", 0.0)): best_rows[tid] = (idx, row, box)
            for tid, (idx, row, box) in best_rows.items():
                sb = box; sc = np.asarray([(sb[0] + sb[2]) * .5 / iw, (sb[1] + sb[3]) * .5 / ih]); vel = np.zeros(2, dtype=np.float32)
                canonical_tid = root(tid)
                prev = states.get(canonical_tid)
                if prev is not None:
                    oldc = np.asarray([(prev["box"][0] + prev["box"][2]) * .5 / iw, (prev["box"][1] + prev["box"][3]) * .5 / ih]); vel = sc - oldc
                lifecycle = str(row.get("lifecycle", "")); status = "dormant" if lifecycle == "termination" else "active"
                states[canonical_tid] = {"track": canonical_tid, "step": step, "box": box, "app": feats[idx].copy(), "velocity": vel, "status": status}
                if canonical_tid != tid: stats["canonical_state_merges"] += 1
            # Preserve explicit termination metadata for tracks without a box.
            for idx, row in frame_rows:
                if str(row.get("lifecycle", "")) == "termination" and row.get("bbox_xyxy") is None:
                    st = states.get(root(int(row["physical_track_id"])))
                    if st is not None: st["status"] = "dormant"
        stats["videos"] += 1
    out_path = ROOT / "outputs/iclr27_phase82r/replays" / f"{args.tag}.jsonl"; atomic_jsonl(out_path, out_rows)
    summary = {"schema_version": "trackocd.phase82r.full_causal_assignment_replay.v2_dormant_only", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "native_path": str(NATIVE), "native_sha256": sha256(NATIVE), "appearance_path": str(APPEARANCE), "appearance_sha256": sha256(APPEARANCE), "tag": args.tag, "accept_score": 0.5, "videos": len(videos), "rows": len(out_rows), "stats": dict(stats), "output": str(out_path), "observed_step_map": True, "dormant_only_candidates": True, "same_frame_collision_fallback": True, "canonical_state_merge": True, "q0_non_birth_proposal_preserved": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(ROOT / "outputs/iclr27_phase82r/metrics" / f"replay_{args.tag}.json", summary); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
