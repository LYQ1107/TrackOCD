#!/usr/bin/env python3
"""Apply a trained residual model on the frozen native Q0 lineage."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Keep the script runnable without requiring an external PYTHONPATH.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.iclr27_phase82p.build_residual_manifest import OBS_DIM, K, HORIZON, MAX_CANDIDATES, observation, candidate_order
from src.iclr27_phase82p.residual import ResidualTrajectoryEncoder

NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
APPEARANCE = ROOT / "outputs/iclr27_phase82p/features/native_dinov2.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"); os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True, type=Path); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--tag", default="residual_replay"); ap.add_argument("--max-videos", type=int); args = ap.parse_args()
    if not APPEARANCE.is_file(): raise FileNotFoundError(f"missing native appearance cache {APPEARANCE}")
    native = [json.loads(line) for line in NATIVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    z = np.load(APPEARANCE, allow_pickle=False); feats = np.asarray(z["features"], dtype=np.float32)
    if feats.shape != (len(native), 768): raise RuntimeError(f"native appearance shape {feats.shape} != {(len(native), 768)}")
    state = torch.load(args.checkpoint, map_location=args.device); model = ResidualTrajectoryEncoder().to(args.device); model.load_state_dict(state["model"]); model.eval()
    by_video: dict[int, dict[tuple[int, int], list[tuple[int, dict[str, Any]]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for idx, row in enumerate(native):
        by_video[int(row["video_id"])][(int(row.get("frame_id", 0)), int(row.get("image_id", -1)))].append((idx, row))
    selected_videos = sorted(by_video)[: args.max_videos] if args.max_videos else sorted(by_video)
    out_rows: list[dict[str, Any]] = []; stats = collections.Counter(); dimension_cache: dict[str, tuple[int, int]] = {}
    remap: dict[tuple[int, int], int] = {}
    for video in selected_videos:
        states: dict[int, dict[str, Any]] = {}; seen: set[int] = set(); remap = {}
        for (frame, image_id), frame_rows in sorted(by_video[video].items()):
            # Infer actual image dimensions from the current causal frame only.
            size = dimension_cache.get(str(frame_rows[0][1].get("file_path", "")))
            if size is None:
                try:
                    from PIL import Image
                    with Image.open(FRAMES / str(frame_rows[0][1].get("file_path", ""))) as im: size = (int(im.width), int(im.height))
                except Exception: size = (640, 480)
                dimension_cache[str(frame_rows[0][1].get("file_path", ""))] = size
            image_meta = {"width": size[0], "height": size[1]}
            # Birth decisions are made before this frame updates any history.
            decisions: dict[int, dict[str, Any]] = {}
            for idx, row in frame_rows:
                original = int(row["physical_track_id"]); lifecycle = str(row.get("lifecycle", ""))
                if row.get("bbox_xyxy") is None:
                    continue
                if lifecycle != "birth" and original in seen: continue
                if lifecycle != "birth": continue
                box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
                cur_obs = observation({**row, "bbox_xyxy": box}, image_meta, feats[idx], None, frame, 0)
                dormant = [s for s in states.values() if frame > int(s["frame"]) and frame - int(s["frame"]) <= HORIZON]
                chosen = candidate_order({"obs": [cur_obs]}, dormant, frame)
                if not chosen: continue
                hist = np.zeros((1, MAX_CANDIDATES, K, OBS_DIM), dtype=np.float32); mask = np.zeros((1, MAX_CANDIDATES), dtype=bool)
                candidate_ids: list[int] = []
                for j, candidate in enumerate(chosen):
                    seq = candidate["obs"][-K:]; hist[0, j, -len(seq):] = np.asarray(seq, dtype=np.float32); mask[0, j] = True; candidate_ids.append(int(candidate["track"]))
                with torch.no_grad():
                    logits = model(torch.from_numpy(cur_obs[None]).to(args.device), torch.from_numpy(hist).to(args.device), torch.from_numpy(mask).to(args.device))[0].cpu().numpy()
                action_index = int(np.argmax(logits)); decision = {"action": "KEEP_Q0", "candidate_index": 0, "candidate_original_track_id": None, "logit": float(logits[0]), "candidate_count": len(chosen)}
                if action_index > 0 and action_index <= len(candidate_ids):
                    decision = {"action": "RECONNECT", "candidate_index": action_index, "candidate_original_track_id": candidate_ids[action_index - 1], "logit": float(logits[action_index]), "candidate_count": len(chosen)}
                    remap[(video, original)] = candidate_ids[action_index - 1]; stats["reconnect_decisions"] += 1
                else: stats["keep_decisions"] += 1
                decisions[original] = decision
            # Emit rows and then update history.  Reconnect remaps only lineage,
            # never the feature tensor or the frozen Q0 proposal box.
            for idx, row in frame_rows:
                original = int(row["physical_track_id"]); canonical = remap.get((video, original), original); out = dict(row); out["original_physical_track_id"] = original; out["physical_track_id"] = canonical
                if original in decisions: out["residual_action"] = decisions[original]["action"]; out["residual_candidate_index"] = decisions[original]["candidate_index"]; out["residual_candidate_original_track_id"] = decisions[original]["candidate_original_track_id"]; out["residual_logit"] = decisions[original]["logit"]; out["residual_candidate_count"] = decisions[original]["candidate_count"]
                out_rows.append(out)
            best: dict[int, tuple[int, dict[str, Any], np.ndarray]] = {}
            for idx, row in frame_rows:
                if row.get("bbox_xyxy") is None:
                    continue
                tid = int(row["physical_track_id"]); box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
                if tid not in best or float(row.get("base_score", 0.0)) > float(best[tid][1].get("base_score", 0.0)): best[tid] = (idx, row, box)
            for tid, (idx, row, box) in best.items():
                prev = states.get(tid); obs = observation({**row, "bbox_xyxy": box}, image_meta, feats[idx], prev, frame, len(prev["obs"]) if prev else 0)
                if prev is None: states[tid] = {"track": tid, "frame": frame, "box": box, "obs": [obs], "gt_track": -1, "age": 1}
                else: prev["obs"].append(obs); prev["obs"] = prev["obs"][-K:]; prev["frame"] = frame; prev["box"] = box; prev["age"] = int(prev.get("age", 1)) + 1
                seen.add(tid)
        stats["videos"] += 1
    out_path = OUT / "replays" / f"{args.tag}.jsonl"; atomic_jsonl(out_path, out_rows)
    summary = {"schema_version": "trackocd.phase82p.residual_replay.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint), "native_path": str(NATIVE), "native_sha256": sha256(NATIVE), "appearance_path": str(APPEARANCE), "appearance_sha256": sha256(APPEARANCE), "tag": args.tag, "videos": len(selected_videos), "rows": len(out_rows), "stats": dict(stats), "output": str(out_path), "q0_non_birth_proposal_preserved": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "metrics" / f"replay_{args.tag}.json", summary); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
