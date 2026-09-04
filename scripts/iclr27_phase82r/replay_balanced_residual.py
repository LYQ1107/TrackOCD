#!/usr/bin/env python3
"""Causal replay of a Phase82R balanced residual on frozen native Q0 rows."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
base = __import__("scripts.iclr27_phase82p.build_residual_manifest", fromlist=["observation", "candidate_order"])
from src.iclr27_phase82r.balanced_residual import BalancedResidualGate, predict

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


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--tag", default="balanced_replay"); ap.add_argument("--max-videos", type=int); args = ap.parse_args()
    native = [json.loads(line) for line in NATIVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    z = np.load(APPEARANCE, allow_pickle=False); feats = np.asarray(z["features"], dtype=np.float32)
    if feats.shape != (len(native), 768): raise RuntimeError(f"native appearance shape {feats.shape} != {(len(native), 768)}")
    model = BalancedResidualGate().to(args.device); ck = torch.load(args.checkpoint, map_location=args.device); model.load_state_dict(ck["model"]); model.eval()
    by_video: dict[int, dict[tuple[int, int], list[tuple[int, dict[str, Any]]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for idx, row in enumerate(native): by_video[int(row["video_id"])][(int(row.get("frame_id", 0)), int(row.get("image_id", -1)))].append((idx, row))
    videos = sorted(by_video); videos = videos[: args.max_videos] if args.max_videos else videos
    out_rows: list[dict[str, Any]] = []; stats = collections.Counter(); dim_cache: dict[str, tuple[int, int]] = {}
    for video in videos:
        frames = sorted(by_video[video]); observed_steps = {key: step for step, key in enumerate(frames)}
        states: dict[int, dict[str, Any]] = {}; seen: set[int] = set(); canonical: dict[int, int] = {}
        def root(tid: int) -> int:
            while canonical.get(tid, tid) != tid:
                canonical[tid] = canonical.get(canonical[tid], canonical[tid]); tid = canonical[tid]
            return tid
        for frame_key in frames:
            frame_id, image_id = frame_key; step = observed_steps[frame_key]; frame_rows = by_video[video][frame_key]
            path_key = str(frame_rows[0][1].get("file_path", "")); size = dim_cache.get(path_key)
            if size is None:
                try:
                    from PIL import Image
                    with Image.open(FRAMES / path_key) as im: size = (int(im.width), int(im.height))
                except Exception: size = (640, 480)
                dim_cache[path_key] = size
            image_meta = {"width": size[0], "height": size[1]}
            decisions: dict[int, dict[str, Any]] = {}
            for idx, row in frame_rows:
                if row.get("bbox_xyxy") is None or str(row.get("lifecycle", "")) != "birth": continue
                original = int(row["physical_track_id"]); box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
                cur_obs = base.observation({**row, "bbox_xyxy": box}, image_meta, feats[idx], None, step, 0)
                dormant = [s for s in states.values() if step > int(s["frame"]) and step - int(s["frame"]) <= base.HORIZON]
                chosen = base.candidate_order({"obs": [cur_obs]}, dormant, step)
                if not chosen: stats["empty_candidate_births"] += 1; continue
                hist = np.zeros((1, base.MAX_CANDIDATES, base.K, base.OBS_DIM), dtype=np.float32); mask = np.zeros((1, base.MAX_CANDIDATES), dtype=bool); ids=[]
                for j, cand in enumerate(chosen):
                    seq = cand["obs"][-base.K:]; hist[0, j, -len(seq):] = np.asarray(seq, dtype=np.float32); mask[0, j] = True; ids.append(int(cand["track"]))
                with torch.no_grad(): out = model(torch.from_numpy(cur_obs[None]).to(args.device), torch.from_numpy(hist).to(args.device), torch.from_numpy(mask).to(args.device)); picked, prob = predict(out); picked = int(picked[0].cpu()) if bool(mask.any()) else 0; probability = float(prob[0].cpu())
                if picked > 0 and picked <= len(ids):
                    parent = ids[picked - 1]; canonical[original] = root(parent); action = "RECONNECT"; stats["reconnect_decisions"] += 1
                else:
                    action = "KEEP_Q0"; parent = None; stats["keep_decisions"] += 1
                decisions[original] = {"action": action, "candidate_index": picked if action == "RECONNECT" else 0, "candidate_original_track_id": parent, "gate_probability": probability, "candidate_count": len(chosen)}
            for idx, row in frame_rows:
                original = int(row["physical_track_id"]); out = dict(row); out["original_physical_track_id"] = original; out["physical_track_id"] = root(original)
                if original in decisions: out.update({"residual_action": decisions[original]["action"], "residual_candidate_index": decisions[original]["candidate_index"], "residual_candidate_original_track_id": decisions[original]["candidate_original_track_id"], "residual_gate_probability": decisions[original]["gate_probability"], "residual_candidate_count": decisions[original]["candidate_count"]})
                out_rows.append(out)
            best: dict[int, tuple[int, dict[str, Any], np.ndarray]] = {}
            for idx, row in frame_rows:
                if row.get("bbox_xyxy") is None: continue
                tid = int(row["physical_track_id"]); box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
                if tid not in best or float(row.get("base_score", 0.0)) > float(best[tid][1].get("base_score", 0.0)): best[tid] = (idx, row, box)
            for tid, (idx, row, box) in best.items():
                prev = states.get(tid); obs = base.observation({**row, "bbox_xyxy": box}, image_meta, feats[idx], prev, step, len(prev["obs"]) if prev else 0)
                if prev is None: states[tid] = {"track": tid, "frame": step, "box": box, "obs": [obs], "gt_track": -1, "age": 1}
                else: prev["obs"] = (prev["obs"] + [obs])[-base.K:]; prev["frame"] = step; prev["box"] = box; prev["age"] = int(prev.get("age", 1)) + 1
                seen.add(tid)
        stats["videos"] += 1
    out_path = ROOT / "outputs/iclr27_phase82r/replays" / f"{args.tag}.jsonl"; atomic_jsonl(out_path, out_rows)
    summary = {"schema_version": "trackocd.phase82r.balanced_residual_replay.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint), "native_path": str(NATIVE), "native_sha256": sha256(NATIVE), "appearance_path": str(APPEARANCE), "appearance_sha256": sha256(APPEARANCE), "tag": args.tag, "videos": len(videos), "rows": len(out_rows), "stats": dict(stats), "output": str(out_path), "observed_step_map": True, "dormant_horizon": base.HORIZON, "q0_non_birth_proposal_preserved": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(ROOT / "outputs/iclr27_phase82r/metrics" / f"replay_{args.tag}.json", summary); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
