#!/usr/bin/env python3
"""Replay a frozen FullAssociation model on the causal native event stream."""
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
from src.iclr27_phase82r.full_association import FullAssociation
from scripts.iclr27_phase82p import build_residual_manifest as base

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


def candidate_sort(current: np.ndarray, states: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    cc, cwh, capp = current[4:6], current[6:8], current[-32:]
    scored = []
    for st in states:
        last = st["obs"][-1]; gap = max(1, step - int(st["step"]))
        pred_c = last[4:6] + last[12:14] * gap
        motion = float(np.linalg.norm(cc - pred_c)) + 0.5 * float(np.linalg.norm(cwh - last[6:8]))
        visual = 1.0 - float(np.dot(capp, last[-32:]) / max(float(np.linalg.norm(capp) * np.linalg.norm(last[-32:])), 1e-8))
        scored.append((motion + 0.35 * visual + 0.01 * gap, (int(st["raw"]),), st))
    scored.sort(key=lambda x: (x[0], x[1])); return [x[2] for x in scored[:16]]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", type=Path, required=True); ap.add_argument("--device", default="cpu"); ap.add_argument("--tag", default="full_assoc_replay_r1"); ap.add_argument("--max-videos", type=int); ap.add_argument("--native", type=Path, default=NATIVE); ap.add_argument("--appearance", type=Path, default=APPEARANCE)
    args = ap.parse_args(); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device); model = FullAssociation().to(device); model.load_state_dict(ck["model"]); model.eval()
    native = [json.loads(line) for line in args.native.read_text(encoding="utf-8").splitlines() if line.strip()]; feats = np.asarray(np.load(args.appearance, allow_pickle=False)["features"], dtype=np.float32)
    if feats.shape != (len(native), 768): raise RuntimeError(f"appearance shape {feats.shape} != {(len(native), 768)}")
    by_video: dict[int, dict[tuple[int, int], list[tuple[int, dict[str, Any]]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for idx, row in enumerate(native): by_video[int(row["video_id"])][(int(row.get("frame_id", 0)), int(row.get("image_id", -1)))].append((idx, row))
    videos = sorted(by_video); videos = videos[: args.max_videos] if args.max_videos else videos
    output: list[dict[str, Any]] = []; stats = collections.Counter(); dim_cache: dict[str, tuple[int, int]] = {}
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
            iw, ih = float(size[0]), float(size[1]); continuations = {root(int(r["physical_track_id"])) for _, r in frame_rows if str(r.get("lifecycle", "")) == "continuation" and r.get("bbox_xyxy") is not None}; decisions: dict[int, dict[str, Any]] = {}; claimed: set[int] = set()
            for idx, row in frame_rows:
                if row.get("bbox_xyxy") is None or str(row.get("lifecycle", "")) != "birth": continue
                raw = int(row["physical_track_id"]); box = np.asarray(row["bbox_xyxy"], dtype=np.float32)
                current = base.observation({**row, "bbox_xyxy": box}, {"width": iw, "height": ih}, feats[idx], None, step, 0)
                eligible = [s for s in states.values() if str(s.get("status", "dormant")) == "dormant" and root(int(s["raw"])) not in continuations and 0 < step - int(s["step"]) <= 16]
                chosen = candidate_sort(current, eligible, step); histories = []; mask = []
                for st in chosen:
                    seq = st["obs"][-8:]; pad = np.zeros((8, 48), dtype=np.float32); pad[-len(seq):] = np.asarray(seq, dtype=np.float32); histories.append(pad); mask.append(True)
                while len(histories) < 16: histories.append(np.zeros((8, 48), dtype=np.float32)); mask.append(False)
                cur_t = torch.from_numpy(current[None]).to(device=device, dtype=torch.float32); hist_t = torch.from_numpy(np.stack(histories)[None]).to(device=device, dtype=torch.float32); mask_t = torch.from_numpy(np.asarray(mask)[None]).to(device=device, dtype=torch.bool)
                with torch.no_grad():
                    out = model(cur_t, hist_t, mask_t); logits = torch.cat((out["new_logit"].unsqueeze(1), out["candidate_logits"]), dim=1); pred = int(logits.argmax(dim=1).item()); probs = torch.softmax(logits, dim=1); conf = float(probs[0, pred].item())
                if pred > 0 and pred <= len(chosen) and root(int(chosen[pred - 1]["raw"])) not in claimed:
                    st = chosen[pred - 1]; parent[raw] = root(int(st["raw"])); action = "RECONNECT"; stats["reconnect_decisions"] += 1; claimed.add(root(int(st["raw"])))
                else:
                    action = "KEEP_Q0"; stats["keep_decisions"] += 1
                decisions[raw] = {"action": action, "pred_index": pred, "confidence": conf, "candidate_count": len(chosen), "candidate_original_track_id": int(chosen[pred - 1]["raw"]) if pred > 0 and pred <= len(chosen) else None}
            # Keep one canonical ID per frame.  Losing children are detached,
            # never dropped, so MOT proposal/row counts remain unchanged.
            by_canonical: dict[int, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
            for _, row in frame_rows:
                if row.get("bbox_xyxy") is not None: by_canonical[root(int(row["physical_track_id"]))].append((int(row["physical_track_id"]), row))
            for cid, group in by_canonical.items():
                if len(group) <= 1: continue
                keep = cid if any(raw == cid for raw, _ in group) else max(group, key=lambda x: float(x[1].get("base_score", 0.0)))[0]
                for raw, _ in group:
                    if raw != keep:
                        parent.pop(raw, None); stats["same_frame_collision_fallback"] += 1
            for idx, row in frame_rows:
                raw = int(row["physical_track_id"]); out = dict(row); out["original_physical_track_id"] = raw; out["physical_track_id"] = root(raw)
                if raw in decisions: out.update({"full_association_action": decisions[raw]["action"], "full_association_pred_index": decisions[raw]["pred_index"], "full_association_confidence": decisions[raw]["confidence"], "full_association_candidate_count": decisions[raw]["candidate_count"], "full_association_candidate_original_track_id": decisions[raw]["candidate_original_track_id"]})
                output.append(out)
            best_rows: dict[int, tuple[int, dict[str, Any]]] = {}
            for idx, row in frame_rows:
                if row.get("bbox_xyxy") is None:
                    if str(row.get("lifecycle", "")) == "termination" and int(row["physical_track_id"]) in states: states[int(row["physical_track_id"])] ["status"] = "dormant"
                    continue
                tid = int(row["physical_track_id"])
                if tid not in best_rows or float(row.get("base_score", 0.0)) > float(best_rows[tid][1].get("base_score", 0.0)): best_rows[tid] = (idx, row)
            for tid, (idx, row) in best_rows.items():
                box = np.asarray(row["bbox_xyxy"], dtype=np.float32); prev = states.get(root(tid)); obs = base.observation({**row, "bbox_xyxy": box}, {"width": iw, "height": ih}, feats[idx], prev, step, len(prev["obs"]) if prev else 0); canonical = root(tid); lifecycle = str(row.get("lifecycle", "")); states[canonical] = {"raw": canonical, "step": step, "frame": step, "box": box, "obs": (prev["obs"] + [obs])[-8:] if prev else [obs], "status": "dormant" if lifecycle == "termination" else "active", "age": int(prev.get("age", 0) + 1) if prev else 1}
            for _, row in frame_rows:
                if str(row.get("lifecycle", "")) == "termination" and row.get("bbox_xyxy") is None:
                    st = states.get(root(int(row["physical_track_id"])))
                    if st is not None: st["status"] = "dormant"
        stats["videos"] += 1
    out_path = ROOT / "outputs/iclr27_phase82r/replays" / f"{args.tag}.jsonl"; atomic_jsonl(out_path, output); summary = {"schema_version": "trackocd.phase82r.full_association_replay.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "native_path": str(args.native), "native_sha256": sha256(args.native), "appearance_path": str(args.appearance), "appearance_sha256": sha256(args.appearance), "tag": args.tag, "videos": len(videos), "rows": len(output), "stats": dict(stats), "output": str(out_path), "candidate_contract": "dormant-only prior observed states, horizon16, max16; explicit NEW", "q0_rows_preserved": True, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}; atomic_json(ROOT / "outputs/iclr27_phase82r/metrics" / f"replay_{args.tag}.json", summary); print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
