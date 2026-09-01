"""Run B0/B1/B2 on the Phase 3A 20-video subset.

Usage:
  python src/iclr27_phase4i/run_subset.py --modes B0 B2 --lambda-s 0.25 \
    --prefix-mode P1 --gpu 3

B0 outputs are compared against the Phase 3A offline replay for
equivalence.  B1/B2 write predictions + per-detection semantic logs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.replay import (
    frames_from_pre_assoc,
    replay_video,
)

EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "detection_features"
REF_REPLAY = ROOT / "outputs" / "iclr27_phase3a" / "trajectories" / "offline_replay_20"
OUT_ROOT = ROOT / "outputs" / "frame_online_trackocd" / "subset"
LOG_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "semantic_logs"


def load_video_ids():
    return [int(p.stem) for p in (EXPORT / "pre_assoc_detections").glob("*.jsonl")]


def equivalence(video_id, out_dir):
    frames = frames_from_pre_assoc(EXPORT, video_id)
    n_img = len(frames)
    for fmeta in frames:
        name = f"{int(fmeta['image_id']):010d}.json"
        ref = REF_REPLAY / name
        got = out_dir / name
        if not ref.exists() or not got.exists():
            return {"video_id": video_id, "ok": False, "reason": "missing",
                    "image": name}
        a = sorted(json.loads(ref.read_text()),
                   key=lambda r: (r.get("track_id", -1), tuple(r["bbox"])))
        b = sorted(json.loads(got.read_text()),
                   key=lambda r: (r.get("track_id", -1), tuple(r["bbox"])))
        if len(a) != len(b):
            return {"video_id": video_id, "ok": False, "reason": "count",
                    "image": name, "ref": len(a), "got": len(b)}
        for x, y in zip(a, b):
            if x["bbox"] != y["bbox"] or x["track_id"] != y["track_id"] or \
                    abs(x["score"] - y["score"]) > 1e-6:
                return {"video_id": video_id, "ok": False,
                        "reason": "mismatch", "image": name}
    return {"video_id": video_id, "ok": True, "images": n_img}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["B0"])
    ap.add_argument("--videos", default="")
    ap.add_argument("--lambda-s", type=float, default=0.25)
    ap.add_argument("--prefix-mode", default="P1")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda")
    videos = [int(v) for v in args.videos.split(",") if v.strip()] or \
        load_video_ids()
    model = None
    if any(m in ("B1", "B2") for m in args.modes):
        from src.orbit_mdc.evaluate_mdc import load_mdc_model
        from src.frame_online_trackocd.semantic import build_semantic_manager
        model, ck = load_mdc_model("runs/orbit_mdc/mdc_m2/model.pth", device)
        model.eval()

    tag = args.tag or (f"l{args.lambda_s:g}" if any(m == "B2" for m in args.modes) else "")
    for mode in args.modes:
        out_dir = OUT_ROOT / mode / (tag or "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        sem = None
        if mode in ("B1", "B2"):
            sem = build_semantic_manager(model, device,
                                         prefix_mode=args.prefix_mode)
        eq_rows = []
        for vid in videos:
            frames = frames_from_pre_assoc(EXPORT, vid)
            log_file = None
            if mode in ("B1", "B2"):
                log_root = LOG_ROOT / f"{mode}_{tag or 'default'}" if tag else \
                    LOG_ROOT / mode
                log_root.mkdir(parents=True, exist_ok=True)
                log_file = open(log_root / f"{vid}.jsonl", "w")
            rows = replay_video(vid, frames, EXPORT, FEAT_ROOT, out_dir,
                                sem_manager=sem, mode=mode,
                                lambda_s=args.lambda_s, device=device,
                                log_writer=log_file)
            if log_file is not None:
                for r in rows:
                    log_file.write(json.dumps(r) + "\n")
                log_file.close()
            if mode == "B0":
                eq_rows.append(equivalence(vid, out_dir))
            print(mode, vid, "frames", len(frames), flush=True)
        if mode == "B0":
            n_ok = sum(1 for r in eq_rows if r["ok"])
            bad = [r for r in eq_rows if not r["ok"]][:5]
            print("B0 equivalence", n_ok, "/", len(eq_rows), bad, flush=True)
            (OUT_ROOT / "b0_equivalence.json").write_text(
                json.dumps(eq_rows, indent=1))


if __name__ == "__main__":
    main()
