"""Phase 4J frame-online subset runs (J1 calibration / J2 commitment).

Reuses the Phase 4I replay driver; the only changes are the calibrated
decision threshold and the global novel-memory commitment gate in
`SemanticStateManager`.
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
OUT_ROOT = ROOT / "outputs" / "iclr27_phase4j" / "subset"
LOG_ROOT = ROOT / "outputs" / "iclr27_phase4j" / "semantic_logs"


def load_video_ids():
    return [int(p.stem) for p in (EXPORT / "pre_assoc_detections").glob("*.jsonl")]


def parse_threshold(s):
    if "," in s:
        return tuple(float(x) for x in s.split(","))
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="")
    ap.add_argument("--lambda-s", type=float, default=0.1)
    ap.add_argument("--prefix-mode", default="P1")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--decision-threshold", default="0.5")
    ap.add_argument("--decision-split-age", type=int, default=None)
    ap.add_argument("--commit-mode", choices=["M0", "M1"], default="M0")
    ap.add_argument("--commit-min-age", type=int, default=2)
    ap.add_argument("--commit-min-support", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda")
    videos = [int(v) for v in args.videos.split(",") if v.strip()] or \
        load_video_ids()
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    sem = build_semantic_manager(
        model, device, prefix_mode=args.prefix_mode,
        decision_threshold=parse_threshold(args.decision_threshold),
        decision_split_age=args.decision_split_age,
        commit_mode=args.commit_mode,
        commit_min_age=args.commit_min_age,
        commit_min_support=args.commit_min_support)

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root = LOG_ROOT / args.tag
    log_root.mkdir(parents=True, exist_ok=True)
    for vid in videos:
        frames = frames_from_pre_assoc(EXPORT, vid)
        log_file = open(log_root / f"{vid}.jsonl", "w")
        rows = replay_video(vid, frames, EXPORT, FEAT_ROOT, out_dir,
                            sem_manager=sem, mode="B2",
                            lambda_s=args.lambda_s, device=device,
                            log_writer=log_file)
        for r in rows:
            log_file.write(json.dumps(r) + "\n")
        log_file.close()
        print("B2", args.tag, vid, "frames", len(frames), flush=True)

    cfg = {
        "tag": args.tag,
        "lambda_s": args.lambda_s,
        "prefix_mode": args.prefix_mode,
        "decision_threshold": args.decision_threshold,
        "decision_split_age": args.decision_split_age,
        "commit_mode": args.commit_mode,
        "commit_min_age": args.commit_min_age,
        "commit_min_support": args.commit_min_support,
        "checkpoint": "runs/orbit_mdc/mdc_m2/model.pth",
    }
    (LOG_ROOT / f"{args.tag}_config.json").write_text(
        json.dumps(cfg, indent=1))
    print("PHASE4J_SUBSET_DONE", args.tag)


if __name__ == "__main__":
    main()
