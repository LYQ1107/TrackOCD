"""Phase 4L held-out replay + evaluation driver.

Parameterized version of the Phase 4K provenance replay for the held-out
export.  `--tag j1b` is the frozen anchor; `--tag m1` the hard gate;
candidate tags pass explicit semantic configuration.

Modes:
  replay: write preds + semantic logs for the held-out videos.
  eval:   TrackEval (HOTA/CLEAR/Identity) + Phase 4J semantic metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.replay import (
    frames_from_pre_assoc,
    replay_video,
)

DEFAULTS = {
    "export": ROOT / "outputs" / "iclr27_phase4l" / "heldout_export",
    "feat": ROOT / "outputs" / "iclr27_phase4l" / "heldout_features",
    "out": ROOT / "outputs" / "iclr27_phase4l" / "heldout" / "runs",
    "tao": ROOT / "outputs" / "iclr27_phase4l" / "heldout" / \
        "validation_heldout_tao.json",
    "trackeval_gt": ROOT / "outputs" / "iclr27_phase4l" / "heldout" / \
        "trackeval" / "gt",
}

CONFIGS = {
    "j1b": {"decision_threshold": 0.30, "commit_mode": "M0",
            "commit_min_age": 2, "commit_min_support": 2},
    "m1": {"decision_threshold": 0.30, "commit_mode": "M1",
           "commit_min_age": 2, "commit_min_support": 2},
}


def load_video_ids(export):
    return [int(p.stem) for p in (export / "pre_assoc_detections").glob(
        "*.jsonl")]


def replay(args, cfg):
    device = torch.device("cuda")
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    sem = build_semantic_manager(
        model, device, prefix_mode="P1",
        decision_threshold=cfg["decision_threshold"],
        commit_mode=cfg["commit_mode"],
        commit_min_age=cfg["commit_min_age"],
        commit_min_support=cfg["commit_min_support"])
    out_root = args.out / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    pred_root = out_root / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "semantic_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    videos = load_video_ids(args.export)
    for vid in sorted(videos):
        frames = frames_from_pre_assoc(args.export, vid)
        log_file = open(log_root / f"{vid}.jsonl", "w")
        rows = replay_video(vid, frames, args.export, args.feat, pred_root,
                            sem_manager=sem, mode="B2", lambda_s=0.1,
                            device=device, log_writer=log_file)
        for r in rows:
            log_file.write(json.dumps(r) + "\n")
        log_file.close()
        print("heldout", args.tag, vid, "frames", len(frames), flush=True)
    cfg["tag"] = args.tag
    cfg["export"] = str(args.export)
    (out_root / "config.json").write_text(json.dumps(cfg, indent=1))
    print("HELDOUT_REPLAY_DONE", args.tag)


def evaluate(args):
    out_root = args.out / args.tag
    pred_root = out_root / "preds"
    # TrackEval input
    te_root = ROOT / "outputs" / "iclr27_phase4l" / "heldout" / "trackeval"
    trackers_root = te_root / "trackers"
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4i" /
                            "build_trackeval_input.py"),
        "--input-dir", str(pred_root),
        "--tracker-name", args.tag,
        "--output-root", str(trackers_root),
    ], check=True)
    env = dict(os.environ)
    env["PHASE4L_TRACKEVAL_GT"] = str(DEFAULTS["trackeval_gt"])
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4i" /
                            "run_trackeval_subset.py"),
        "--trackers-folder", str(trackers_root),
        "--names", args.tag,
        "--out", str(te_root / f"tracking_{args.tag}.json"),
    ], env=env, check=True)
    # semantic metrics
    env["PHASE4L_TAO_JSON"] = str(DEFAULTS["tao"])
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4j" /
                            "semantic_eval.py"),
        "--log-root", str(out_root / "semantic_logs"),
        "--out", str(te_root / f"semantic_{args.tag}.csv"),
        "--out-tracklets", str(te_root / f"tracklets_{args.tag}.csv"),
    ], env=env, check=True)
    print("HELDOUT_EVAL_DONE", args.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--export", type=Path, default=DEFAULTS["export"])
    ap.add_argument("--feat", type=Path, default=DEFAULTS["feat"])
    ap.add_argument("--out", type=Path, default=DEFAULTS["out"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--decision-threshold", type=float, default=None)
    ap.add_argument("--commit-mode", choices=["M0", "M1"], default=None)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = dict(CONFIGS.get(args.tag, CONFIGS["j1b"]))
    if args.decision_threshold is not None:
        cfg["decision_threshold"] = args.decision_threshold
    if args.commit_mode is not None:
        cfg["commit_mode"] = args.commit_mode
    if args.eval:
        evaluate(args)
    else:
        replay(args, cfg)


if __name__ == "__main__":
    main()
