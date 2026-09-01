"""Phase 4M held-out replay + evaluation (24 videos, one-shot).

Tags:
  m0  corrected J1b anchor on held-out (reproducibility)
  m1  frozen Candidate A: margin-based causal semantic deferral

Everything is frozen before this run; no tuning after seeing held-out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from src.frame_online_trackocd.replay import (
    frames_from_pre_assoc,
    replay_video,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
HELDOUT = ROOT / "outputs" / "iclr27_phase4l" / "heldout"
HELDOUT_EXPORT = ROOT / "outputs" / "iclr27_phase4l" / "heldout_export"
HELDOUT_FEAT = ROOT / "outputs" / "iclr27_phase4l" / "heldout_features"
OUT_ROOT = ROOT / "outputs" / "iclr27_phase4m" / "heldout"
PROV_ROOT = ROOT / "outputs" / "iclr27_phase4m" / "audit"

CONFIGS = {
    "m0": {"deferral_mode": "none"},
    "m1": {"deferral_mode": "margin", "defer_margin": 0.05},
}


def load_video_ids(export):
    return sorted(int(p.stem) for p in
                  (export / "pre_assoc_detections").glob("*.jsonl"))


def replay(args, cfg):
    device = torch.device("cuda")
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    from src.iclr27_phase4k.provenance import (
        AssociationInterventionLogger,
        ProvenanceLogger,
    )
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    out_root = OUT_ROOT / "runs" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    pred_root = out_root / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "semantic_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    prov_dir = PROV_ROOT / f"prov_ho_{args.tag}"
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov = ProvenanceLogger(prov_dir, args.tag)
    assoc = AssociationInterventionLogger(prov_dir, args.tag)
    sem = build_semantic_manager(
        model, device, prefix_mode="P1", decision_threshold=0.30,
        commit_mode="M0", commit_min_age=2, commit_min_support=2,
        provenance=prov, matching_mode="absolute", **cfg)
    for vid in load_video_ids(args.export):
        frames = frames_from_pre_assoc(args.export, vid)
        log_file = open(log_root / f"{vid}.jsonl", "w")
        rows = replay_video(vid, frames, args.export, args.feat, pred_root,
                            sem_manager=sem, mode="B2", lambda_s=0.1,
                            device=device, log_writer=log_file,
                            association_logger=assoc)
        for r in rows:
            log_file.write(json.dumps(r) + "\n")
        log_file.close()
        print("heldout", args.tag, vid, "frames", len(frames),
              "protos", sem.memory.size(), flush=True)
    prov.flush()
    assoc.flush()
    link = prov_dir / "semantic_logs"
    if not link.exists():
        link.symlink_to(log_root)
    cfg["tag"] = args.tag
    (out_root / "config.json").write_text(json.dumps(cfg, indent=1))
    print("HELDOUT_REPLAY_DONE", args.tag, "protos", sem.memory.size(),
          "deferred", sem.branch_deferred)


def evaluate(args):
    out_root = OUT_ROOT / "runs" / args.tag
    te_root = OUT_ROOT / "trackeval"
    trackers_root = te_root / "trackers"
    trackers_root.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4i" /
                            "build_trackeval_input.py"),
        "--input-dir", str(out_root / "preds"),
        "--tracker-name", args.tag,
        "--output-root", str(trackers_root),
    ], check=True)
    env = dict(os.environ)
    env["PHASE4L_TRACKEVAL_GT"] = str(
        HELDOUT / "trackeval" / "gt")
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4i" /
                            "run_trackeval_subset.py"),
        "--trackers-folder", str(trackers_root),
        "--names", args.tag,
        "--out", str(te_root / f"tracking_{args.tag}.json"),
    ], env=env, check=True)
    env["PHASE4L_TAO_JSON"] = str(
        HELDOUT / "validation_heldout_tao.json")
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
    ap.add_argument("--tag", required=True, choices=list(CONFIGS))
    ap.add_argument("--export", type=Path, default=HELDOUT_EXPORT)
    ap.add_argument("--feat", type=Path, default=HELDOUT_FEAT)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.eval:
        evaluate(args)
    else:
        replay(args, dict(CONFIGS[args.tag]))


if __name__ == "__main__":
    main()
