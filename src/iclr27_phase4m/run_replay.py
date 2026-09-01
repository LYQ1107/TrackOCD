"""Phase 4M replay + evaluation driver (dev / held-out).

Writes exclusively under outputs/iclr27_phase4m so Phase 4L artifacts are
never overwritten.  Baseline tag j1b reproduces the corrected anchor;
deferral tags are added with explicit semantic configuration.
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

MODES = {
    "dev": {
        "export": ROOT / "outputs" / "iclr27_phase3a" / "smoke",
        "feat": ROOT / "outputs" / "iclr27_phase4i" / "audit" /
        "detection_features",
        "out": ROOT / "outputs" / "iclr27_phase4m" / "runs" / "dev",
        "tao": ROOT / "outputs" / "iclr27_phase3a" / "smoke" /
        "tao_subset" / "validation_20.json",
        "trackeval_gt": None,
    },
    "heldout": {
        "export": ROOT / "outputs" / "iclr27_phase4l" / "heldout_export",
        "feat": ROOT / "outputs" / "iclr27_phase4l" / "heldout_features",
        "out": ROOT / "outputs" / "iclr27_phase4m" / "runs" / "heldout",
        "tao": ROOT / "outputs" / "iclr27_phase4l" / "heldout" /
        "validation_heldout_tao.json",
        "trackeval_gt": ROOT / "outputs" / "iclr27_phase4l" / "heldout" /
        "trackeval" / "gt",
    },
}


def load_video_ids(export):
    return [int(p.stem) for p in
            (export / "pre_assoc_detections").glob("*.jsonl")]


def load_dev_video_order():
    """Phase 4L dev order: global novel memory makes order matter."""
    log = ROOT / "runs" / "iclr27_phase4k" / "prov_j1b.log"
    order = []
    if log.exists():
        for line in log.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "prov" and \
                    parts[1] == "j1b":
                order.append(int(parts[2]))
    return order


def replay(args, mode_cfg, cfg):
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
    prov_root = ROOT / "outputs" / "iclr27_phase4m" / "prov" / \
        f"{args.mode}_{args.tag}"
    prov_root.mkdir(parents=True, exist_ok=True)
    prov = ProvenanceLogger(prov_root, args.tag)
    assoc = AssociationInterventionLogger(prov_root, args.tag)
    sem = build_semantic_manager(model, device, prefix_mode="P1",
                                 decision_threshold=cfg["tau"],
                                 commit_mode=cfg.get("commit_mode", "M0"),
                                 commit_min_age=2, commit_min_support=2,
                                 provenance=prov,
                                 matching_mode=cfg.get("matching_mode",
                                                       "absolute"),
                                 margin_threshold=cfg.get(
                                     "margin_threshold", 0.05),
                                 entropy_threshold=cfg.get(
                                     "entropy_threshold", 1.6),
                                 deferral_mode=cfg.get("deferral_mode",
                                                       "none"),
                                 defer_margin=cfg.get("defer_margin", 0.10),
                                 defer_entropy=cfg.get("defer_entropy", 1.6),
                                 defer_nk=cfg.get("defer_nk", 0.25),
                                 validity_mode=cfg.get("validity_mode",
                                                       "none"),
                                 validity_config=cfg.get(
                                     "validity_config"),
                                 validity_threshold=cfg.get(
                                     "validity_threshold", 0.03))
    out_root = mode_cfg["out"] / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    pred_root = out_root / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "semantic_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    if args.videos:
        videos = [int(v) for v in args.videos.split(",") if v.strip()]
    elif args.mode == "dev":
        videos = load_dev_video_order() or \
            sorted(load_video_ids(mode_cfg["export"]))
    else:
        videos = load_video_ids(mode_cfg["export"])
    for vid in sorted(videos):
        frames = frames_from_pre_assoc(mode_cfg["export"], vid)
        log_file = open(log_root / f"{vid}.jsonl", "w")
        rows = replay_video(vid, frames, mode_cfg["export"],
                            mode_cfg["feat"], pred_root,
                            sem_manager=sem, mode="B2", lambda_s=0.1,
                            device=device, log_writer=log_file,
                            association_logger=assoc)
        for r in rows:
            log_file.write(json.dumps(r) + "\n")
        log_file.close()
        print(args.mode, args.tag, vid, "frames", len(frames), flush=True)
    prov.flush()
    assoc.flush()
    link = prov_root / "semantic_logs"
    if not link.exists():
        link.symlink_to(out_root / "semantic_logs")
    cfg["tag"] = args.tag
    cfg["mode"] = args.mode
    (out_root / "config.json").write_text(json.dumps(cfg, indent=1))
    print("PHASE4M_REPLAY_DONE", args.mode, args.tag)


def evaluate(args, mode_cfg):
    out_root = mode_cfg["out"] / args.tag
    pred_root = out_root / "preds"
    te_root = mode_cfg["out"] / "trackeval"
    te_root.mkdir(parents=True, exist_ok=True)
    trackers_root = te_root / "trackers"
    trackers_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if mode_cfg["trackeval_gt"] is not None:
        env["PHASE4L_TRACKEVAL_GT"] = str(mode_cfg["trackeval_gt"])
    env["PHASE4L_TAO_JSON"] = str(mode_cfg["tao"])
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4i" /
                            "build_trackeval_input.py"),
        "--input-dir", str(pred_root), "--tracker-name", args.tag,
        "--output-root", str(trackers_root),
    ], check=True)
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4i" /
                            "run_trackeval_subset.py"),
        "--trackers-folder", str(trackers_root),
        "--names", args.tag,
        "--out", str(te_root / f"tracking_{args.tag}.json"),
    ], env=env, check=True)
    subprocess.run([
        sys.executable, str(ROOT / "src/iclr27_phase4j" /
                            "semantic_eval.py"),
        "--log-root", str(out_root / "semantic_logs"),
        "--out", str(te_root / f"semantic_{args.tag}.csv"),
        "--out-tracklets", str(te_root / f"tracklets_{args.tag}.csv"),
    ], env=env, check=True)
    print("PHASE4M_EVAL_DONE", args.mode, args.tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--videos", default="")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--commit-mode", default="M0")
    ap.add_argument("--matching-mode", default="absolute")
    ap.add_argument("--margin-threshold", type=float, default=0.05)
    ap.add_argument("--entropy-threshold", type=float, default=1.6)
    ap.add_argument("--deferral-mode", default="none",
                    choices=["none", "margin", "entropy", "hybrid"])
    ap.add_argument("--defer-margin", type=float, default=0.10)
    ap.add_argument("--defer-entropy", type=float, default=1.6)
    ap.add_argument("--defer-nk", type=float, default=0.25)
    ap.add_argument("--validity-mode", default="none",
                    choices=["none", "logistic"])
    ap.add_argument("--validity-threshold", type=float, default=0.01)
    ap.add_argument("--validity-config", type=Path, default=ROOT /
                    "configs" / "frontend_validity" /
                    "validity_logistic.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    mode_cfg = MODES[args.mode]
    if args.videos:
        mode_cfg = dict(mode_cfg)
        mode_cfg["out"] = mode_cfg["out"] / "smoke"
    cfg = {
        "tau": args.tau,
        "commit_mode": args.commit_mode,
        "matching_mode": args.matching_mode,
        "margin_threshold": args.margin_threshold,
        "entropy_threshold": args.entropy_threshold,
        "deferral_mode": args.deferral_mode,
        "defer_margin": args.defer_margin,
        "defer_entropy": args.defer_entropy,
        "defer_nk": args.defer_nk,
        "validity_mode": args.validity_mode,
        "validity_threshold": args.validity_threshold,
        "validity_config": (json.loads(args.validity_config.read_text())
                            if args.validity_mode != "none" and
                            args.validity_config.exists() else None),
    }
    replay(args, mode_cfg, cfg)
    if args.eval:
        evaluate(args, mode_cfg)


if __name__ == "__main__":
    main()
