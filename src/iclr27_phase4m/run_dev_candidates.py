"""Phase 4M development-subset replays.

Tags:
  m0  corrected J1b anchor (deferral off; reproducibility check)
  m1  margin-based causal semantic deferral (margin < 0.05)
  m2  entropy-based causal semantic deferral (entropy > 1.6)
  m3  audit-supported 3-feature ambiguity deferral (frozen logistic,
      fit on j1b identity decisions only)

Everything else (detector, DINO, M2, tau=.30, lambda_s=.1) is frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / \
    "detection_features"
OUT_ROOT = ROOT / "outputs" / "iclr27_phase4m" / "dev"
PROV_ROOT = ROOT / "outputs" / "iclr27_phase4m" / "audit"

CFGS = {
    "m0": {"deferral_mode": "none"},
    "m1": {"deferral_mode": "margin", "defer_margin": 0.05},
    "m2": {"deferral_mode": "entropy", "defer_entropy": 1.6},
    "m3": {
        "deferral_mode": "ambiguity",
        "defer_ambiguity_coef": [-3.0754, -1.2081, -0.5705],
        "defer_ambiguity_intercept": 1.7935,
        "defer_ambiguity_threshold": 0.6097,
    },
}


def load_video_order():
    log = ROOT / "runs/iclr27_phase4k/prov_j1b.log"
    order = []
    if log.exists():
        for line in log.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "prov" and \
                    parts[1] == "j1b":
                order.append(int(parts[2]))
    if not order:
        order = sorted(int(p.stem) for p in
                       (EXPORT / "pre_assoc_detections").glob("*.jsonl"))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=list(CFGS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--videos", default="")
    args = ap.parse_args()
    tag = args.tag
    cfg = CFGS[tag]
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    from src.iclr27_phase4k.provenance import (
        AssociationInterventionLogger,
        ProvenanceLogger,
    )
    from src.frame_online_trackocd.replay import (
        frames_from_pre_assoc,
        replay_video,
    )
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = out_dir / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root = out_dir / "semantic_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    prov_dir = PROV_ROOT / f"prov_dev_{tag}"
    prov_dir.mkdir(parents=True, exist_ok=True)
    prov = ProvenanceLogger(prov_dir, tag)
    assoc = AssociationInterventionLogger(prov_dir, tag)
    sem = build_semantic_manager(
        model, device, prefix_mode="P1", decision_threshold=0.30,
        commit_mode="M0", commit_min_age=2, commit_min_support=2,
        provenance=prov, matching_mode="absolute", **cfg)
    videos = [int(v) for v in args.videos.split(",") if v.strip()] or \
        load_video_order()
    for vid in videos:
        frames = frames_from_pre_assoc(EXPORT, vid)
        log_file = open(log_root / f"{vid}.jsonl", "w")
        rows = replay_video(vid, frames, EXPORT, FEAT_ROOT, pred_root,
                            sem_manager=sem, mode="B2", lambda_s=0.1,
                            device=device, log_writer=log_file,
                            association_logger=assoc)
        for r in rows:
            log_file.write(json.dumps(r) + "\n")
        log_file.close()
        print("dev", tag, vid, "frames", len(frames),
              "protos", sem.memory.size(),
              "unresolved", sem.branch_deferred, flush=True)
    prov.flush()
    assoc.flush()
    print("matching stats: branch sticky/soft/new/defer",
          sem.branch_sticky, sem.branch_soft, sem.branch_new,
          sem.branch_deferred, flush=True)
    link = prov_dir / "semantic_logs"
    if not link.exists():
        link.symlink_to(log_root)
    (out_dir / "config.json").write_text(json.dumps(
        {"tag": tag, **cfg}, indent=1))
    print("PHASE4M_DEV_CANDIDATE_DONE", tag,
          "protos", sem.memory.size(), "deferred", sem.branch_deferred)


if __name__ == "__main__":
    main()
