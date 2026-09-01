"""Phase 4L development-subset candidate replays.

Candidates:
  a1  detector-score admissibility (A1)
  a2  tracking-evidence admissibility (A2)
  a3  combined admissibility (A3)
  b1  relative matching: best + best-second margin
  b2  relative matching: best + top-5 density (entropy)
  ab  combined admissibility (A3) + margin matching (B1)

The 20-video dev subset is used only for mechanism development; final
candidates are frozen and then evaluated on the held-out subset.
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
OUT_ROOT = ROOT / "outputs" / "iclr27_phase4l" / "dev"
PROV_ROOT = ROOT / "outputs" / "iclr27_phase4l" / "audit"

from src.frame_online_trackocd.replay import (
    frames_from_pre_assoc,
    replay_video,
)


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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--videos", default="")
    args = ap.parse_args()
    tag = args.tag
    cfgs = {
        "j1b": {},
        "a1": {"admissibility_mode": "detector"},
        "a2": {"admissibility_mode": "tracking"},
        "a3": {"admissibility_mode": "combined"},
        "b1": {"matching_mode": "margin", "margin_threshold": 0.05},
        "b2": {"matching_mode": "entropy", "entropy_threshold": 1.6},
        "ab": {"admissibility_mode": "combined",
               "matching_mode": "margin", "margin_threshold": 0.05},
    }
    assert tag in cfgs, tag
    admiss_cfg = None
    if cfgs[tag].get("admissibility_mode") in ("detector", "tracking",
                                               "combined"):
        key = {"detector": "a1", "tracking": "a2", "combined": "a3"}[
            cfgs[tag]["admissibility_mode"]]
        admiss_cfg = json.loads(
            (ROOT / "configs/iclr27_phase4l/admissibility_logistic.json")
            .read_text())[key]

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
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_root = out_dir / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root = out_dir / "semantic_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    prov = ProvenanceLogger(PROV_ROOT / f"prov_dev_{tag}", tag)
    assoc = AssociationInterventionLogger(PROV_ROOT / f"prov_dev_{tag}", tag)
    sem = build_semantic_manager(
        model, device, prefix_mode="P1", decision_threshold=0.30,
        commit_mode="M0", commit_min_age=2, commit_min_support=2,
        provenance=prov,
        admissibility_mode=cfgs[tag].get("admissibility_mode", "none"),
        admissibility_config=admiss_cfg,
        matching_mode=cfgs[tag].get("matching_mode", "absolute"),
        margin_threshold=cfgs[tag].get("margin_threshold", 0.05),
        entropy_threshold=cfgs[tag].get("entropy_threshold", 1.6))
    import os
    sem._debug_branch = os.environ.get("PHASE4L_DEBUG_BRANCH") == "1"
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
        print("dev", tag, vid, "frames", len(frames), flush=True)
    prov.flush()
    assoc.flush()
    print("matching stats: calls", sem.memory.rel_calls,
          "rejects", sem.memory.rel_rejects,
          "branch sticky/soft/new", sem.branch_sticky,
          sem.branch_soft, sem.branch_new, flush=True)
    link = PROV_ROOT / f"prov_dev_{tag}" / "semantic_logs"
    if not link.exists():
        link.symlink_to(OUT_ROOT / tag / "semantic_logs")
    cfg = {"tag": tag, **cfgs[tag]}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print("PHASE4L_DEV_CANDIDATE_DONE", tag)


if __name__ == "__main__":
    main()
