"""Phase 4K provenance replays on the 20-video development subset.

Tags (frozen Phase 4J configurations):
  j0  : tau=0.50, M0 immediate memory            (= J0 / B2)
  j1b : tau=0.30, M0 immediate memory            (= J1b)
  m1  : tau=0.30, M1 age>=2 support>=2           (= J2b)

The semantic manager is shared across videos in the exact Phase 4I order
so memory states reproduce the Phase 4J runs.  Predictions are written to
a scratch dir and byte-compared with the frozen Phase 4J outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.replay import (
    frames_from_pre_assoc,
    replay_video,
)

EXPORT = ROOT / "outputs" / "iclr27_phase3a" / "smoke"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "detection_features"

CONFIGS = {
    "j0": {"decision_threshold": 0.5, "commit_mode": "M0",
           "commit_min_age": 2, "commit_min_support": 2,
           "ref": ROOT / "outputs" / "iclr27_phase4j" / "subset" / "J0"},
    "j1b": {"decision_threshold": 0.30, "commit_mode": "M0",
            "commit_min_age": 2, "commit_min_support": 2,
            "ref": ROOT / "outputs" / "iclr27_phase4j" / "subset" / "J1b"},
    "m1": {"decision_threshold": 0.30, "commit_mode": "M1",
           "commit_min_age": 2, "commit_min_support": 2,
           "ref": ROOT / "outputs" / "iclr27_phase4j" / "subset" / "J2b"},
}


def load_video_ids():
    return [int(p.stem) for p in
            (EXPORT / "pre_assoc_detections").glob("*.jsonl")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--videos", default="")
    args = ap.parse_args()
    cfg = CONFIGS[args.tag]
    device = torch.device("cuda")
    videos = [int(v) for v in args.videos.split(",") if v.strip()] or \
        load_video_ids()

    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    from src.frame_online_trackocd.semantic import build_semantic_manager
    from src.iclr27_phase4k.provenance import (
        AssociationInterventionLogger,
        ProvenanceLogger,
    )
    model, _ = load_mdc_model(str(ROOT / "runs/orbit_mdc/mdc_m2/model.pth"),
                              device)
    model.eval()

    out_root = ROOT / "outputs" / "iclr27_phase4k" / "audit" / \
        f"prov_{args.tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    prov = ProvenanceLogger(out_root, args.tag)
    assoc = AssociationInterventionLogger(out_root, args.tag)
    sem = build_semantic_manager(
        model, device, prefix_mode="P1",
        decision_threshold=cfg["decision_threshold"],
        commit_mode=cfg["commit_mode"],
        commit_min_age=cfg["commit_min_age"],
        commit_min_support=cfg["commit_min_support"],
        provenance=prov)

    pred_root = out_root / "preds"
    pred_root.mkdir(parents=True, exist_ok=True)
    log_root = out_root / "semantic_logs"
    log_root.mkdir(parents=True, exist_ok=True)
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
        print("prov", args.tag, vid, "frames", len(frames), flush=True)
    prov.flush()
    assoc.flush()

    # byte equivalence vs frozen Phase 4J predictions
    ref = cfg["ref"]
    bad, missing = [], []
    for p in sorted(pred_root.glob("*.json")):
        rp = ref / p.name
        if not rp.exists():
            missing.append(p.name)
            continue
        if json.loads(p.read_text()) != json.loads(rp.read_text()):
            bad.append(p.name)
    extra = sorted({p.name for p in ref.glob("*.json")} -
                   {p.name for p in pred_root.glob("*.json")})
    eq = {"tag": args.tag, "images": len(list(pred_root.glob("*.json"))),
          "byte_exact": not bad and not missing and not extra,
          "diff": bad[:10], "missing": missing[:10], "extra": extra[:10]}
    (out_root / "equivalence.json").write_text(json.dumps(eq, indent=1))
    print(json.dumps(eq, indent=1))
    print("PHASE4K_PROVENANCE_DONE", args.tag)


if __name__ == "__main__":
    main()
