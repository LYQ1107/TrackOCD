"""Freeze a CHP candidate with full config + SHA256 before any official run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--compat_threshold", type=float, default=0.45)
    ap.add_argument("--compat_margin", type=float, default=0.05)
    ap.add_argument("--episode_mode", default=None)
    ap.add_argument("--hardness_definition", default=None)
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    record = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "sha256": sha256(args.checkpoint),
        "gate_threshold": args.gate_threshold,
        "compat_threshold": args.compat_threshold,
        "compat_margin": args.compat_margin,
        "episode_mode": args.episode_mode or ck.get("episode_mode", "?"),
        "hardness_definition": args.hardness_definition or (
            "train-side adapted-space leave-one-out mean best-known cosine"),
        "variant": ck.get("variant", "?"),
        "seed": ck.get("seed", "?"),
        "compat_feats": ck.get("compat_feats", "?"),
        "train_adapter": ck.get("train_adapter", False),
        "init_checkpoint": ck.get("init_checkpoint", "?"),
        "class_split": "outputs/orbit/splits/meta_train_classes.csv + "
                       "meta_dev_classes.csv (frozen)",
        "protocol": "Pure Full, seed1027, strict online causal, GT-track",
    }
    out_dir = ROOT / "outputs" / "orbit_chp" / "frozen_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.json"
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
