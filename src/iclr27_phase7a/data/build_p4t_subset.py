"""Build a DINOv2-extractable subset of the Phase 4T TRAIN stream.

The original outputs/iclr27_phase4t/train_stream/feats.npz is NOT the DINOv2
bbox-crop embedding used by TSE (TSE max-known-sim ~0.08 vs Q1 ~0.64), so
Phase 7A re-extracts DINOv2 features for:
  - every supported-known row (gt_role == 'known', 9047 rows);
  - a deterministic sample of FP rows (default 30000) for memory
    contamination dynamics.
novel_role rows are excluded (no true-novel information may enter training).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
CSV = ROOT / "outputs/iclr27_phase7a/assets/p4t_stream_rebuilt.csv"
OUT = ROOT / "outputs/iclr27_phase7a/assets/p4t_dinov2_subset.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=1027)
    args = ap.parse_args()
    df = pd.read_csv(CSV)
    known = df[df["gt_role"] == "known"].copy()
    fp = df[df["gt_role"] == "fp"].copy()
    fp = fp.sample(n=min(args.fp_sample, len(fp)), random_state=args.seed)
    sub = pd.concat([known, fp], ignore_index=True)
    sub = sub.sort_values(["video_id", "frame_id", "image_id", "track_id"],
                          kind="stable")
    sub.to_csv(OUT, index=False)
    print(f"subset rows {len(sub)} (known {len(known)}, fp {len(fp)}), "
          f"written {OUT}")


if __name__ == "__main__":
    main()
