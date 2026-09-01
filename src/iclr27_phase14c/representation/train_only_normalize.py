"""Single train-only normalization: re-normalize proposal rows to unit norm.

The public TRAIN cache supplies the reference norm statistic.  DINOv2 rows
are already unit-normalized; this procedure therefore has no hidden DEV+
scale/threshold selection but records the measured identity transform.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    z = np.load(ROOT / args.inp); x = z["feats"].astype(np.float32)
    tr = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz")["mean_feats"].astype(np.float32)
    train_norm = np.linalg.norm(tr, axis=1); ref = float(train_norm.mean())
    norms = np.linalg.norm(x, axis=1, keepdims=True); y = x / np.maximum(norms, 1e-12)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_suffix(out.suffix + ".tmp")
    np.savez_compressed(tmp, feats=y, row_keys=z["row_keys"]); generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp; os.replace(generated, out)
    meta = {"procedure": "unit_l2_renormalization_using_public_train_reference", "public_train_reference": "outputs/iclr27_phase6d/assets/full_tao_tracks.npz", "train_reference_norm_mean": ref, "proposal_norm_mean_before": float(norms.mean()), "proposal_norm_mean_after": float(np.linalg.norm(y, axis=1).mean()), "q1_label_used": False, "devplus_labels_used": False}
    mp = ROOT / args.meta; mp.parent.mkdir(parents=True, exist_ok=True); mt = mp.with_suffix(mp.suffix + ".tmp"); mt.write_text(json.dumps(meta, indent=2)); os.replace(mt, mp); print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
