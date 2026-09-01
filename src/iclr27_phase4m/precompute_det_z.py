"""Precompute single-frame M2 embeddings for every detection.

Phase 4M needs the exact single-frame z used by the online semantic
manager for every detection so that per-track prefix embeddings can be
reconstructed offline from the causal association history (TrackSemState
keeps the last 8 single-frame z's; prefix P1 is their mean).

The M2 model and the feature root are the same ones used by Phase 4L, so
the cache is shared across tags (j1b/b1/b2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
MODEL_PTH = ROOT / "runs/orbit_mdc/mdc_m2/model.pth"
FEAT_ROOT = ROOT / "outputs/iclr27_phase4i/audit/detection_features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids", default="")
    ap.add_argument("--feat-root", type=Path, default=FEAT_ROOT)
    ap.add_argument("--out", type=Path,
                    default=ROOT /
                    "outputs/iclr27_phase4m/audit/det_z_cache")
    args = ap.parse_args()
    vids = [int(v) for v in args.video_ids.split(",") if v.strip()]
    if not vids:
        vids = sorted(int(p.stem) for p in args.feat_root.iterdir()
                      if p.is_dir())
    args.out.mkdir(parents=True, exist_ok=True)
    from src.orbit_mdc.evaluate_mdc import load_mdc_model
    device = torch.device("cuda:0")
    model, _ = load_mdc_model(str(MODEL_PTH), device)
    model.eval()
    for vid in vids:
        dst = args.out / f"{vid}.npz"
        if dst.exists():
            print("skip cached", vid)
            continue
        z = np.load(args.feat_root / str(vid) / "feats.npz")
        feats = z["feats"].astype(np.float32)
        ids = z["det_local_ids"].tolist()
        frames = z["frame_orders"].tolist()
        out = np.zeros((len(feats), 768), dtype=np.float16)
        for s in range(0, len(feats), 512):
            x = torch.as_tensor(feats[s:s + 512], dtype=torch.float32,
                                device=device).view(-1, 1, 768)
            mask = torch.ones(x.shape[0], 1, dtype=torch.bool, device=device)
            with torch.no_grad():
                o = model.aggregate(x, mask)
            zz = o["z"]
            if zz.dim() == 3:
                zz = zz[:, 0]
            out[s:s + len(zz)] = zz.float().cpu().numpy().astype(np.float16)
        np.savez(dst, det_local_ids=np.asarray(ids, dtype=np.int64),
                 frame_orders=np.asarray(frames, dtype=np.int64),
                 z=out)
        print("done", vid, len(feats), flush=True)
    print("DET_Z_CACHE_DONE", len(vids))


if __name__ == "__main__":
    main()
