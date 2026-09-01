"""CPU/GPU-free smoke validation for the Phase 17 local DINOv3 loader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.iclr27_phase17.representation.dinov3_local import LocalDinoV3


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--out", type=Path, default=Path("outputs/iclr27_phase17/manifests/dinov3_local_validation.json")); args = ap.parse_args()
    m = LocalDinoV3(args.device, "dense")
    with torch.no_grad():
        x = torch.randn(2, 3, 256, 256, device=args.device)
        y = m.embed(x)
    value = {**m.metadata(), "input_shape": [2, 3, 256, 256], "output_shape": list(y.shape),
             "finite": bool(torch.isfinite(y).all()), "norms": [float(v) for v in torch.linalg.norm(y, dim=1)]}
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = args.out.with_suffix(args.out.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(args.out); print(json.dumps(value, indent=2))


if __name__ == "__main__": main()
