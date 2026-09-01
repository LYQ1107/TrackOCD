from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(PROJECT_ROOT))

from src.trackocd_v1.trajectory_encoder import TrajectoryEncoder


def validate_checkpoint(path):
    """Validate the existing semantic transformer checkpoint (TrackOCD-v1
    bake-off configuration). Returns (model, ckpt) or raises."""
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["variant"] == "transformer", ckpt["variant"]
    model = TrajectoryEncoder(len(ckpt["classes"]), variant="transformer")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, ckpt, n_params


if __name__ == "__main__":
    model, ckpt, n = validate_checkpoint(
        PROJECT_ROOT / "runs/trackocd_v1/traj_enc_transformer/model.pth"
    )
    print("checkpoint OK", "params", n, "classes", len(ckpt["classes"]))
