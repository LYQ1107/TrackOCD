from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DualBranchConfig:
    project_root: Path = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
    transformer_ckpt: Path = Path(
        "runs/trackocd_v1/traj_enc_transformer/model.pth"
    )
    semantic_variant: str = "transformer"
    b2_threshold_d0: float = 0.45
    b2_threshold_d1: float = 0.45  # overwritten by proxy calibration
    discovery_dim: int = 768
    semantic_dim: int = 256
    max_frames: int = 8
    streams: tuple = ("main", "main_seed1027", "main_seed1028", "main_seed1029")
    subsets: tuple = ("full", "repeated", "balanced")
    protocols: tuple = ("pure", "ov_assisted")
    out_dir: Path = Path("outputs/dual_branch/metrics")
    runs_dir: Path = Path("runs/dual_branch")

    def resolve(self, p: Path) -> Path:
        return self.project_root / p
