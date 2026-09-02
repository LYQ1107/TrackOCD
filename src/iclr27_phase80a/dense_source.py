"""DINOv3 dense crop evidence used by the Phase80A diagnostic.

The implementation is independent of the frozen Phase15S/76 relation code.
It exposes the CLS token and a deterministic spatial subset of patch tokens,
while keeping labels and identifiers outside the model input.  The local
weight is the timm distribution already present in the environment; it is not
claimed to be byte-identical to Meta's official checkpoint.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.iclr27_phase75d.protocol import CSV_PATH


def row_key(row: dict[str, str]) -> str:
    """Canonical five-field key shared with the frozen Phase23 protocol."""
    value = str(row.get("row_key", ""))
    if value:
        return value
    return ":".join(str(row.get(k, "")) for k in ("video_id", "frame_id", "proposal_local_id", "track_id", "image_id"))


MODEL_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
WEIGHT_SOURCE = "TIMM_DISTRIBUTION"
WEIGHT_SHA256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"
PROCESSOR_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
INPUT_SIZE = 256
CONTEXT = 0.10
TOKENS_PER_FRAME = 32


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def selected_patch_indices() -> np.ndarray:
    """Return a fixed 4x8 grid from the 16x16 patch map (row-major)."""
    rows = np.asarray([1, 5, 9, 13], dtype=np.int64)
    cols = np.asarray([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64)
    return (rows[:, None] * 16 + cols[None, :]).reshape(-1)


PATCH_INDICES = selected_patch_indices()


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def crop_box(image: Image.Image, box: Iterable[float], context: float = CONTEXT) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    nw, nh = bw * (1.0 + 2.0 * context), bh * (1.0 + 2.0 * context)
    xa, ya = max(0.0, cx - nw * 0.5), max(0.0, cy - nh * 0.5)
    xb, yb = min(float(width), cx + nw * 0.5), min(float(height), cy + nh * 0.5)
    if xb - xa < 2 or yb - ya < 2:
        xa, ya, xb, yb = max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)
    return image.crop((int(xa), int(ya), int(xb), int(yb)))


class DinoV3Dense:
    """Frozen timm DINOv3 ViT-B/16 returning CLS and selected patch tokens."""

    def __init__(self, device: str = "cuda:2") -> None:
        import timm

        self.device = torch.device(device)
        self.model = timm.create_model(
            f"hf_hub:{MODEL_ID}", pretrained=True, num_classes=0
        ).eval().to(self.device)
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.feature_dim = int(self.model.num_features)
        self.patch_indices = torch.as_tensor(PATCH_INDICES, device=self.device)

    @torch.inference_mode()
    def encode(self, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.model.forward_features(batch.to(self.device, non_blocking=True))
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise RuntimeError(f"unexpected DINOv3 output: {type(tokens)} {getattr(tokens, 'shape', None)}")
        if tokens.shape[-1] != self.feature_dim or tokens.shape[1] < int(PATCH_INDICES.max()) + 6:
            raise RuntimeError(f"unexpected token shape {tuple(tokens.shape)}")
        # timm Eva exposes [CLS, 4 register, 16x16 patches] for this model.
        cls = torch.nn.functional.normalize(tokens[:, 0].float(), dim=-1)
        patches = tokens[:, 5:][:, self.patch_indices]
        patches = torch.nn.functional.normalize(patches.float(), dim=-1)
        return cls.cpu().numpy().astype(np.float32), patches.cpu().numpy().astype(np.float32)


def load_rows() -> list[dict[str, str]]:
    import csv

    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cache_meta(rows: list[dict[str, str]], device: str, shard: int, num_shards: int) -> dict[str, object]:
    return {
        "protocol": "phase80a_dense_visual_evidence_v1",
        "model_id": MODEL_ID,
        "weight_source": WEIGHT_SOURCE,
        "weight_sha256": WEIGHT_SHA256,
        "processor_revision": PROCESSOR_REVISION,
        "input_size": INPUT_SIZE,
        "context_padding": CONTEXT,
        "patch_grid": [16, 16],
        "register_tokens_excluded": 4,
        "tokens_per_frame": TOKENS_PER_FRAME,
        "patch_selection": "fixed visual-only 4x8 grid; rows=[1,5,9,13], cols=[0,2,4,6,8,10,12,14]",
        "rows": len(rows),
        "shard": shard,
        "num_shards": num_shards,
        "device": device,
        "future_frames_used": False,
        "category_text_used": False,
        "semantic_or_physical_id_input": False,
        "held_or_devplus_input": False,
    }
