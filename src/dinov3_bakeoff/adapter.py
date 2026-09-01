from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

MODEL_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
WEIGHT_SOURCE = "timm converted distribution (W4)"
WEIGHT_SHA256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"
HF_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 256


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DinoV3Adapter:
    """Frozen DINOv3 ViT-B/16 LVD-1689M adapter (timm W4)."""

    def __init__(self, device="cuda", feature_mode="cls"):
        import timm
        self.model = timm.create_model(
            f"hf_hub:{MODEL_ID}", pretrained=True, num_classes=0
        )
        self.model.eval().to(device)
        self.device = device
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.feature_dim = self.model.num_features
        self.weight_sha256 = WEIGHT_SHA256
        self.model_id = MODEL_ID
        self.weight_source = WEIGHT_SOURCE
        self.processor_revision = HF_REVISION
        self.feature_mode = feature_mode

    @torch.no_grad()
    def embed_crops(self, tensors):
        """tensors: list of normalized [3,256,256] tensors; returns L2 rows.
        feature_mode='cls' uses the official CLS token (official linear eval);
        'pooled' uses timm's global avg-pooled output (supplementary)."""
        x = torch.stack(tensors).to(self.device)
        if self.feature_mode == "cls":
            feats = self.model.forward_features(x)[:, 0]
        else:
            feats = self.model(x)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)

    def cache_meta(self):
        return {
            "model_id": self.model_id,
            "weight_source": WEIGHT_SOURCE,
            "weight_sha256": self.weight_sha256,
            "processor_revision": self.processor_revision,
            "feature_dim": self.feature_dim,
            "n_params": self.n_params,
            "feature_mode": self.feature_mode,
        }


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_track_cache(adapter, row, frames, embeds, cache_dir, mode):
    sample_id = row["sample_id"]
    out_path = cache_dir / f"{sample_id}.json"
    if out_path.exists():
        return False
    embeds = np.asarray(embeds, dtype=np.float32)
    single = embeds[len(embeds) // 2] if len(embeds) else None
    mean = embeds.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    result = {
        "sample_id": sample_id,
        "model_id": adapter.model_id,
        "weight_source": adapter.weight_source,
        "weight_sha256": adapter.weight_sha256,
        "processor_revision": adapter.processor_revision,
        "selected_frame_ids": frames,
        "frame_embeddings": embeds.astype(np.float16).tolist(),
        "single_embedding": (
            single.astype(np.float16).tolist() if single is not None else None
        ),
        "mean_embedding": mean.astype(np.float16).tolist(),
        "num_valid_frames": len(embeds),
    }
    atomic_write_text(out_path, json.dumps(result, separators=(",", ":")))
    return True
