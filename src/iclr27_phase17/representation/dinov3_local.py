"""Offline local DINOv3 ViT-B/16 adapter used by the Phase 17 audit.

The available AVI environment has timm 1.0.15, while the converted DINOv3
checkpoint was produced for a newer timm model entry.  We construct the
equivalent EVA-style ViT explicitly and bind a small positional-token shim so
the local checkpoint can be loaded without network access.  The shim is
tested by ``validate_local_dinov3.py`` and its checkpoint hash is recorded in
the official-model audit; the historical adapter is not modified.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType
from typing import Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
MODEL_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
WEIGHTS = ROOT / "data/iclr27_phase17/checkpoints/dinov3_model.safetensors"
WEIGHT_SHA256 = "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332"
DIM = 768


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.resolve().open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fixed_pos_embed(self, x):
    """timm-1.0.15 token assembly compatible with the converted DINOv3 W4."""
    b = x.shape[0]
    if x.ndim == 4:
        x = x.reshape(b, -1, x.shape[-1])
    cls = self.cls_token.expand(b, -1, -1) if self.cls_token is not None else x[:, :0]
    reg = self.reg_token.expand(b, -1, -1) if self.reg_token is not None else x[:, :0]
    x = torch.cat((cls, reg, x), dim=1)
    return x, self.rope.get_embed() if self.rope is not None else None


class LocalDinoV3:
    def __init__(self, device: str = "cuda:0", feature_mode: str = "dense"):
        import timm
        from safetensors.torch import load_file

        if sha256_file(WEIGHTS) != WEIGHT_SHA256:
            raise RuntimeError("DINOv3 checkpoint hash mismatch")
        kwargs = dict(img_size=256, patch_size=16, embed_dim=768, depth=12,
                      num_heads=12, qkv_fused=True, mlp_ratio=4,
                      swiglu_mlp=False, scale_mlp=False, scale_attn_inner=False,
                      use_rot_pos_emb=True, ref_feat_shape=(16, 16),
                      use_abs_pos_emb=False, global_pool="token", num_reg_tokens=4,
                      init_values=1e-5, num_classes=0, qkv_bias=False)
        model = timm.create_model("eva02_base_patch16_clip_224", pretrained=False, **kwargs)
        missing, unexpected = model.load_state_dict(load_file(str(WEIGHTS)), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"unexpected local DINOv3 state: missing={missing}, unexpected={unexpected}")
        model._pos_embed = MethodType(_fixed_pos_embed, model)
        self.model = model.eval().to(device)
        self.device = device
        self.feature_mode = feature_mode
        self.model_id = MODEL_ID
        self.weight_sha256 = WEIGHT_SHA256
        self.feature_dim = DIM
        self.patch_size = 16
        self.num_reg_tokens = 4

    @torch.no_grad()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized CLS or mean dense patch features."""
        x = batch.to(self.device, non_blocking=True)
        tok = self.model.forward_features(x)
        if self.feature_mode == "cls":
            f = tok[:, 0]
        elif self.feature_mode == "dense":
            f = tok[:, 1 + self.num_reg_tokens:].mean(dim=1)
        else:
            f = self.model(x)
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    def embed_crops(self, tensors: Iterable[torch.Tensor]) -> np.ndarray:
        return self.embed(torch.stack(list(tensors))).detach().cpu().numpy().astype(np.float32)

    def metadata(self) -> dict:
        return {"model_id": self.model_id, "weight_source": "timm converted W4 local",
                "checkpoint": str(WEIGHTS.resolve()), "weight_sha256": self.weight_sha256,
                "feature_mode": self.feature_mode, "feature_dim": self.feature_dim,
                "patch_size": self.patch_size, "num_reg_tokens": self.num_reg_tokens,
                "positional_compatibility_shim": "timm1.0.15 EVA token assembly; no parameter changes",
                "future_frames_used": False, "physical_id_used_as_feature": False,
                "gt_boxes_as_deployed_input": False}
