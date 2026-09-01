#!/usr/bin/env python3
"""Reserved comparison between Meta official DINOv3 weights and the timm
converted distribution. Only runs when official weights are present;
otherwise it reports that no equivalence is claimed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT))

META_DIR = ROOT / "checkpoints" / "dinov3" / "meta_official"


def main():
    meta_files = sorted(META_DIR.glob("*.pth")) if META_DIR.exists() else []
    if not meta_files:
        print("meta_official weights not present; no equivalence claim.")
        return 0
    import timm
    tf = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.new("RGB", (256, 256))
    x = torch.stack([tf(img) for _ in range(4)])
    m = timm.create_model("hf_hub:timm/vit_base_patch16_dinov3.lvd1689m",
                          pretrained=True, num_classes=0).eval()
    with torch.no_grad():
        timm_feat = torch.nn.functional.normalize(m.forward_features(x)[:, 0], dim=-1)
    print("meta files:", [p.name for p in meta_files])
    print("embedding cosine to timm: run after loading official state dict;",
          "currently only shape recorded:", tuple(timm_feat.shape))
    return 0


if __name__ == "__main__":
    sys.exit(main())
