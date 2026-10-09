"""Small runtime compatibility shim for PANDAS on torchvision 0.21.

PANDAS was written against a torchvision version whose Faster R-CNN factory
did not default to downloading ImageNet/Faster R-CNN weights.  The current
factory does, so make the old explicit-random-initialization behavior
unambiguous while leaving the official PANDAS source untouched.
"""

from __future__ import annotations

import torchvision
import torch


_original_fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn


def fasterrcnn_resnet50_fpn_without_default_weights(*args, **kwargs):
    kwargs.setdefault("weights", None)
    kwargs.setdefault("weights_backbone", None)
    return _original_fasterrcnn(*args, **kwargs)


torchvision.models.detection.fasterrcnn_resnet50_fpn = (
    fasterrcnn_resnet50_fpn_without_default_weights
)


# PyTorch 2.6 changed torch.load's default to weights_only=True. Official
# PANDAS checkpoints include argparse.Namespace in their metadata, so the
# legacy checkpoint format needs this process-local compatibility default.
if not getattr(torch, "_codex_pandas_load_compat", False):
    _original_torch_load = torch.load

    def torch_load_pandas_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_torch_load(*args, **kwargs)

    torch.load = torch_load_pandas_compat
    torch._codex_pandas_load_compat = True


def patch_pandas_roi_heads() -> None:
    """Handle the official PANDAS ``objs=None`` default on torchvision 0.21."""
    from model_components.rcnn_components import RoIHeadsModified

    if getattr(RoIHeadsModified, "_codex_objs_compat", False):
        return
    original_forward = RoIHeadsModified.forward

    def forward_with_empty_objs(self, features, proposals, image_shapes, targets=None, objs=None):
        if objs is None:
            objs = []
        return original_forward(self, features, proposals, image_shapes, targets, objs)

    RoIHeadsModified.forward = forward_with_empty_objs
    RoIHeadsModified._codex_objs_compat = True
