"""Explicit frozen image-only MASA proposal route; not the public-Detic runner.

Upstream modules stay byte-identical in a private, manifest-verified source
directory. Only required image modules are registered; no dataset/text APIs,
checkpoint class-name fallback or offline future-dependent postprocessor.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/trackocd_core/audit/masa_candidate_source"


def literal_config(node: ast.AST, symbols: dict):
    """Decode only literals/dict constructors/allowlisted symbols, never eval."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in symbols:
        return copy.deepcopy(symbols[node.id])
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [literal_config(v, symbols) for v in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        return {literal_config(k, symbols): literal_config(v, symbols) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = literal_config(node.operand, symbols)
        if type(value) not in (int, float):
            raise ValueError("Expected numeric config negation")
        return -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = literal_config(node.left, symbols), literal_config(node.right, symbols)
        if type(left) not in (int, float) or type(right) not in (int, float) or not right:
            raise ValueError("Expected finite numeric literal division with nonzero divisor")
        import math
        value = left / right
        if not math.isfinite(value):
            raise ValueError("Non-finite config arithmetic")
        return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        if any(k.arg is None for k in node.keywords):
            raise ValueError("No config dictionary expansion")
        return {k.arg: literal_config(k.value, symbols) for k in node.keywords}
    raise ValueError("Unsupported config expression: " + type(node).__name__)


def assignment(source: str, name: str, symbols: dict):
    node, = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    return literal_config(node, symbols)


def verified_sources() -> list[dict]:
    manifest = json.loads((ROOT / "configs/trackocd_core/masa_native_source_manifest.json").read_text())
    records = []
    for record in manifest["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["bytes"] > 32768:
            raise ValueError("Unsafe source allowlist")
        path = SOURCE / relative
        if path.is_symlink() or not path.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError("Unexpected source link")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if len(data) != record["bytes"] or blob != record["git_blob_sha1"]:
            raise ValueError("Pinned source identity mismatch: " + record["path"])
        records.append({**record, "sha256": hashlib.sha256(data).hexdigest()})
    return records


def native_config() -> dict:
    sam_source = (SOURCE / "configs/masa-sam/sam-vitb.py").read_text()
    dim = assignment(sam_source, "prompt_embed_dim", {})
    detector = assignment(sam_source, "model", {"prompt_embed_dim": dim})
    source = (SOURCE / "configs/masa-sam/open_vocabulary_mot_test/masa_sam_vitb_open_vocabulary_test.py").read_text()
    config = assignment(source, "model", {"detector": detector})
    # No execution of _base_, dataset declarations, init_cfg checkpoints or APIs.
    config.update(load_public_dets=False, public_det_path=None, given_dets=False,
                  freeze_detector=True, freeze_masa_adapter=True, freeze_object_prior_distillation=True)
    config["detector"].pop("init_cfg", None)
    config["train_cfg"] = None  # Training assigners are irrelevant to frozen prediction.
    config["track_head"]["train_cfg"] = None
    if (config["roi_head"]["bbox_head"]["num_classes"] != 1
            or config["tracker"]["with_cats"] is not False
            or config["tracker"]["memo_tracklet_frames"] != 10
            or config["test_cfg"]["rcnn"]["score_thr"] != .02
            or config["test_cfg"]["rcnn"]["max_per_img"] != 50):
        raise ValueError("Preregistered native configuration changed")
    return config


def register_image_modules() -> list[str]:
    """Package shells avoid executing upstream dataset/semantic __init__ files."""
    paths = ["masa", "masa.models", "masa.models.sam", "masa.models.detectors",
             "masa.models.necks", "masa.models.losses", "masa.models.roi_heads",
             "masa.models.roi_heads.track_heads", "masa.models.tracker", "masa.models.mot"]
    for name in paths:
        if name in sys.modules:
            raise RuntimeError("Unexpected pre-existing MASA package; use a fresh worker")
        shell = types.ModuleType(name)
        shell.__path__ = [str(SOURCE / name.replace(".", "/"))]
        sys.modules[name] = shell
    modules = ["masa.models.sam.common", "masa.models.sam.transformer", "masa.models.sam.prompt_encoder",
               "masa.models.sam.mask_decoder", "masa.models.sam.image_encoder", "masa.models.detectors.sam_masa",
               "masa.models.necks.simplefpn", "masa.models.losses.unbiased_contrastive_loss",
               "masa.models.roi_heads.track_heads.masa_track_head", "masa.models.tracker.masa_tao_tracker",
               "masa.models.mot.masa"]
    for name in modules:
        importlib.import_module(name)
    return modules


def build_frozen_model():
    import torch
    from mmengine.config import ConfigDict
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS

    records = verified_sources()
    modules = register_image_modules()
    init_default_scope("mmdet")
    config = native_config()
    model = MODELS.build(ConfigDict(config))
    plan = json.loads((ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json").read_text())
    spec = plan["weights"][0]
    checkpoint = ROOT / "checkpoints/masa_sam_candidate" / spec["filename"]
    digest = hashlib.sha256()
    with checkpoint.open("rb") as reader:
        for block in iter(lambda: reader.read(1024**2), b""):
            digest.update(block)
    if checkpoint.stat().st_size != spec["bytes"] or digest.hexdigest() != spec["sha256_from_repository_lfs_metadata"]:
        raise ValueError("Frozen checkpoint identity changed")
    # torch 2.1 mmap accepts a filename string, unlike newer PathLike support.
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    mismatches = [k for k in set(expected) & set(state) if tuple(expected[k].shape) != tuple(state[k].shape)]
    if missing or unexpected or mismatches:
        raise ValueError(json.dumps({"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatches}))
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"source_files": records, "registered_upstream_modules": modules,
                   "checkpoint_sha256": digest.hexdigest(), "strict_state_tensor_keys": len(expected),
                   "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [],
                   "all_parameters_frozen": not any(p.requires_grad for p in model.parameters()),
                   "native_model_config": config, "second_pretrain_weight_used": False,
                   "upstream_dataset_text_api_or_postprocess_imported": False}


def prepare_current_image(image_rgb, device):
    """Match native resize, RGB normalization and normalized-zero square padding."""
    import cv2
    import numpy as np
    import torch
    from mmdet.structures import DetDataSample

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Expected current RGB uint8 image only")
    height, width = image_rgb.shape[:2]
    scale = min(1024 / width, 1024 / height)
    new_width, new_height = int(width * scale + .5), int(height * scale + .5)
    resized = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pixels = torch.from_numpy(resized.copy()).permute(2, 0, 1).to(device=device, dtype=torch.float32)
    mean = pixels.new_tensor([123.675, 116.28, 103.53])[:, None, None]
    std = pixels.new_tensor([58.395, 57.12, 57.375])[:, None, None]
    pixels = (pixels - mean) / std
    pixels = torch.nn.functional.pad(pixels, (0, 1024 - new_width, 0, 1024 - new_height)).unsqueeze(0)
    sample = DetDataSample(metainfo={"ori_shape": (height, width), "img_shape": (new_height, new_width),
                                   "batch_input_shape": (1024, 1024), "pad_shape": (1024, 1024),
                                   "scale_factor": (new_width / width, new_height / height)})
    return pixels, sample


def infer_current_frame(model, image_rgb, ordinal: int, device):
    """Only current pixels/geometry and ordinal reach image heads and past memo."""
    import torch
    if ordinal < 0:
        raise ValueError("Expected video-local current ordinal")
    pixels, sample = prepare_current_image(image_rgb, device)
    sample.set_metainfo({"frame_id": ordinal})
    with torch.inference_mode():
        features = model.detector.backbone.forward_base_multi_level(pixels)
        adapted = model.masa_adapter(features)
        proposals = model.rpn_head.predict(adapted, [sample], rescale=False)
        sample.pred_instances = model.roi_head.predict(adapted, proposals, [sample], rescale=True)[0]
        if sample.pred_instances.labels.numel() and not torch.all(sample.pred_instances.labels == 0):
            raise ValueError("Native one-class foreground contract violated")
        tracks = model.tracker.track(model=model, img=pixels, feats=adapted, data_sample=sample, rescale=True, with_segm=False)
    return sample.pred_instances, tracks
