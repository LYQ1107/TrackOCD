"""Load a frozen PANDAS detector with anonymous prototype outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def load_anonymous_model(
    pandas_root: Path,
    base_checkpoint: Path,
    prototype_checkpoint: Path,
    device: str,
):
    script_root = Path(__file__).resolve().parent
    if str(script_root) not in sys.path:
        sys.path.insert(0, str(script_root))
    import pandas_torchvision_compat

    if str(pandas_root) not in sys.path:
        sys.path.insert(0, str(pandas_root))
    pandas_torchvision_compat.patch_pandas_roi_heads()
    from model_components.faster_rcnn_predictors import FasterRCNNPredictorNCDMaskOrig
    from model_components.pandas_model import NCDModel

    prototype_state = torch.load(prototype_checkpoint, map_location="cpu", weights_only=False)
    class_prototypes = prototype_state["class_prototypes"].float()
    base_count = int(prototype_state["base_count"])
    num_clusters = int(prototype_state["num_clusters"])
    total_classes = int(class_prototypes.shape[0]) + 1
    if total_classes != base_count + num_clusters + 1:
        raise ValueError("prototype metadata does not match prototype tensor")

    model = NCDModel(
        num_classes=total_classes,
        num_base_classes=91,
        base_checkpoint=str(base_checkpoint),
        num_clusters=num_clusters,
        prototype_init="pandas",
        similarity_metric=str(prototype_state.get("similarity_metric", "invert_square")),
        proba_norm=str(prototype_state.get("proba_norm", "l1")),
        background_classifier=str(prototype_state.get("background_classifier", "softmax")),
        device=device,
        dataset="tao",
        dets_per_img=int(prototype_state.get("dets_per_image", 300)),
        score_thresh=float(prototype_state.get("score_thresh", 0.0)),
        kmeans_n_init=int(prototype_state.get("kmeans_n_init", 10)),
        kmeans_max_iter=int(prototype_state.get("kmeans_max_iter", 1000)),
        kmeans_seed=int(prototype_state.get("kmeans_seed", 42)),
    )
    model.class_prototypes = class_prototypes.to(device)
    model.num_prototypes = len(class_prototypes)
    model.num_base_classes = base_count
    model.base_ids = np.arange(1, base_count + 1, dtype=np.int64)
    model.od_model.roi_heads.score_thresh = float(prototype_state.get("score_thresh", 0.0))
    model.od_model.roi_heads.detections_per_img = int(prototype_state.get("dets_per_image", 300))

    original_predictor = model.od_model.roi_heads.box_predictor
    model.od_model.roi_heads.box_predictor = FasterRCNNPredictorNCDMaskOrig(
        bbox_pred=original_predictor.bbox_pred,
        prediction_model=model,
        cls_score_orig=original_predictor.cls_score,
        cluster_mapping=None,
        l2_normalize=model.l2_normalize,
        num_classes=total_classes,
        mask_type="orig_model",
        device=device,
    )
    model.od_model.eval()
    return model, prototype_state
