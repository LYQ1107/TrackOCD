"""Read-only protocol facade for Phase25.

Phase25 deliberately reuses the corrected Phase23/24 loader and candidate
construction without modifying those namespaces.  All model-facing inputs
remain causal proposal/feature fields; TRAIN labels are used only when
building the fold manifests and validation targets.
"""
from src.iclr27_phase24.protocol import (  # noqa: F401
    CSV_PATH, FEAT_PATH, FEAT_META_PATH, POS_PATH, P22_MANIFEST,
    PREFIXES, IOU_THRESHOLDS, SCALE_FACTORS, CENTER_SHIFTS, MAX_HISTORY,
    by_track, box_iou, event_indices, fval, fixed_transforms,
    load_aligned_features, load_events, normalized_gt, order_key,
    parse_box, raw_box, row_key, track_key, track_positions,
)
from scripts.iclr27_phase23.train_quality_ranker import (  # noqa: F401
    TRANSFORM_META, candidate_arrays, feature_batch, iou_vec,
)

PROTOCOL = "trackocd_iclr27_phase25_mot_preserving_proposal_generalization"
RELIABLE_RULE = "parent assigned == 1 and transformed true normalized IoU >= 0.5"
TOP_KS = (5, 10, 20, 27)
GEOM_FIELDS = (
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm",
    "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm",
    "box_aspect_log", "border_left_norm", "border_top_norm",
    "border_right_norm", "border_bottom_norm", "causal_prefix_age_norm",
    "causal_box_stability_iou",
)
