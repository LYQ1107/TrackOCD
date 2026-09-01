"""Read-only Phase24 wrappers around the corrected Phase23 protocol.

All large frozen inputs and evaluators stay in their previous namespaces.  This
module only exposes the key-aligned row/candidate helpers so Phase24 artifacts
can be regenerated without modifying Phase20--23 files.
"""
from __future__ import annotations

from src.iclr27_phase23.protocol import (  # noqa: F401
    CSV_PATH, FEAT_PATH, FEAT_META_PATH, POS_PATH, P22_MANIFEST,
    PREFIXES, IOU_THRESHOLDS, SCALE_FACTORS, CENTER_SHIFTS, MAX_HISTORY,
    by_track, box_iou, event_indices, fval, fixed_transforms, load_aligned_features,
    load_events, normalized_gt, order_key, parse_box, raw_box, row_key,
    track_key, track_positions,
)
from scripts.iclr27_phase23.train_quality_ranker import (  # noqa: F401
    TRANSFORM_META, candidate_arrays, feature_batch, iou_vec,
)

PROTOCOL = "trackocd_iclr27_phase24_proposal_selection_source_generalization"
RELIABLE_RULE = "parent assigned == 1 and true normalized IoU >= 0.5"

