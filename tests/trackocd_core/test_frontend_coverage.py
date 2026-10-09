import numpy as np
import pytest

from scripts.trackocd_core.audit_frontend_coverage import match_video
from src.evaluation.track_matching import temporal_iou


def test_temporal_union_matches_legacy_geometry_and_ignores_category():
    box = [0., 0., 10., 10.]
    gt = [{"key": "g", "category": 999, "boxes": {1: box, 2: box}}]
    frames = {1: (np.array([7]), np.array([box])),
              3: (np.array([7]), np.array([box]))}
    result = match_video(gt, frames)["matches"]["g"]
    assert result["temporal_iou"] == pytest.approx(temporal_iou(gt[0]["boxes"], {1: box, 3: box}))
    assert not result["reliable"]
    gt[0]["category"] = -123
    assert match_video(gt, frames)["matches"]["g"] == result


def test_one_physical_prediction_cannot_cover_two_gt_tracks():
    box = [0., 0., 10., 10.]
    gt = [{"key": key, "boxes": {1: box}} for key in ("a", "b")]
    result = match_video(gt, {1: (np.array([7]), np.array([box]))})
    assert sum(row["reliable"] for row in result["matches"].values()) == 1
