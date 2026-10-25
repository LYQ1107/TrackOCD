import numpy as np

from scripts.trackocd_cvpr.audit_frontend_iou_sensitivity import _threshold_match_counts


def test_threshold_match_counts_respects_iou_cutoff_and_assignment():
    matrix = np.array([[0.8, 0.2], [0.1, 0.7]], dtype=float)
    gt = [(1, "g0"), (1, "g1")]
    pred = [(1, "p0"), (1, "p1")]
    matched, matched_gt, matched_pred = _threshold_match_counts(matrix, gt, pred, 0.5)
    assert matched == 2
    assert matched_gt == set(gt)
    assert matched_pred == set(pred)
    matched, _, _ = _threshold_match_counts(matrix, gt, pred, 0.9)
    assert matched == 0
