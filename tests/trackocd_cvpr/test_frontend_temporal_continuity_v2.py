from scripts.trackocd_cvpr.audit_frontend_temporal_continuity_v2 import _adjacent_continuity_summary


def test_adjacent_continuity_uses_annotation_order_not_source_frame_plus_one():
    gt_frames = {(7, "gt"): [30, 60, 90]}
    matched = {
        (7, "gt"): {
            30: (7, "pred-a"),
            60: (7, "pred-a"),
            90: (7, "pred-b"),
        }
    }
    result = _adjacent_continuity_summary(gt_frames, matched)
    assert result["gt_adjacent_annotation_pairs"] == 2
    assert result["gt_adjacent_pairs_with_both_stream_matches"] == 2
    assert result["gt_adjacent_pairs_with_source_gap_one"] == 0
    assert result["gt_adjacent_pairs_with_source_gap_gt_one"] == 2
    assert result["adjacent_same_pred_id_pairs"] == 1
    assert result["adjacent_pred_id_switch_pairs"] == 1
