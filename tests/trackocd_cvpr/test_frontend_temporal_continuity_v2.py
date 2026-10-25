from scripts.trackocd_cvpr.audit_frontend_temporal_continuity_v2 import _adjacent_continuity_summary


def test_adjacent_continuity_uses_annotation_order_not_source_frame_plus_one():
    gt_frames = {(7, "gt"): [30, 60, 120]}
    matched = {
        (7, "gt"): {
            30: (7, "pred-a"),
            60: (7, "pred-a"),
            120: (7, "pred-b"),
        }
    }
    result = _adjacent_continuity_summary(gt_frames, matched)
    assert result["gt_adjacent_annotation_pairs"] == 2
    assert result["gt_adjacent_pairs_with_both_stream_matches"] == 2
    assert result["gt_adjacent_pairs_with_source_gap_one"] == 0
    assert result["gt_adjacent_pairs_with_source_gap_gt_one"] == 2
    assert result["adjacent_same_pred_id_pairs"] == 1
    assert result["adjacent_pred_id_switch_pairs"] == 1
    by_gap = {row["gap"]: row for row in result["adjacent_pair_outcomes_by_source_gap_top20"]}
    assert by_gap[30]["same_pred_id"] == 1
    assert by_gap[60]["pred_id_switch"] == 1
