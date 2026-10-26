import json

from scripts.trackocd_cvpr import candidate_family_peak_order_control as control


def test_family_peak_order_control_is_fixed_and_unselected():
    config = json.loads(control.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    result = json.loads(control.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert config["orders"] == [[0, 1], [1, 0]]
    assert result["status"] == "PASS_FAMILY_PEAK_ORDER_CONTROL_SMOKE"
    assert [row["order"] for row in result["per_order"]] == [[0, 1], [1, 0]]
    assert all(row["frames"] == 4 for row in result["per_order"])
    assert result["mechanism"]["order_winner_selected"] is False
    assert result["resources"]["optimizer_steps"] == 0
    assert result["boundaries"]["labels_or_gt_in_numeric_forward"] is False
