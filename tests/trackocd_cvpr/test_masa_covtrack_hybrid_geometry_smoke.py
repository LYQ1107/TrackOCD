import json

from scripts.trackocd_cvpr import audit_masa_covtrack_hybrid_geometry_smoke as smoke


def test_hybrid_rule_is_fixed_and_category_free():
    config = json.loads(smoke.CONFIG.read_text())
    assert config["fusion_rule"]["name"] == "covtrack_first_union_dedup"
    assert config["fusion_rule"]["iou_threshold"] == 0.5
    assert config["fusion_rule"]["score_threshold_search"] is False
    assert config["boundaries"]["prediction_to_gt_mapping_persisted"] is False
    assert config["boundaries"]["training_started"] is False


def test_hybrid_dedup_keeps_covtrack_first():
    import numpy as np

    cov = [("c", np.asarray([0, 0, 10, 10], dtype=np.float32))]
    masa = [
        ("m-overlap", np.asarray([1, 1, 9, 9], dtype=np.float32)),
        ("m-new", np.asarray([20, 20, 30, 30], dtype=np.float32)),
    ]
    result = smoke._hybrid(cov, masa, 0.5)
    assert [item[0] for item in result] == ["c", "m-new"]
