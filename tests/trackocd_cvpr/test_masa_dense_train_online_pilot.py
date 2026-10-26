import json

from scripts.trackocd_cvpr import run_masa_dense_train_online_pilot as pilot


def test_masa_dense_pilot_is_fixed_train_only_and_does_not_overwrite():
    config = json.loads(pilot.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert len(config["image_paths"]) == 32
    assert config["image_paths"] == sorted(config["image_paths"])
    assert config["source_split"] == "train_image_only"
    assert config["causal_contract"]["training"] is False
    assert config["causal_contract"]["val_or_test_access"] is False
    assert config["causal_contract"]["gt_or_category_fields_read"] is False
    assert pilot.DEFAULT_OUTPUT.parent.name == "masa_dense_train_online_pilot_bdd32_v1"
