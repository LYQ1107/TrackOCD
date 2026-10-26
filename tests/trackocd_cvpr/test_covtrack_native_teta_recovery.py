import json

from scripts.trackocd_cvpr.run_covtrack_native_teta_recovery import CONFIG, ROOT


def test_teta_recovery_config_is_val_only_and_nonselecting():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["teta_commit"] == "a62a9c0affec3a97f2cd0263141c53bcfb9c79f7"
    assert config["boundaries"]["val_only"] is True
    assert config["boundaries"]["tao_test_read"] is False
    assert config["boundaries"]["frontend_inference_started"] is False
    assert config["boundaries"]["model_selection_performed"] is False
    for key in ("native_stream", "annotation", "known_ids", "novel_ids", "output_directory"):
        assert "test" not in config[key].lower()


def test_teta_recovery_receipt_is_complete_and_reference_only():
    path = ROOT / "outputs/trackocd_cvpr/cvpr5/covtrack_native_teta_recovery_v2/result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "PASS_VAL_TETA_REFERENCE_RECOVERED_NO_SELECTION"
    assert result["inputs"]["tao_test_read"] is False
    assert result["inputs"]["frontend_inference_started"] is False
    assert result["inputs"]["training_started"] is False
    assert result["teta_reference"]["status"] == "COMPLETE"
    assert result["inputs"]["converted_rows"] == 1133872
    assert result["inputs"]["converted_categories"] == 295
    assert result["boundaries"]["goal_acceptance"] is False
