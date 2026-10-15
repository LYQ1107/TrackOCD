from pathlib import Path

from scripts.trackocd_v2.audit_public_covtrack_recovery import _asset, _config_contract


def test_public_covtrack_config_contract_is_explicit(tmp_path):
    config = tmp_path / "ovtrack_r50_ctao_train.py"
    config.write_text(
        "\n".join(
            [
                "prompt_path='saved_models/pretrained_models/detpro_prompt.pt'",
                "only_validation_categories=True",
                "num_classes=1203",
                "use_special_text_prompt=True",
                "use_special_image_prompt=True",
                "confused_features=True",
                "saved_models/ctao_dataset/ctao_base.json",
            ]
        ),
        encoding="utf-8",
    )
    result = _config_contract(config)
    assert result["exists"] is True
    assert all(result["tokens"].values())
    assert result["interpretation"]["prompt_used_for_model_selection"] is False


def test_public_covtrack_asset_hash_is_checked(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"trackocd")
    result = _asset(path, 8, "6bb9e7a1f7d0f3f9e3e4d7c8f5f9c67a9f9c8f3b6f7a3d9e8b3cc5e0c5e5e2a2")
    assert result["exists"] is True
    assert result["bytes"] == 8
    assert result["integrity"] == "FAIL"
    assert Path(result["path"]) == path.resolve()
