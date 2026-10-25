import json

import pytest

from scripts.trackocd_cvpr import audit_masa_dense_train_online_smoke as smoke


def test_dense_masa_smoke_is_frozen_train_only_and_adjacent():
    config = json.loads(smoke.CONFIG.read_text())
    assert config["source_split"] == "train_image_only"
    assert len(config["image_paths"]) == 8
    assert config["image_paths"] == sorted(config["image_paths"])
    assert config["causal_contract"] == {
        "worker_inputs": ["current_image", "video_local_frame_ordinal"],
        "future_observations_used": False,
        "gt_or_category_fields_read": False,
        "val_or_test_access": False,
        "training": False,
        "offline_filtering": False,
        "public_vocabulary_used": False,
        "formal_metrics": False,
    }
    paths = smoke._image_paths(config)
    assert [int(path.stem) for path in paths] == list(range(1141, 1149))


def test_dense_masa_smoke_does_not_overwrite_existing_receipt(tmp_path, monkeypatch):
    existing = tmp_path / "run"
    existing.mkdir()
    monkeypatch.setattr(smoke, "RUN_ROOT", existing)
    monkeypatch.setattr(smoke, "OUTPUT", tmp_path / "result.json")
    with pytest.raises(ValueError, match="preserve an existing dense MASA smoke"):
        smoke.parent()
