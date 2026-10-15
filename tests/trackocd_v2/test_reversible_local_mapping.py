from pathlib import Path

from scripts.trackocd_v2 import prepare_reversible_local_mapping as mapping


def test_mapping_is_dry_run_and_refuses_existing_directory(tmp_path):
    tao = tmp_path / "tao"
    roles = tmp_path / "roles"
    (tao / "annotations").mkdir(parents=True)
    (tao / "frames").mkdir()
    roles.mkdir()
    for name in ("train.json", "validation.json"):
        (tao / "annotations" / name).write_text(name, encoding="utf-8")
    for name in ("known_ids.json", "unknown_ids_val.json", "distractor_ids.json"):
        (roles / name).write_text(name, encoding="utf-8")
    original = mapping.EXPECTED.copy()
    try:
        mapping.EXPECTED.update({
            "train_annotation": mapping._sha256(tao / "annotations/train.json"),
            "val_annotation": mapping._sha256(tao / "annotations/validation.json"),
            "known_ids": mapping._sha256(roles / "known_ids.json"),
            "unknown_ids_val": mapping._sha256(roles / "unknown_ids_val.json"),
            "distractor_ids": mapping._sha256(roles / "distractor_ids.json"),
        })
        report = mapping.build_report(tmp_path, tao, roles)
        assert report["status"] == "READY_TO_APPLY"
        assert report["copy_performed"] is False
        assert not (tmp_path / "data").exists()
        (tmp_path / "data/raw/tao/frames").mkdir(parents=True)
        refused = mapping.build_report(tmp_path, tao, roles)
        assert refused["status"] == "REFUSED_SOURCE_OR_TARGET_CHECK"
        assert "frames_train_val_only:refuse_existing_non_symlink_target" in refused["problems"]
    finally:
        mapping.EXPECTED.clear()
        mapping.EXPECTED.update(original)


def test_apply_creates_only_reversible_symlinks(tmp_path):
    tao = tmp_path / "tao"
    roles = tmp_path / "roles"
    (tao / "annotations").mkdir(parents=True)
    (tao / "frames").mkdir()
    roles.mkdir()
    for name in ("train.json", "validation.json"):
        (tao / "annotations" / name).write_text(name, encoding="utf-8")
    for name in ("known_ids.json", "unknown_ids_val.json", "distractor_ids.json"):
        (roles / name).write_text(name, encoding="utf-8")
    original = mapping.EXPECTED.copy()
    try:
        for key, path in {
            "train_annotation": tao / "annotations/train.json",
            "val_annotation": tao / "annotations/validation.json",
            "known_ids": roles / "known_ids.json",
            "unknown_ids_val": roles / "unknown_ids_val.json",
            "distractor_ids": roles / "distractor_ids.json",
        }.items():
            mapping.EXPECTED[key] = mapping._sha256(path)
        report = mapping.build_report(tmp_path, tao, roles)
        manifest = tmp_path / "manifest.json"
        applied = mapping.apply_mapping(report, manifest_path=manifest)
        assert applied["status"] == "APPLIED_REVERSIBLE_SYMLINK_MAPPING"
        assert applied["copy_performed"] is False
        assert manifest.is_file()
        for item in applied["assets"]:
            assert Path(item["target"]).is_symlink()
        assert applied["test_paths_mapped"] == []
        assert applied["predicted_stream_touched"] is False
    finally:
        mapping.EXPECTED.clear()
        mapping.EXPECTED.update(original)
