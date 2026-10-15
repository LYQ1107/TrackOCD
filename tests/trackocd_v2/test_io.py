from pathlib import Path


def test_output_layout_uses_reversible_project_local_link(monkeypatch, tmp_path):
    from src.trackocd_v2 import io

    target = tmp_path / "runtime" / "project_outputs"
    link = tmp_path / "project" / "outputs" / "trackocd_v2"
    monkeypatch.setattr(io, "OUTPUT_TARGET", target)
    monkeypatch.setattr(io, "OUTPUT_LINK", link)

    assert io.ensure_output_layout() == target
    assert link.is_symlink()
    assert Path(link).resolve() == target.resolve()
    assert (target / "audit").is_dir()
    assert (target / "features").is_dir()


def test_default_output_target_is_not_the_old_server_namespace():
    from src.trackocd_v2 import io

    assert "/data2/usr_for_deadline/trackocd_v2" not in str(io._DEFAULT_OUTPUT_TARGET)
    assert io._DEFAULT_OUTPUT_TARGET == (io.PROJECT_ROOT / "outputs/trackocd_v2_runtime")
