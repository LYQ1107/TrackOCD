"""Literal native config and source identity; no upstream model execution."""
import ast
import inspect

import pytest

from src.trackocd_core.masa_native import assignment, literal_config, native_config, verified_sources
from src.trackocd_core.masa_native import SOURCE
from src.trackocd_core.masa_native import build_frozen_model


def test_native_literal_decoder_does_not_execute_config_or_base():
    source = "raise RuntimeError('not executed')\n_base_=['missing_dataset.py']\nmodel=dict(detector=allowed, max_distance=-1)\n"
    assert assignment(source, "model", {"allowed": {"type": "SamMasa"}}) == {
        "detector": {"type": "SamMasa"}, "max_distance": -1}


@pytest.mark.parametrize("expression", ["__import__('os')", "unknown", "factory()", "dict(**unknown)", "'a'/2", "True/2", "1/0", "1+2"])
def test_native_literal_decoder_rejects_dynamic_expressions(expression):
    with pytest.raises(ValueError):
        literal_config(ast.parse(expression, mode="eval").body, {})


def test_native_numeric_division_remains_literal_only():
    assert literal_config(ast.parse("1.0/9.0", mode="eval").body, {}) == 1.0 / 9.0


def test_frozen_load_uses_string_filename_for_torch21_mmap_without_unsafe_fallback():
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(build_frozen_model)))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
    call, = calls
    assert isinstance(call.args[0], ast.Call) and call.args[0].func.id == "str"
    assert {k.arg: k.value.value for k in call.keywords} == {"map_location": "cpu", "weights_only": True, "mmap": True}


def test_exact_source_allowlist_has_no_dataset_semantic_or_demo_modules():
    if not SOURCE.exists():
        pytest.skip("Private pinned source assets are not included in Git")
    records = verified_sources()
    assert len(records) == 14 and sum(r["bytes"] for r in records) == 125292
    assert not any("datasets/" in r["path"] or "demo/" in r["path"] or "apis/" in r["path"] for r in records)
    assert all(len(r["sha256"]) == 64 for r in records)


def test_native_config_preserves_fixed_one_class_proposals_without_public_inputs():
    if not SOURCE.exists():
        pytest.skip("Private pinned source assets are not included in Git")
    config = native_config()
    assert config["load_public_dets"] is False and config["given_dets"] is False
    assert config["public_det_path"] is None
    assert config["detector"]["type"] == "SamMasa"
    assert "init_cfg" not in config["detector"]
    assert config["roi_head"]["bbox_head"]["num_classes"] == 1
    assert config["roi_head"]["bbox_head"]["reg_class_agnostic"]
    assert config["tracker"]["with_cats"] is False
    assert config["test_cfg"]["rcnn"]["score_thr"] == .02
    assert config["test_cfg"]["rcnn"]["max_per_img"] == 50
    assert config["train_cfg"] is None and config["track_head"]["train_cfg"] is None
