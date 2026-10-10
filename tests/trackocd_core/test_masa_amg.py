"""No inference/data/GPU; pinned defaults, box math, barriers and no promotion."""
import ast
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.trackocd_core import masa_amg as amg
from scripts.trackocd_core import smoke_masa_amg as smoke


def test_exact_source_manifest_and_official_default_parameters():
    records = amg.verified_sources()
    assert len(records) == 4 and sum(r["bytes"] for r in records) == 43189
    source = (amg.SOURCE / "segment_anything/automatic_mask_generator.py").read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in zip(ctor.args.args[-len(ctor.args.defaults):], ctor.args.defaults)
                if arg.arg in amg.DEFAULTS}
    assert amg.DEFAULTS == defaults == json.loads(smoke.PLAN.read_text())["amg_defaults"]
    amg.require_defaults(defaults)
    with pytest.raises(ValueError):
        amg.require_defaults({**defaults, "pred_iou_thresh": .85})


def test_grid_is_full_fixed_current_image_not_semantic_or_gt_prompts():
    utility, transforms = amg.utility_modules()
    grid = utility.build_point_grid(32)
    assert grid.shape == (1024, 2)
    np.testing.assert_allclose(grid[0], [1 / 64, 1 / 64])
    np.testing.assert_allclose(grid[-1], [63 / 64, 63 / 64])
    transform = transforms.ResizeLongestSide(1024)
    assert transform.get_preprocess_shape(100, 200, 1024) == (512, 1024)
    np.testing.assert_allclose(transform.apply_coords(np.array([[100., 50.]]), (100, 200)), [[512, 256]])


def test_strict_iou_filter_empty_masks_and_disclosed_degenerate_counts():
    utility, _ = amg.utility_modules()
    logits = torch.full((6, 6, 8), -2.)
    logits[:3, 1:5, 2:6] = 2.
    logits[4, 1, 2] = 2.  # Original max-pixel convention yields zero area.
    logits[5, 1:5, 2:6] = .5  # Unstable at official +/-1 offset.
    quality = torch.tensor([.88, .89, 1.2, .9, .9, .9])
    boxes, raw_quality, stability, counts = amg.filter_masks(logits, quality, utility)
    assert counts == {"raw_masks": 6, "after_predicted_iou": 5, "undefined_stability_excluded": 1,
                      "after_stability": 3, "degenerate_boxes_excluded": 1, "before_nms": 2}
    torch.testing.assert_close(boxes, torch.tensor([[2., 1., 5., 4.], [2., 1., 5., 4.]]))
    torch.testing.assert_close(raw_quality, torch.tensor([.89, 1.2]))  # No sigmoid or clipping.
    torch.testing.assert_close(stability, torch.ones(2))


@pytest.mark.parametrize("fault", ["logit_nan", "quality_inf", "shape"])
def test_bad_raw_outputs_fail_without_repair(fault):
    utility, _ = amg.utility_modules()
    logits, quality = torch.zeros(1, 3, 3), torch.ones(1)
    if fault == "logit_nan":
        logits[0, 0, 0] = float("nan")
    elif fault == "quality_inf":
        quality[0] = float("inf")
    else:
        quality = torch.ones(2)
    with pytest.raises(ValueError):
        amg.filter_masks(logits, quality, utility)


def test_no_qualifying_primary_full_val_or_training_permission():
    config = json.loads(smoke.PLAN.read_text())
    assert config["primary_freeze_permitted"] is False and config["scientific_pass_permitted"] is False
    assert config["training"] is False and config["test_access"] is False
    assert config["limits"]["unique_train_images"] == 4 and config["limits"]["real_image_forwards"] == 8
    assert config["limits"]["gpu_workers"] == 1 and config["amg_defaults"]["crop_n_layers"] == 0
    assert config["worker_model_input_fields"] == ["current_image", "image_geometry", "video_local_frame_ordinal"]
    for relative, digest in config["unchanged_sources_sha256"].items():
        assert smoke.sha256_file(smoke.ROOT / relative) == digest
    source = inspect.getsource(amg.infer_current_frame)
    assert "rpn_head" not in source and "roi_head" not in source and "sigmoid" not in source
    assert "max_per_img" not in source and "with_segm=False" in source


def test_runtime_barrier_in_fresh_process():
    code = '''
from scripts.trackocd_core.smoke_masa_amg import worker_input_barrier
import socket, subprocess
worker_input_barrier()
actions = [lambda: socket.create_connection(("127.0.0.1", 1)),
           lambda: subprocess.run(["true"]),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/annotations/train.json"),
           lambda: open("/tmp/recovered_splits/roles.json"),
           lambda: open("/tmp/gt_train_known/selection_plan.json"),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/frames/val/anonymous.jpg"),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/frames/test/anonymous.jpg")]
for action in actions:
    try: action()
    except PermissionError: pass
    else: raise AssertionError("Boundary did not reject input")
print("7 boundary rejections")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=smoke.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "7 boundary rejections"


def test_existing_or_partial_attempt_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "RUN", tmp_path)
    with pytest.raises(ValueError, match="Preserve"):
        smoke.preflight("not-a-commit")
