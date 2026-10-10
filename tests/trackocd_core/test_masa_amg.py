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


def test_actual_bounded_result_is_only_engineering_and_frozen_causal():
    result = json.loads(smoke.SUMMARY.read_text())
    assert result["status"] == "PASS_BOUNDED_SAM_GRID_INTERFACE_NOT_PRIMARY_QUALIFICATION"
    assert result["preregistration_commit"] == "d1a53d055f4e2b1483d21b128505dfae0bf99b38"
    assert result["config_sha256"] == smoke.sha256_file(smoke.PLAN)
    assert result["real_image_forwards_started"] == 8 and result["input"]["unique_train_images"] == 4
    assert result["strict_state_tensor_keys"] == 419 and result["all_parameters_frozen"]
    assert result["model_state_initial_sha256"] == result["model_state_final_sha256"] == "33443cac52ad4eb4bad4f7c87627443c5c2d04268273c2b96963c6feb373264f"
    assert result["supervisor"]["worker_returncode"] == 0
    assert all(c["same_current_pixels"] and c["exact_arrays_equal"] for c in result["prefix_invariance"]["comparisons"])
    assert result["prefix_invariance"]["future_pixels_actually_changed"]
    assert not any(result[k] for k in ("primary_freeze_permitted", "scientific_pass_permitted", "training", "optimizer_used",
                                      "formal_cache_started", "val_or_test_access", "gt_runtime_input", "external_process_interference"))
    limits = json.loads(smoke.PLAN.read_text())["limits"]
    assert result["elapsed_seconds"] < limits["wall_seconds"]
    assert result["peak_host_rss_bytes"] < limits["host_rss_bytes"]
    assert result["peak_gpu_reserved_bytes"] < limits["gpu_reserved_bytes"]
    assert result["supervisor"]["private_allocated_bytes"] < limits["output_allocated_bytes"]


def test_actual_native_iou_quality_not_silently_treated_as_probability():
    result = json.loads(smoke.SUMMARY.read_text())
    assert [[f["detections"]["count"] for f in r["frames"]] for r in result["replays"]] == [[36, 32, 35, 40], [36, 32, 37, 35]]
    for replay in result["replays"]:
        for frame in replay["frames"]:
            assert frame["masks"]["raw_masks"] == 3072 and frame["masks"]["prompt_batches"] == 16
            assert frame["masks"]["encoder_forwards"] == 1 and frame["masks"]["crop_layers"] == 0
            assert frame["masks"]["proposal_cap"] is None
            assert frame["masks"]["raw_predicted_iou_max"] > 1  # Real native quality; no clamp/sigmoid rescue.
            assert frame["masks"]["degenerate_boxes_excluded"] == 0
            assert .95 <= frame["tracks"]["score_min"] <= frame["tracks"]["score_max"] <= 1
            assert "not class/foreground probability" in frame["masks"]["quality_contract"]
