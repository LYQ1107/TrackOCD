"""Separate mask-route seals, fixed metadata protocol and evaluator barriers."""
import inspect
import json
import subprocess
import sys
import numpy as np
import pytest

from src.trackocd_core.amg_physical_diagnostic import validate_arrays, seal_video, completed_video
from scripts.trackocd_core import run_masa_amg_diagnostic as runner
from scripts.trackocd_core import evaluate_masa_amg_diagnostic as evaluator


def fixture_arrays(count=60):
    video = {"video_id": 4, "images": [{"image_id": 4, "frame_index": 750}]}
    boxes = np.asarray([[i, 0, i + 1, 2] for i in range(count)], dtype=np.float32).reshape(-1, 4)
    arrays = {"image_id": np.asarray([4], dtype=np.int64), "frame_index": np.asarray([750], dtype=np.int64),
              "frame_offsets": np.asarray([0, count], dtype=np.int64), "det_offsets": np.asarray([0, count], dtype=np.int64),
              "track_id": np.arange(count, dtype=np.int64), "boxes": boxes, "det_boxes": boxes.copy(),
              "score": np.ones(count, dtype=np.float32), "det_score": np.ones(count, dtype=np.float32)}
    return video, arrays


@pytest.mark.parametrize("count", [0, 60])
def test_full_mask_output_over_50_and_empty_frames_are_sealed(tmp_path, count):
    video, arrays = fixture_arrays(count)
    validate_arrays(arrays, video)
    (tmp_path / "shards").mkdir()
    row = seal_video(tmp_path, video, arrays, "registered", {"frozen_state_unchanged": True})
    assert completed_video(tmp_path, video, "registered") == row
    assert row["prediction_rows"] == count and row["detection_rows"] == count
    with pytest.raises(ValueError, match="overwrite"):
        seal_video(tmp_path, video, arrays, "registered", {"frozen_state_unchanged": True})
    with pytest.raises(ValueError):
        completed_video(tmp_path, video, "different-config")


@pytest.mark.parametrize("fault", ["missing_frame", "offset", "duplicate_id", "nan", "zero_area", "bad_quality"])
def test_invalid_masks_are_not_repaired_or_gt_filtered(fault):
    video, arrays = fixture_arrays()
    if fault == "missing_frame": arrays["image_id"][0] = 5
    if fault == "offset": arrays["frame_offsets"][1] -= 1
    if fault == "duplicate_id": arrays["track_id"][1] = 0
    if fault == "nan": arrays["boxes"][0, 0] = np.nan
    if fault == "zero_area": arrays["boxes"][0, 2] = arrays["boxes"][0, 0]
    if fault == "bad_quality": arrays["score"][0] = 1.2
    with pytest.raises(ValueError):
        validate_arrays(arrays, video)


def test_fixed_original_image_plan_has_no_novel_target_resampling():
    config, plan, digest = runner.load_plan()
    assert config["video_ids"] == [4, 20, 22, 23] and config["images_per_video"] == [16] * 4
    assert sum(len(v["images"]) for v in plan["videos"]) == 64
    assert config["private_plan_sha256"] == "f04fce6b4ee1173a2aef2efde145df6946183ae7ce27de1c68be051d3245b02e"
    assert len(digest) == 64 and config["limits"]["max_gpu_workers"] == 1
    assert not config["primary_freeze_permitted"] and not config["scientific_pass_permitted"]
    assert not config["training"] and not config["test_access"]
    assert not config["evaluation"]["categories_for_input_or_tuning"]
    assert not config["evaluation"]["full_val_or_ocd_metrics_claim"]


def test_worker_barrier_rejects_gt_test_and_external_children():
    code = '''
from scripts.trackocd_core.run_masa_amg_diagnostic import input_barrier
import socket,subprocess
input_barrier()
actions = [lambda: socket.create_connection(("127.0.0.1", 1)), lambda: subprocess.run(["true"]),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json"),
           lambda: open("/tmp/recovered_splits/roles.json"),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/frames/test/forbidden.jpg")]
for action in actions:
    try: action()
    except PermissionError: pass
    else: raise AssertionError("Boundary not rejected")
print("5 rejections")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=runner.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "5 rejections"


def test_evaluator_cannot_read_gt_before_new_and_old_seals_verified():
    source = inspect.getsource(evaluator.main)
    assert source.index("completed_video(RUN") < source.index("json.loads(annotation.read_text())")
    assert source.index("supervisor[\"worker_returncode\"]") < source.index("json.loads(annotation.read_text())")
    assert source.index("Do not rerun/replace baseline bytes") < source.index("json.loads(annotation.read_text())")
    assert "MAX_DETECTIONS=300" in source and 'SUBSET="all"' in source
    assert '"MASA_SAM_GRID", "MASA_NATIVE", "PANDAS_BT_FG0"' in source
    assert "historical_baselines_reproduced" in source and "INCOMPARABLE_BASELINE_REPLAY_MISMATCH" in source


def test_ended_or_partial_prediction_cannot_be_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    with pytest.raises(ValueError, match="Preserve"):
        runner.parent("irrelevant")
