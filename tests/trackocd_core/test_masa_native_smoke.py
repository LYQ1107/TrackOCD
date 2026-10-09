"""Private image/runtime rejection paths; no data, model or GPU execution."""
import subprocess
import sys
import json
from pathlib import Path

import pytest

import scripts.trackocd_core.smoke_masa_native as smoke


def test_only_four_existing_unique_images_from_one_train_video(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "DATASET", tmp_path)
    video = tmp_path / "train" / "anonymous_video"
    video.mkdir(parents=True)
    paths = ["train/anonymous_video/" + str(i) + ".jpg" for i in range(4)]
    for path in paths:
        (tmp_path / path).write_bytes(b"fixture, not decoded")
    assert smoke.validate_train_images(paths) == [tmp_path / p for p in paths]
    for bad in [paths[:3], [paths[0]] * 4, ["val/x.jpg", *paths[1:]], ["test/x.jpg", *paths[1:]],
                ["train/../../elsewhere", *paths[1:]], [str(tmp_path / paths[0]), *paths[1:]]]:
        with pytest.raises(ValueError):
            smoke.validate_train_images(bad)


def test_train_image_symlink_cannot_escape_train_root(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "DATASET", tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "linked").symlink_to(outside, target_is_directory=True)
    for i in range(4):
        (outside / f"{i}.jpg").write_bytes(b"fixture")
    with pytest.raises(ValueError):
        smoke.validate_train_images([f"train/linked/{i}.jpg" for i in range(4)])


def test_network_and_external_child_guard_in_fresh_process():
    code = """
from scripts.trackocd_core.smoke_masa_native import deny_network_and_external_children
import socket,subprocess,sys
deny_network_and_external_children()
for action in [lambda:socket.getaddrinfo('example.com',443),lambda:subprocess.run([sys.executable,'-c','print(1)'])]:
    try:
        action()
    except PermissionError:
        continue
    raise RuntimeError('guard failed')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_delivered_real_smoke_is_bounded_anonymous_and_not_frontend_qualification():
    path = smoke.ROOT / "outputs/trackocd_core/audit/masa_native_smoke.json"
    if not path.exists():
        pytest.skip("Small bounded-smoke receipt is not present")
    result = json.loads(path.read_text())
    assert result["status"] == "PASS_BOUNDED_FROZEN_NATIVE_SMOKE_NOT_M1_QUALIFICATION"
    assert result["real_image_forwards_started"] == 8
    assert result["frozen_model"]["strict_state_tensor_keys"] == 419
    assert result["frozen_model"]["all_parameters_frozen"] and result["all_state_tensors_finite"]
    assert not result["frozen_model"]["second_pretrain_weight_used"]
    assert not any(result["boundary"].values())
    assert result["prefix_invariance"]["future_pixels_actually_changed"]
    assert all(r["same_current_pixels"] and r["exact_detection_and_track_arrays_equal"]
               for r in result["prefix_invariance"]["comparisons"])
    assert len(result["replays"]) == 2
    assert all(len(replay["frames"]) == 4 for replay in result["replays"])
    assert all(row["detections"]["count"] == 50 for replay in result["replays"] for row in replay["frames"])
    assert result["peak_host_rss_bytes"] < 4 * 1024**3
    assert result["peak_gpu_reserved_bytes"] < 8 * 1024**3
    assert result["supervisor"]["peak_candidate_allocated_bytes"] < 8 * 1024**3


def test_delivered_runtime_summary_distinguishes_uv_operation_and_record_integrity():
    path = smoke.ROOT / "outputs/trackocd_core/audit/masa_runtime_install_summary.json"
    if not path.exists():
        pytest.skip("Small isolated-install summary is not present")
    result = json.loads(path.read_text())
    assert result["completed_package_count"] == 48
    assert result["base_distribution_snapshot_unchanged"]
    assert result["hash_required"] and result["binary_only"] and result["no_dependencies_or_indexes"]
    assert not any(result["boundary"].values())
    assert result["record_hashed_files"] == 20423 and result["record_hashed_bytes"] == 5325373594
    assert all(r["installed_record_integrity"]["mismatch_count"] == 0 for r in result["completed_packages"])
    assert all(r["require_hashes"] and r["returncode"] == 0 for r in result["verified_install_operations"].values())
