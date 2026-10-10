"""Fixed cached-detector association, causal input and classical seal boundaries."""
import inspect
import json
import subprocess
import sys
import numpy as np
import pytest

from src.trackocd_core import rpn_bytetrack as bt
from scripts.trackocd_core import run_rpn_bytetrack_diagnostic as runner
from scripts.trackocd_core import evaluate_rpn_bytetrack_diagnostic as evaluator
from src.trackocd_core.evaluation.physical_clip import score_route


def fixture_data():
    video = {"video_id": 4, "images": [{"image_id": i + 4, "frame_index": 750 + 30 * i} for i in range(4)]}
    data = {"image_id": np.arange(4, 8, dtype=np.int64), "frame_index": np.arange(750, 870, 30, dtype=np.int64),
            "det_offsets": np.arange(5, dtype=np.int64), "det_boxes": np.tile([0., 0., 20., 20.], (4, 1)).astype(np.float32),
            "det_score": np.full(4, .9, dtype=np.float32)}
    return video, data


def test_tracker_gets_current_boxes_scores_only_and_preserves_empty_frames():
    assert list(inspect.signature(bt.step).parameters) == ["tracker", "boxes_xyxy", "foreground_scores"]
    _, data = fixture_data(); digest = bt.detector_digest(data); tracker = bt.new_tracker(bt.PARAMETERS)
    first = bt.step(tracker, data["det_boxes"][:1], data["det_score"][:1])
    second = bt.step(tracker, data["det_boxes"][1:2], data["det_score"][1:2])
    assert first.shape == second.shape == (1, 6) and first[0, 4] == second[0, 4]
    assert first[0, 5] == data["det_score"][0]
    assert bt.step(tracker, np.empty((0, 4)), np.empty(0)).shape == (0, 6)
    assert tracker.frame_id == 3 and bt.detector_digest(data) == digest
    assert bt.new_tracker(bt.PARAMETERS).frame_id == 0


@pytest.mark.parametrize("key", ["track_thresh", "low_thresh", "match_thresh", "new_track_thresh", "track_buffer", "frame_rate", "min_box_area", "filter_mot_aspect"])
def test_no_threshold_window_or_filter_search(key):
    changed = dict(bt.PARAMETERS); changed[key] = not changed[key] if isinstance(changed[key], bool) else changed[key] + .01
    with pytest.raises(ValueError, match="no search"): bt.new_tracker(changed)


def test_nontrivial_exact_prefix_probe_changes_only_future_boxes():
    _, data = fixture_data(); before = bt.detector_digest(data)
    result = bt.causal_probe(data, bt.PARAMETERS)
    assert result["pass"] and result["frame_updates"] == 8
    original, changed = result["replays"]
    assert original[:2] == changed[:2] and all(row["tracks"] > 0 for row in original[:2])
    assert all(original[i]["input_sha256"] != changed[i]["input_sha256"] for i in (2, 3))
    assert bt.detector_digest(data) == before


def test_reader_only_accesses_five_raw_detector_fields(monkeypatch):
    video, data = fixture_data(); accessed = []
    class RestrictedNPZ:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key):
            accessed.append(key)
            assert key in data, "No native track ID/box/semantic access"
            return data[key]
    monkeypatch.setattr(bt.np, "load", lambda *args, **kw: RestrictedNPZ())
    loaded = bt.read_detector_video("unused", video)
    assert accessed == list(data) and bt.detector_digest(loaded) == bt.detector_digest(data)


@pytest.mark.parametrize("fault", ["metadata", "offset", "nan", "zero_area", "negative_score", "overscore"])
def test_invalid_raw_detections_rejected_without_repair(tmp_path, fault):
    video, data = fixture_data()
    if fault == "metadata": data["image_id"][0] = 99
    if fault == "offset": data["det_offsets"][2] = 0
    if fault == "nan": data["det_boxes"][0, 0] = np.nan
    if fault == "zero_area": data["det_boxes"][0, 2] = 0
    if fault == "negative_score": data["det_score"][0] = -.01
    if fault == "overscore": data["det_score"][0] = 1.01
    path = tmp_path / "raw.npz"; np.savez_compressed(path, **data)
    with pytest.raises(ValueError): bt.read_detector_video(path, video)


def test_classical_seal_no_fake_neural_state_and_no_overwrite(tmp_path):
    video, data = fixture_data(); (tmp_path / "shards").mkdir()
    arrays = {**data, "frame_offsets": data["det_offsets"].copy(), "boxes": data["det_boxes"].copy(),
              "score": data["det_score"].copy(), "track_id": np.ones(4, dtype=np.int64)}
    row = bt.seal_video(tmp_path, video, arrays, "config", "source", "tracker", {"empty_output_frames": 0})
    assert bt.completed_video(tmp_path, video, "config", "tracker") == row
    assert row["status"] == "COMPLETE_CLASSICAL_VIDEO" and row["source_npz_sha256"] == "source"
    assert not row["learned_model_weights"] and not row["nn_frozen_state_claim"]
    with pytest.raises(ValueError, match="overwrite"):
        bt.seal_video(tmp_path, video, arrays, "config", "source", "tracker", {})
    with pytest.raises(ValueError): bt.completed_video(tmp_path, video, "new-config", "tracker")
    with pytest.raises(ValueError): bt.completed_video(tmp_path, video, "config", "different-tracker")


def test_current_score_cannot_be_replaced_by_tracker():
    class BadTracker:
        def update(self, rows): return np.asarray([[0., 0., 20., 20., 1., .7501]])
    with pytest.raises(ValueError, match="score changed"):
        bt.step(BadTracker(), np.asarray([[0., 0., 20., 20.]]), np.asarray([.9]))


def test_actual_frozen_source_plan_without_replay_or_promotion():
    config, plan, files, digest = runner.load_plan()
    assert config["video_ids"] == [4, 20, 22, 23] and len(files) == 4 and len(digest) == 64
    assert sum(len(v["images"]) for v in plan["videos"]) == 64
    assert config["tracker_parameters"] == bt.PARAMETERS
    assert config["limits"]["total_real_frame_updates_with_causal_probe"] == 72
    assert not any(config[k] for k in ("primary_freeze_permitted", "scientific_pass_permitted", "training", "test_access", "pixel_inference_repeated"))
    assert all(set(bt.read_detector_video(files[v["video_id"]], v)) == set(config["detector_input_fields"]) for v in plan["videos"])


def test_cpu_worker_rejects_pixels_gt_weights_test_network_and_children():
    code = '''
from scripts.trackocd_core.run_rpn_bytetrack_diagnostic import input_barrier
import socket,subprocess,sys
input_barrier()
actions = [lambda: socket.create_connection(("127.0.0.1", 1)), lambda: subprocess.run(["true"]),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/annotations/validation.json"),
           lambda: open("/tmp/recovered_splits/roles.json"), lambda: open("/tmp/checkpoints/weight.pth"),
           lambda: open("/data3/liuyeqiang/TAO-Amodal/frames/test/forbidden.jpg")]
for action in actions:
    try: action()
    except PermissionError: pass
    else: raise AssertionError("Boundary not rejected")
assert "torch" not in sys.modules
print("6 rejections; no Torch")
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=runner.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6 rejections; no Torch"


def test_evaluator_gt_after_entire_prediction_and_source_seal():
    source = inspect.getsource(evaluator.main)
    access = source.index("json.loads(annotation.read_text())")
    for check in ("completed_video(RUN", 'supervisor["worker_returncode"]', 'marker["source_npz_sha256"]', '"Old frozen NPZ changed"'):
        assert source.index(check) < access
    scorer = inspect.getsource(score_route)
    assert 'SUBSET="all"' in scorer and "MAX_DETECTIONS=300" in scorer
    assert "observed_purity(purity_frames)" in scorer and "missing_or_unreliable" in scorer
    assert '"INCOMPARABLE_BASELINE_REPLAY_MISMATCH"' in source
    assert '"primary_freeze_permitted": False' in source and '"semantic_feedback": False' in source


def test_partial_or_completed_replay_cannot_be_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RUN", tmp_path)
    with pytest.raises(ValueError, match="Preserve"): runner.parent("irrelevant")
