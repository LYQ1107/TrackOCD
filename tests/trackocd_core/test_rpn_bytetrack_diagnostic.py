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


def test_actual_cpu_prediction_sealed_without_new_pixels_scores_or_torch():
    config, plan, files, digest = runner.load_plan()
    prediction = json.loads((runner.RUN / "prediction_manifest.json").read_text())
    supervisor = json.loads((runner.RUN / "supervisor.json").read_text())
    assert prediction["status"] == "SEALED_BOUNDED_RPN_BYTETRACK_NOT_QUALIFICATION"
    assert prediction["production_frame_updates"] == 64 and prediction["total_frame_updates"] == 72
    assert prediction["causality_probe"]["pass"] and prediction["causal_probe_frame_updates"] == 8
    assert not prediction["torch_imported"] and prediction["input_arrays_unchanged"] and prediction["tracker_source_unchanged"]
    assert not any(prediction["boundary"].values()) and not prediction["primary_freeze_permitted"]
    assert supervisor["worker_returncode"] == 0 and supervisor["error"] is None and not supervisor["gpu_used"]
    assert prediction["peak_host_rss_bytes"] <= config["limits"]["max_host_rss_bytes"]
    assert prediction["elapsed_seconds"] <= config["limits"]["max_seconds"]
    seals = [bt.completed_video(runner.RUN, v, digest, config["tracker_source_sha256"]) for v in plan["videos"]]
    assert seals == prediction["videos"] and sum(s["prediction_rows"] for s in seals) == 1475
    assert sum(s["detection_rows"] for s in seals) == 3200 and sum(s["npz_bytes"] for s in seals) == 91943
    for video, seal in zip(plan["videos"], seals):
        assert seal["source_detector_arrays_sha256"] == bt.detector_digest(bt.read_detector_video(files[video["video_id"]], video))
        assert not seal["learned_model_weights"] and not seal["nn_frozen_state_claim"]


def test_actual_four_route_comparison_no_unknown_filter_or_primary_promotion():
    result = json.loads(evaluator.SUMMARY.read_text())
    assert result["status"] == "BOUNDED_RPN_BYTETRACK_COMPLETE_NOT_PRIMARY_QUALIFICATION"
    assert result["selected_images"] == 64 and result["selected_gt_rows"] == 328
    assert result["historical_baselines_reproduced"]
    assert all(v == 0 for row in result["baseline_absolute_metric_differences"].values() for v in row.values())
    assert not any(result[k] for k in ("primary_freeze_permitted", "scientific_pass_permitted", "labels_for_model_input_or_tuning", "ocd_or_m9_metrics", "semantic_feedback"))
    route = result["results"]["RPN_BYTETRACK"]
    assert route["canonical_tracking"]["HOTA"] == .15107958129314517
    assert route["raw_prediction_rows"] == route["canonical_evaluated_prediction_rows"] == 1475
    assert route["canonical_preprocessing_removed_rows"] == 0
    assert route["coverage"]["known"]["reliably_observed"] == 2 and route["coverage"]["known"]["gt_clip_tracks"] == 26
    assert route["coverage"]["novel"]["reliably_observed"] == 1 and route["coverage"]["novel"]["gt_clip_tracks"] == 2
    assert route["annotated_projection_lengths"]["physical_tracks"] == 223
    assert route["annotated_projection_lengths"]["single_observation_tracks"] == 40
    assert route["purity"]["matched_rows"] == 85 and route["purity"]["unmatched_unknown_rows"] == 1390
    assert route["purity"]["tracks_with_unknown_observations"] == 219
    assert route["purity"]["entire_observed_track_matched_to_one_category"] == 4
    assert route["purity"]["observed_multiple_gt_identity_tracks"] == 3
    assert result["resources"]["cpu_workers"] == 1 and not result["resources"]["gpu_used"]


def test_full_cache_readonly_preflight_is_not_association_or_qualification():
    from scripts.trackocd_core.audit_rpn_bytetrack_cache import worker
    from src.trackocd_v2.io import sha256_file
    row = json.loads((runner.ROOT / "outputs/trackocd_core/audit/rpn_bytetrack_full_cache_preflight.json").read_text())
    assert sha256_file(runner.ROOT / row["script"]) == row["script_sha256"]
    assert row["videos"] == 988 and row["images"] == 36375 and row["raw_detections"] == 1811677
    assert sum(row["score_histogram_width_0_05"]) == row["raw_detections"]
    assert row["score_min"] >= 0 and row["score_max"] <= 1
    assert row["peak_rss_bytes"] <= row["host_rss_limit_bytes"] == 256 * 1024 * 1024
    assert row["seconds"] <= row["wall_limit_seconds"] and row["fresh_single_cpu_worker"]
    assert not any(row[k] for k in ("source_mutated", "gt_pixels_weights_test_read", "association_or_model_executed", "full_replay_started", "primary_freeze_permitted", "scientific_pass_permitted"))
    source = inspect.getsource(worker)
    assert "input_barrier()" in source and "completed_video(run,v,digest)" in source
    assert "sha256_file(path) == row[\"npz_sha256\"]" in source
    assert "sha256_file(mpath) == manifest_before" in source
    assert "new_tracker(" not in source and "step(" not in source
