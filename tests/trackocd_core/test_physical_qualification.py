"""Entire metadata universe, no-repair atomic shards and canonical composition."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from src.trackocd_core.physical_qualification import metadata_universe, val_image, validate_arrays, seal_video, completed_video, aggregate_purity
from src.trackocd_core.physical_diagnostic import observed_purity


def video():
    return {"video_id": 4, "images": [{"image_id": i, "frame_index": i, "image_path": f"val/v/{i}.jpg"} for i in (1, 2)]}


def arrays(empty=False):
    n = 0 if empty else 2
    return {"image_id": np.array([1, 2]), "frame_index": np.array([1, 2]),
            "frame_offsets": np.array([0, n // 2, n]), "det_offsets": np.array([0, n // 2, n]),
            "track_id": np.zeros(n, dtype=np.int64), "boxes": np.tile([0., 0., 2., 2.], (n, 1)).astype(np.float32),
            "score": np.ones(n, dtype=np.float32), "det_boxes": np.tile([0., 0., 2., 2.], (n, 1)).astype(np.float32),
            "det_score": np.ones(n, dtype=np.float32)}


def test_all_metadata_not_label_selected_and_chronology():
    class NoLabels(dict):
        def __getitem__(self, key):
            assert key not in {"annotations", "categories", "tracks"}
            return super().__getitem__(key)
    data = NoLabels(videos=[{"id": 9}, {"id": 4}], images=[
        {"id": 2, "video_id": 4, "frame_index": 2, "file_name": "val/v/2.jpg"},
        {"id": 3, "video_id": 9, "frame_index": 1, "file_name": "val/w/1.jpg"},
        {"id": 1, "video_id": 4, "frame_index": 1, "file_name": "val/v/1.jpg"}])
    rows = metadata_universe(data)
    assert [v["video_id"] for v in rows] == [4, 9]
    assert [i["image_id"] for i in rows[0]["images"]] == [1, 2]
    assert sum(len(v["images"]) for v in rows) == 3


@pytest.mark.parametrize("change", ["empty", "duplicate_video", "duplicate_image", "duplicate_frame", "unknown_video"])
def test_corrupt_metadata_never_shrinks_universe(change):
    data = {"videos": [{"id": 4}], "images": [{"id": i, "video_id": 4, "frame_index": i, "file_name": f"val/v/{i}.jpg"} for i in (1, 2)]}
    if change == "empty": data["videos"].append({"id": 9})
    if change == "duplicate_video": data["videos"].append({"id": 4})
    if change == "duplicate_image": data["images"][1]["id"] = 1
    if change == "duplicate_frame": data["images"][1]["frame_index"] = 1
    if change == "unknown_video": data["images"][1]["video_id"] = 9
    with pytest.raises(ValueError): metadata_universe(data)


def test_missing_escape_or_other_split_not_substituted(tmp_path):
    (tmp_path / "val").mkdir(); (tmp_path / "val" / "1.jpg").write_bytes(b"image")
    assert val_image(tmp_path, "val/1.jpg").is_file()
    for relative in ("test/1.jpg", "train/1.jpg", "val/missing.jpg", "../val/1.jpg", "/val/1.jpg"):
        with pytest.raises(ValueError): val_image(tmp_path, relative)
    (tmp_path / "val" / "escape.jpg").symlink_to(tmp_path / "foreign.jpg")
    (tmp_path / "foreign.jpg").write_bytes(b"x")
    with pytest.raises(ValueError): val_image(tmp_path, "val/escape.jpg")


@pytest.mark.parametrize("empty", [False, True])
def test_completed_atomic_video_reused_even_if_empty_without_reinference(tmp_path, empty):
    (tmp_path / "shards").mkdir()
    a = arrays(empty)
    r = seal_video(tmp_path, video(), a, "a" * 64, "owned_attempt", {"frozen_state_unchanged": True})
    assert completed_video(tmp_path, video(), "a" * 64) == r
    assert r["prediction_rows"] == (0 if empty else 2)
    with pytest.raises(ValueError): seal_video(tmp_path, video(), a, "a" * 64, "another", {"frozen_state_unchanged": True})
    with pytest.raises(ValueError): completed_video(tmp_path, video(), "b" * 64)
    altered = video(); altered["images"][0]["image_sha256"] = "b" * 64
    with pytest.raises(ValueError): completed_video(tmp_path, altered, "a" * 64)
    path = tmp_path / "shards" / r["npz_filename"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError): completed_video(tmp_path, video(), "a" * 64)


@pytest.mark.parametrize("kind", ["nan", "score", "degenerate", "missing_image", "decreasing_offsets", "negative_id", "float_id"])
def test_invalid_outputs_fail_without_repair(kind):
    a = arrays()
    if kind == "nan": a["boxes"][0, 0] = np.nan
    if kind == "score": a["score"][0] = 1.1
    if kind == "degenerate": a["boxes"][0, 2] = 0
    if kind == "missing_image": a["image_id"][0] = 3
    if kind == "decreasing_offsets": a["frame_offsets"] = np.array([0, 3, 2], dtype=np.uint64)
    if kind == "negative_id": a["track_id"][0] = -1
    if kind == "float_id": a["track_id"] = a["track_id"].astype(float)
    with pytest.raises(ValueError): validate_arrays(a, video())


def test_unsealed_attempt_not_misread_as_valid_shard(tmp_path):
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards" / "video_0004.interrupted.npz").write_bytes(b"partial")
    assert completed_video(tmp_path, video(), "a" * 64) is None
    with pytest.raises(ValueError): seal_video(tmp_path, video(), arrays(), "a" * 64, "../escape", {"frozen_state_unchanged": True})
    with pytest.raises(ValueError): seal_video(tmp_path, video(), arrays(), "a" * 64, "owned", {"frozen_state_unchanged": False})


def test_purity_combination_preserves_video_local_identity_and_unknown_denominator():
    first = observed_purity([{"pred_ids": [0], "pred_boxes": [[0, 0, 2, 2]], "gt_ids": [1], "gt_categories": [10], "gt_boxes": [[0, 0, 2, 2]]}])
    second = observed_purity([{"pred_ids": [0], "pred_boxes": [[0, 0, 2, 2]], "gt_ids": [], "gt_categories": [], "gt_boxes": []}])
    r = aggregate_purity([first, second])
    assert r["physical_tracks"] == 2 and r["prediction_rows"] == 2
    assert r["matched_rows"] == 1 and r["unmatched_unknown_rows"] == 1
    assert r["observed_majority_category_fraction_over_matched_rows"] == 1
    assert r["entire_observed_track_matched_to_one_category"] == 1


def test_per_video_canonical_composition_equals_full_adapter_with_empty_predictions(tmp_path):
    canonical = Path("/data3/liuyeqiang/InterMOT/third_party/MOTIP/TrackEval")
    if not canonical.exists(): pytest.skip("Canonical evaluator asset absent")
    gt = {"videos": [{"id": v, "name": f"v{v}", "neg_category_ids": [], "not_exhaustive_category_ids": []} for v in (4, 9)],
          "categories": [{"id": 1, "name": "object"}], "tracks": [{"id": v, "video_id": v, "category_id": 1} for v in (4, 9)],
          "images": [{"id": v * 10 + i, "video_id": v, "frame_index": i} for v in (4, 9) for i in (1, 2)],
          "annotations": [{"id": v * 10 + i, "video_id": v, "image_id": v * 10 + i, "track_id": v, "category_id": 1, "bbox": [0, 0, 2, 2], "area": 4} for v in (4, 9) for i in (1, 2)]}
    pred = [{"video_id": 4, "image_id": 40 + i, "track_id": 0, "category_id": 1, "bbox": [0, 0, 2, 2], "score": 1.} for i in (1, 2)]
    for suffix, vids in (("all", {4, 9}), ("4", {4}), ("9", {9})):
        g = tmp_path / suffix / "gt"; p = tmp_path / suffix / "pred" / "fixture" / "data"; g.mkdir(parents=True); p.mkdir(parents=True)
        data = copy.deepcopy(gt)
        for field in ("videos", "images", "tracks", "annotations"):
            data[field] = [r for r in data[field] if (r["id"] if field == "videos" else r["video_id"]) in vids]
        (g / "validation.json").write_text(json.dumps(data)); (p / "pred.json").write_text(json.dumps([r for r in pred if r["video_id"] in vids]))
    code = """
import sys,numpy as np
from pathlib import Path
np.float=float; np.int=int
sys.path.insert(0,sys.argv[1]); import trackeval
root=Path(sys.argv[2]); metric=trackeval.metrics.HOTA()
def run(suffix):
 c=trackeval.datasets.TAO_OW.get_default_dataset_config()
 c.update(GT_FOLDER=str(root/suffix/'gt'),TRACKERS_FOLDER=str(root/suffix/'pred'),TRACKERS_TO_EVAL=['fixture'],SUBSET='all',SPLIT_TO_EVAL='val',PRINT_CONFIG=False)
 d=trackeval.datasets.TAO_OW(c)
 return {s:metric.eval_sequence(d.get_preprocessed_seq_data(d.get_raw_seq_data('fixture',s),'object')) for s in d.seq_list}
full=metric.combine_sequences(run('all')); separate=metric.combine_sequences({**run('4'),**run('9')})
for key in full: assert np.allclose(full[key],separate[key]),key
"""
    r = subprocess.run([sys.executable, "-c", code, str(canonical), str(tmp_path)], capture_output=True, text=True, timeout=25)
    assert r.returncode == 0, r.stderr


def test_registered_full_scope_not_m9_and_no_tuning_or_automatic_primary():
    path = Path(__file__).resolve().parents[2] / "configs/trackocd_core/masa_physical_qualification.json"
    if not path.exists(): pytest.skip("Plan not generated yet")
    p = json.loads(path.read_text())
    assert p["videos"] == 988 and p["images"] == 36375 and sum(p["image_counts"].values()) == 36375
    assert p["candidate_model_unchanged_from_completed_smoke_and_val4"] and p["roi_cap"] == 50
    assert not p["new_weight_or_training_or_threshold_search"] and not p["evaluation"]["automatic_primary_freeze"]
    assert not p["evaluation"]["labels_for_model_input_or_tuning"]
    assert p["asset_inventory"]["new_image_copy_bytes"] == p["asset_inventory"]["new_weight_bytes"] == 0
    assert not p["provenance"]["exact_private_sam_release_training_stage_binding_verified"]


def test_actual_full_val_prediction_receipt_has_unique_complete_universe_and_no_primary_claim():
    path = Path(__file__).resolve().parents[2] / "outputs/trackocd_core/audit/masa_full_val_prediction_summary.json"
    if not path.exists(): pytest.skip("Completed real physical prediction receipt absent")
    p = json.loads(path.read_text())
    assert p["status"] == "SEALED_COMPLETE_FULL_VAL_PHYSICAL_STREAM" and p["videos"] == 988 and p["images"] == 36375
    assert p["prediction_rows"] == 1540022 and p["detection_rows"] == 1811677 and p["compressed_npz_bytes"] == 63222355
    assert p["frame_statistics"]["frames"] == 36375 and p["all_complete_video_states_unchanged"]
    assert not any(p["boundary"].values()) and not p["independent_gt_evaluation_complete"]
    assert p["frozen_model"]["strict_state_tensor_keys"] == 419 and p["frozen_model"]["all_parameters_frozen"]
    assert p["supervisor"]["worker_returncodes"] == [0] * 4 and p["supervisor"]["remote_preregistration_verified"]
    workers = p["worker_resources"]
    assert sum(w["actual_forwards_completed_this_attempt"] for w in workers) == 36375
    assert all(w["initial_frozen_state_sha256"] == w["final_frozen_state_sha256"] for w in workers)
    assert len({w["initial_frozen_state_sha256"] for w in workers}) == 1
