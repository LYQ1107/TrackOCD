"""Complete fixed-route universe, seal-first evaluation and weighted composition."""
import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from src.trackocd_core.rpn_bytetrack import PARAMETERS, detector_digest
from src.trackocd_core.rpn_bytetrack_full import replay_video
from src.trackocd_core.evaluation.physical_full import compare_reference
from scripts.trackocd_core import rpn_bytetrack_full_runtime as runtime
from scripts.trackocd_core import run_rpn_bytetrack_full as predictor
from scripts.trackocd_core import evaluate_rpn_bytetrack_full as evaluator


def test_complete_video_replay_includes_empty_frames_and_no_source_transform():
    video = {"video_id": 4, "images": [{"image_id": i, "frame_index": i * 30} for i in (1, 2, 3)]}
    data = {"image_id": np.arange(1, 4), "frame_index": np.arange(30, 91, 30), "det_offsets": np.asarray([0, 1, 1, 2]),
            "det_boxes": np.asarray([[0., 0., 20., 20.]] * 2, dtype=np.float32), "det_score": np.full(2, .9, dtype=np.float32)}
    digest = detector_digest(data); calls = []
    arrays, stats = replay_video(data, video, PARAMETERS, lambda: calls.append(True))
    assert len(calls) == 3 and stats["empty_output_frames"] == 1
    assert len(stats["frames"]) == 3 and stats["frames"][1]["output_tracks"] == 0
    assert arrays["frame_offsets"].tolist() == [0, 1, 1, 2]
    assert detector_digest(data) == detector_digest(arrays) == digest
    assert np.array_equal(arrays["score"], data["det_score"])
    assert list(inspect.signature(replay_video).parameters) == ["data", "video", "parameters", "guard"]


def test_unchanged_existing_parameters_not_a_new_tunable_variant():
    video = {"images": []}; data = {k: np.empty(0) for k in ("image_id", "frame_index", "det_offsets", "det_boxes", "det_score")}
    with pytest.raises(ValueError, match="no search"):
        replay_video(data, video, {**PARAMETERS, "track_buffer": 31}, lambda: None)


def test_actual_entire_source_plan_not_novel_target_enriched():
    config, plan, records, digest = runtime.load_plan()
    assert config["videos"] == len(plan["videos"]) == len(records) == 988
    assert config["images"] == sum(len(v["images"]) for v in plan["videos"]) == 36375
    assert config["raw_detections"] == 1811677 and config["limits"]["total_frame_updates_with_probe"] == 36383
    assert config["tracker_parameters"] == PARAMETERS and len(digest) == 64
    assert config["evaluation"]["expected_known_gt_tracks"] == 4413 and config["evaluation"]["expected_novel_gt_tracks"] == 819
    assert not any(config[k] for k in ("training", "test_access", "pixel_inference_repeated", "parameter_search", "primary_freeze_permitted", "scientific_pass_permitted", "semantic_feedback"))
    assert config["limits"]["cpu_workers"] == 1 and config["new_weights_data_environment_bytes"] == 0
    assert not config["provenance"]["exact_private_sam_release_training_stage_binding_verified"]
    assert runtime.verified_source(plan["videos"][0], config, records).is_file()


def test_evaluator_gt_only_after_successful_full_prediction_and_all_baselines():
    source = inspect.getsource(evaluator.worker); access = source.index("json.loads(annotation.read_text())")
    for boundary in ("verify_prediction(config", "detector_digest(read_detector_video", '"Do not rerun/replace existing complete baseline source"'):
        assert source.index(boundary) < access
    check = inspect.getsource(runtime.verify_prediction)
    assert 'supervisor["worker_returncode"] != 0' in check and 'prediction["production_frame_updates"] != config["images"]' in check
    assert 'prediction["total_frame_updates"] != config["limits"]["total_frame_updates_with_probe"]' in check
    assert '"INCOMPARABLE_FULL_BASELINE_REPLAY_MISMATCH"' in source and '"primary_freeze_permitted": False' in source
    assert "TemporaryDirectory" in source and "score_video" in source  # No whole-Val scratch copy.
    assert "input_barrier()" in inspect.getsource(predictor.worker)


def test_complete_or_partial_run_not_automatically_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUN", tmp_path)
    with pytest.raises(ValueError, match="Preserve"): runtime.supervise("prediction", Path("irrelevant"), "irrelevant")


def test_owned_storage_snapshot_tolerates_disappearing_own_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUN", tmp_path)
    monkeypatch.setattr(runtime.os, "scandir", lambda path: (_ for _ in ()).throw(FileNotFoundError()))
    assert runtime.owned_output_bytes() == tmp_path.stat().st_blocks * 512


def test_reference_comparison_rejects_filtered_unknowns_or_missing_video():
    reference = {"canonical_tracking": {k: .2 for k in ("HOTA", "AssA", "DetA", "DetRe")}, "coverage": {}, "purity": {"unknown": 3},
                 "annotated_projection_lengths": {}, "raw_prediction_rows": 4, "canonical_evaluated_prediction_rows": 4,
                 "per_video": [{"video_id": 4, "images": 1, "gt_rows": 1, "raw_prediction_rows": 4, "canonical_evaluated_prediction_rows": 4,
                                "coverage": {}, "purity": {"unknown": 3}, "source_npz_bytes": 1, "source_npz_sha256": "same",
                                "canonical_tracking": {k: .2 for k in ("HOTA", "AssA", "DetA", "DetRe")}}]}
    assert compare_reference(reference, reference, 1e-12)["pass"]
    changed = copy.deepcopy(reference); changed["purity"]["unknown"] = 0
    assert not compare_reference(changed, reference, 1e-12)["pass"]
    changed = copy.deepcopy(reference); changed["per_video"] = []
    assert not compare_reference(changed, reference, 1e-12)["pass"]


def test_legacy_gt_target_insertion_order_retained_even_if_not_chronological(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.trackocd_core.evaluation import physical_full
    video = {"video_id": 4, "images": [{"image_id": i, "frame_index": i} for i in (41, 42)]}
    subset = {"annotations": [{"image_id": 42, "track_id": 9, "category_id": 2, "bbox": [0, 0, 20, 20]},
                              {"image_id": 41, "track_id": 4, "category_id": 1, "bbox": [0, 0, 20, 20]}]}
    frames = {i: (np.empty(0, dtype=np.int64), np.empty((0, 4)), np.empty(0)) for i in (41, 42)}
    monkeypatch.setattr(physical_full, "read_projection", lambda *args: frames)
    captured = []
    def matching(targets, predictions):
        captured.extend(t["key"] for t in targets)
        return {"matches": {}}
    monkeypatch.setattr(physical_full, "match_video", matching)
    class Dataset:
        seq_list = ["fixture"]
        @staticmethod
        def get_default_dataset_config(): return {}
        def __init__(self, options): pass
        def get_raw_seq_data(self, *args): return {}
        def get_preprocessed_seq_data(self, *args): return {"num_gt_dets": 2, "num_tracker_dets": 0}
    class Metric:
        def eval_sequence(self, data): return {k: np.zeros(19) for k in ("HOTA", "AssA", "DetA", "DetRe")}
    trackeval = SimpleNamespace(datasets=SimpleNamespace(TAO_OW=Dataset), metrics=SimpleNamespace(HOTA=Metric))
    source = tmp_path / "fixture.npz"; source.write_bytes(b"fixture")
    physical_full.score_video("fixture", source, video, subset, {1}, {2}, tmp_path, tmp_path / "scratch", trackeval, lambda: None)
    assert captured == ["4_9", "4_4"]  # Legacy annotation order, not first-observation order.


def test_count_weighted_per_video_scorer_equals_complete_canonical_including_empty(tmp_path):
    # Synthetic only; no real annotation/role reads, inference or model inputs.
    code = '''
from pathlib import Path
from collections import Counter
import copy,sys,numpy as np
from src.trackocd_v2.io import atomic_json
from src.trackocd_core.evaluation.physical_clip import score_route
from src.trackocd_core.evaluation.physical_full import score_video,combine_route
from scripts.trackocd_core.evaluate_masa_physical_qualification import CANONICAL
np.float=float; np.int=int; sys.path.insert(0,str(CANONICAL)); import trackeval
root=Path(sys.argv[1]); plan={"videos":[{"video_id":v,"images":[{"image_id":v*10+i,"frame_index":i} for i in (1,2)]} for v in (4,9)]}
gt={"videos":[{"id":v,"name":f"v{v}","neg_category_ids":[],"not_exhaustive_category_ids":[]} for v in (4,9)],
"categories":[{"id":1,"name":"object"}],"tracks":[{"id":v,"video_id":v,"category_id":1} for v in (4,9)],
"images":[{"id":v*10+i,"video_id":v,"frame_index":i} for v in (4,9) for i in (1,2)],
"annotations":[{"id":v*10+i,"video_id":v,"image_id":v*10+i,"track_id":v,"category_id":1,"bbox":[0,0,20,20],"area":400} for v in (4,9) for i in (1,2)]}
files={}
for v in (4,9):
 count=2 if v==4 else 0; files[v]=root/f"{v}.npz"
 np.savez_compressed(files[v],image_id=np.asarray([v*10+1,v*10+2]),frame_offsets=np.asarray([0,count//2,count]),track_id=np.zeros(count,dtype=np.int64),boxes=np.tile([0.,0.,20.,20.],(count,1)).reshape(-1,4),score=np.ones(count))
g=root/"allgt"; g.mkdir(); atomic_json(g/"validation.json",gt)
allscore=score_route("fixture",files,plan,gt,{1},set(),g,root/"all",trackeval,lambda:None)
seq={}; summaries=[]; hist=Counter()
for video in plan["videos"]:
 v=video["video_id"]; subset=copy.deepcopy(gt)
 for field in ("videos","images","tracks","annotations"):
  subset[field]=[r for r in subset[field] if (r["id"] if field=="videos" else r["video_id"])==v]
 g=root/str(v)/"gt"; g.mkdir(parents=True); atomic_json(g/"validation.json",subset)
 scores,summary,histogram=score_video("fixture",files[v],video,subset,{1},set(),g,root/str(v),trackeval,lambda:None)
 seq[str(v)]=scores; summaries.append(summary); hist.update(histogram)
combined=combine_route(trackeval.metrics.HOTA(),seq,summaries,hist)
for k,v in combined["canonical_raw_combined"].items():
 assert np.allclose(v,allscore["canonical_raw_combined"][k],atol=1e-12,rtol=0),k
assert combined["coverage"]["known"]["gt_tracks"]==allscore["coverage"]["known"]["gt_clip_tracks"]==2
assert combined["purity"]["unmatched_unknown_rows"]==allscore["purity"]["unmatched_unknown_rows"]
print("weighted composition pass")
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=runtime.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("weighted composition pass")
