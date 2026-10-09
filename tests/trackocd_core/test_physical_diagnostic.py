"""Sampling and post-hoc mixing tests; no real data/model/network execution."""
import numpy as np
import pytest
import json
from pathlib import Path
import subprocess
import sys
from src.trackocd_core.physical_diagnostic import metadata_prefixes, observed_purity, pairwise_iou
from scripts.trackocd_core.audit_masa_training_route import inspect_route


def test_sampling_does_not_access_labels_or_skip_short_videos():
    class LabelsForbidden(dict):
        def __getitem__(self, key):
            assert key not in {"annotations", "categories", "tracks"}
            return super().__getitem__(key)
    annotation = LabelsForbidden(videos=[{"id": 8}, {"id": 2}], images=[
        {"id": 3, "video_id": 8, "frame_index": 2, "file_name": "val/v8/2.jpg"},
        {"id": 2, "video_id": 2, "frame_index": 5, "file_name": "val/v2/5.jpg"},
        {"id": 1, "video_id": 8, "frame_index": 1, "file_name": "val/v8/1.jpg"}])
    rows = metadata_prefixes(annotation, videos=2, frames=16)
    assert [r["video_id"] for r in rows] == [2, 8]
    assert [i["image_id"] for i in rows[1]["images"]] == [1, 3]
    assert len(rows[0]["images"]) == 1


def frame(gt_id=1, category=10, matched=True):
    return {"gt_ids": [gt_id], "gt_categories": [category], "gt_boxes": [[0, 0, 2, 2]],
            "pred_ids": [0], "pred_boxes": [[0, 0, 2, 2]] if matched else [[4, 4, 5, 5]]}


def test_same_category_different_individual_is_not_semantic_mix():
    r = observed_purity([frame(), frame(gt_id=2)])
    assert r["observed_multiple_gt_identity_tracks"] == 1
    assert r["observed_multicategory_tracks"] == 0


def test_different_categories_poison_observed_physical_track():
    r = observed_purity([frame(), frame(gt_id=2, category=20)])
    assert r["observed_multicategory_tracks"] == 1
    assert r["observed_majority_category_fraction_over_matched_rows"] == .5


def test_unknown_unmatched_observation_prevents_whole_observed_track_purity():
    r = observed_purity([frame(), frame(matched=False)])
    assert r["unmatched_unknown_rows"] == 1
    assert r["entire_observed_track_matched_to_one_category"] == 0


def test_empty_and_duplicate_predictions_are_explicit():
    assert observed_purity([])["observed_majority_category_fraction_over_matched_rows"] is None
    row = frame(); row["pred_ids"] = [0, 0]; row["pred_boxes"] *= 2
    with pytest.raises(ValueError):
        observed_purity([row])
    assert pairwise_iou([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_published_route_decode_does_not_execute_training_code():
    config = "raise RuntimeError('not run')\ndataset_type='MASADataset'\nimg_scale=(1024,1024)\ntrain_pipeline=[]\ntrain_dataloader=dict(dataset=dict(dataset=dict(fixed_length=200000,datasets=[dict(type=dataset_type,ann_file='data/sam/sam_annotations/jsons/sa1b_coco_fmt_500k_bbox_anno.json',data_prefix=dict(img='data/sam/batch0/'))])))"
    converter = "def process_file(x):\n a={'category_id':1}\n b={'category_id':1}\n"
    dataset = "class MASADataset:\n METAINFO={'classes':'object'}\n"
    r = inspect_route(config, converter, dataset)
    assert r["all_selected_leaf_paths_are_sa1b"] and r["converter_assigns_one_generic_category"]


def test_real_preregistration_has_fixed_label_blind_sampling_and_no_primary_freeze():
    root=Path(__file__).resolve().parents[2]
    p=json.loads((root/"configs/trackocd_core/masa_physical_diagnostic.json").read_text())
    assert p["video_ids"]==[4,20,22,23] and p["images_per_video"]==[16]*4
    assert p["limits"]["max_images"]==64 and p["limits"]["max_videos"]==4
    assert not p["new_weight_or_training_or_threshold_search"]
    assert not p["evaluation"]["categories_for_input_or_tuning"]
    assert not p["evaluation"]["full_val_or_ocd_metrics_claim"]
    r=json.loads((root/"outputs/trackocd_core/audit/masa_training_route.json").read_text())
    assert r["source_bytes"]==32848 and len(r["source_files"])==6
    assert not r["exact_checkpoint_stage_config_supervision_binding_verified"]
    assert not r["forbidden_supervision_proven"] and not r["primary_freeze_permitted_by_this_audit"]
    assert not r["sam_training_config_paths_in_complete_pinned_tree"]


def test_unchanged_canonical_tao_adapter_perfect_fixture_in_fresh_process(tmp_path):
    canonical=Path("/data3/liuyeqiang/InterMOT/third_party/MOTIP/TrackEval")
    if not canonical.exists():
        pytest.skip("Private canonical external evaluator asset not present")
    gt=tmp_path/"gt"; pred=tmp_path/"pred"/"fixture"/"data"
    gt.mkdir(); pred.mkdir(parents=True)
    data={"videos":[{"id":4,"name":"anonymous_fixture","neg_category_ids":[],"not_exhaustive_category_ids":[]}],
          "categories":[{"id":1,"name":"object"}],"tracks":[{"id":1,"video_id":4,"category_id":1}],
          "images":[{"id":i,"video_id":4,"frame_index":i} for i in range(1,4)],
          "annotations":[{"id":i,"video_id":4,"image_id":i,"track_id":1,"category_id":1,"bbox":[0,0,2,2],"area":4} for i in range(1,4)]}
    (gt/"validation.json").write_text(json.dumps(data))
    rows=[{"image_id":i,"video_id":4,"track_id":0,"category_id":1,"bbox":[0,0,2,2],"score":1.} for i in range(1,4)]
    (pred/"pred.json").write_text(json.dumps(rows))
    code="""
import sys,numpy as np,json
np.float=float; np.int=int
sys.path.insert(0,sys.argv[1])
import trackeval
c=trackeval.datasets.TAO_OW.get_default_dataset_config()
c.update(GT_FOLDER=sys.argv[2],TRACKERS_FOLDER=sys.argv[3],TRACKERS_TO_EVAL=['fixture'],SUBSET='all',SPLIT_TO_EVAL='val',PRINT_CONFIG=False)
d=trackeval.datasets.TAO_OW(c)
raw=d.get_raw_seq_data('fixture',d.seq_list[0]); data=d.get_preprocessed_seq_data(raw,'object')
r=trackeval.metrics.HOTA().eval_sequence(data)
assert all(np.allclose(r[k],1.) for k in ['HOTA','AssA','DetA','DetRe']),r
"""
    result=subprocess.run([sys.executable,"-c",code,str(canonical),str(gt),str(tmp_path/"pred")],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
