"""Tests of geometric suppression, metadata invariance and frame lifecycle."""
import inspect

import numpy as np
import pytest
import torch

from src.baselines.pandas_bytetrack.tracking_postprocess import prepare_tracking_detections, track_video
from src.iclr27_phase3b.bytetrack import BYTETracker


def fixture():
    return (np.array([[0,0,20,20], [0,0,20,20], [40,0,60,20], [80,0,100,20]], np.float32),
            np.array([.9,.9,.95,.8], np.float32),
            np.array([.01,.5,.1,.99], np.float32), np.array([7,500,200,3], np.int32))


def test_duplicate_prototypes_suppressed_and_distinct_locations_preserved():
    d = prepare_tracking_detections(*fixture(), nms_iou=.7, max_detections=50)
    assert d["original_detection_index"].tolist() == [2,0,3]
    assert d["prototype_ids"].tolist() == [200,7,3]
    assert np.all(np.diff(d["foreground_scores"]) <= 0)


def test_metadata_never_changes_selection_or_inputs():
    data = fixture(); original = [v.copy() for v in data]
    a = prepare_tracking_detections(*data, nms_iou=.5, max_detections=2)
    b = prepare_tracking_detections(data[0],data[1],data[2][::-1],data[3][::-1], nms_iou=.5,max_detections=2)
    assert np.array_equal(a["original_detection_index"],b["original_detection_index"])
    for v,o in zip(data,original):
        assert np.array_equal(v,o)
    assert np.array_equal(a["semantic_scores"], data[2][a["original_detection_index"]])
    assert np.array_equal(a["prototype_ids"], data[3][a["original_detection_index"]])
    a["semantic_scores"][:] = 0
    assert np.array_equal(data[2], original[2])


def test_empty_frame_and_explicit_source_offsets():
    b,fg,sem,pro = fixture()
    data = {"frame_index":np.arange(4), "image_id":np.array([100,-1,-1,101]),
            "frame_offsets":np.array([0,4,4,4,8]), "boxes":np.vstack([b,b]),
            "foreground_scores":np.tile(fg,2),"scores":np.tile(sem,2),"prototype_id":np.tile(pro,2)}
    tracker = BYTETracker(filter_mot_aspect=False)
    calls = []
    orig_update = tracker.update
    def count_calls(rows):
        calls.append(len(rows)); return orig_update(rows)
    tracker.update = count_calls
    result,counts,_ = track_video(data,tracker,nms_iou=.7,max_detections=50)
    assert calls == [3,0,0,3] and tracker.frame_id == 4
    assert counts.tolist() == calls
    assert np.array_equal(result["frame_index"],data["frame_index"])
    assert np.array_equal(result["image_id"],data["image_id"])
    assert np.array_equal(result["source_frame_offsets"],data["frame_offsets"])
    assert result["candidate_frame_offsets"].tolist() == [0,3,3,3,6]
    assert result["candidate_original_detection_index"].tolist() == [2,0,3,6,4,7]
    assert result["frame_offsets"][1] == result["frame_offsets"][2] == result["frame_offsets"][3]


def test_no_gt_or_novel_text_inputs():
    assert list(inspect.signature(prepare_tracking_detections).parameters) == [
        "boxes_xyxy","foreground_scores","semantic_scores","prototype_ids","nms_iou","max_detections"]
    with pytest.raises(TypeError):
        prepare_tracking_detections(*fixture(),nms_iou=.7,max_detections=50,gt_label=1)


def test_zero_detections_and_validation():
    d = prepare_tracking_detections(np.empty((0,4),np.float32),np.empty(0),np.empty(0),
                                    np.empty(0,np.int32),nms_iou=.7,max_detections=50)
    assert d["boxes"].shape == (0,4)
    b,fg,sem,pro = fixture()
    with pytest.raises(ValueError):
        prepare_tracking_detections(b,fg[:-1],sem,pro,nms_iou=.7,max_detections=50)


def test_score_order_and_ties_repeat_deterministically():
    results = [prepare_tracking_detections(*fixture(),nms_iou=.5,max_detections=50) for _ in range(5)]
    assert all(r["original_detection_index"].tolist() == [2,0,3] for r in results)
    assert results[0]["foreground_scores"].tolist() == fixture()[1][[2,0,3]].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_consistency():
    values = fixture()
    cpu = prepare_tracking_detections(*values,nms_iou=.7,max_detections=50)
    gpu = prepare_tracking_detections(*(torch.as_tensor(x,device="cuda") for x in values),
                                      nms_iou=.7,max_detections=50)
    for k in cpu:
        np.testing.assert_array_equal(cpu[k],gpu[k])
