from src.iclr27_phase74r.physical_index import PhysicalIndex
from src.iclr27_phase74r.tracklet_alignment import TrackletAligner


def _event_asset(key):
    return {"content_asset_key": key}


def _row(image, frame, box="[10,10,40,40]"):
    return {"row_key": f"1:{frame}:0:2:{image}", "image_id": image, "frame_id": frame, "event_rank": frame, "bbox_xyxy": box, "assigned": "1", "row_iou": "0.8"}


def test_joint_reliability_and_ambiguous_overlap():
    key = "tao|Scene/clip|frame=1"
    assets = {10: _event_asset(key)}
    rows = [{"bbox": [10, 10, 30, 30], "image_id": 100, "video_id": 4, "track_id": 1, "candidate_order": 0}, {"bbox": [10, 10, 30, 30], "image_id": 101, "video_id": 4, "track_id": 2, "candidate_order": 1}]
    out = TrackletAligner(assets, PhysicalIndex({key: rows})).align({"event_key": "e", "fold": 0, "kind": "positive"}, "target", "v1:p1", 1, [_row(10, 1)])
    assert out["mapping_classification"] == "AMBIGUOUS_OVERLAP"
    assert all(x["joint_reliable"] for v in out["candidate_physical_tracks"].values() for x in v["rows"])


def test_nonoverlap_tracks_are_fragmentation():
    k1, k2 = "tao|Scene/clip|frame=1", "tao|Scene/clip|frame=2"
    assets = {10: _event_asset(k1), 11: _event_asset(k2)}
    index = PhysicalIndex({k1: [{"bbox": [10, 10, 30, 30], "image_id": 100, "video_id": 4, "track_id": 1}], k2: [{"bbox": [10, 10, 30, 30], "image_id": 101, "video_id": 4, "track_id": 2}]})
    out = TrackletAligner(assets, index).align({"event_key": "e", "fold": 0, "kind": "positive"}, "target", "v1:p1", 2, [_row(10, 1), _row(11, 2)])
    assert out["mapping_classification"] == "PHYSICAL_FRAGMENTATION"
    assert len(out["segments"]) == 2
