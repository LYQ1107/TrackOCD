from src.iclr27_phase74.tracklet_alignment import align_tracklet


def event(): return {"event_key": "e", "fold": 0, "kind": "positive_existing"}


def row(i=0): return {"row_key": f"1:{i}:0:1:{i}", "bbox_xyxy": "[0,0,10,10]", "canonical_image_key": "img"}


def q(track, box=(0, 0, 10, 10), score=.5): return {"video_id": 9, "track_id": track, "bbox": list(box), "score": score}


def test_no_candidate_is_unmatched():
    x = align_tracklet(event(), "source", "v1:p1", 1, [row()], {}, source_file="x")
    assert x["mapping_classification"] == "UNMATCHED"


def test_unique_physical_track_mapping():
    x = align_tracklet(event(), "source", "v1:p1", 1, [row()], {"img": [q(1)]}, source_file="x")
    assert x["mapping_classification"] == "UNIQUE_MAPPING"


def test_multiple_eligible_tracks_are_ambiguous():
    x = align_tracklet(event(), "source", "v1:p1", 1, [row()], {"img": [q(1), q(2)]}, source_file="x")
    assert x["mapping_classification"] == "AMBIGUOUS"


def test_score_does_not_force_ambiguous_selection():
    x = align_tracklet(event(), "source", "v1:p1", 1, [row()], {"img": [q(1, score=.1), q(2, score=.9)]}, source_file="x")
    assert len(x["eligible_physical_tracks"]) == 2
