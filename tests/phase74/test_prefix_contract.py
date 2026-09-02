from src.iclr27_phase74.prefix_contract import get_visible_source_rows, get_visible_target_rows


def fixture():
    return {"v1:p1": [{"frame_id": i, "event_rank": i, "row_key": f"1:{i}:0:1:{i}"} for i in range(3)], "v2:p2": [{"frame_id": i, "event_rank": i, "row_key": f"2:{i}:0:2:{i}"} for i in range(4)]}


def test_source_registered_before_target():
    e = {"source_tracklet_keys": ["v1:p1"], "target_tracklet_key": "v2:p2"}
    assert [x["position"] for x in get_visible_source_rows(e, {}, fixture())] == [0, 1, 2]


def test_target_prefix_visibility_exact():
    e = {"source_tracklet_keys": ["v1:p1"], "target_tracklet_key": "v2:p2"}
    assert len(get_visible_target_rows(e, 2, {}, fixture())) == 2


def test_multiple_source_tracklets_not_concatenated():
    rows = fixture(); rows["v1:p3"] = [{"frame_id": 0, "event_rank": 0, "row_key": "1:0:0:3:8"}]
    e = {"source_tracklet_keys": ["v1:p1", "v1:p3"], "target_tracklet_key": "v2:p2"}
    out = get_visible_source_rows(e, {}, rows)
    assert [x["tracklet_key"] for x in out] == ["v1:p1"] * 3 + ["v1:p3"]


def test_source_tracklet_positions_are_independent():
    rows = fixture(); rows["v1:p3"] = [{"frame_id": 5, "event_rank": 5, "row_key": "1:5:0:3:8"}]
    e = {"source_tracklet_keys": ["v1:p1", "v1:p3"], "target_tracklet_key": "v2:p2"}
    out = get_visible_source_rows(e, {}, rows)
    assert [x["position"] for x in out] == [0, 1, 2, 0]


def test_target_positions_are_monotonic():
    e = {"source_tracklet_keys": ["v1:p1"], "target_tracklet_key": "v2:p2"}
    assert [x["position"] for x in get_visible_target_rows(e, 4, {}, fixture())] == [0, 1, 2, 3]


def test_future_target_rows_not_visible_at_early_prefix():
    e = {"source_tracklet_keys": ["v1:p1"], "target_tracklet_key": "v2:p2"}
    assert len(get_visible_target_rows(e, 1, {}, fixture())) == 1
