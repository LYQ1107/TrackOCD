import pytest

from scripts.trackocd_v2.build_test_track_stream import _records
from scripts.trackocd_v2.build_test_trackeval_gt import _class_agnostic


def _annotation_without_semantic_fields():
    return {
        "videos": [{"id": 1}],
        "images": [{"id": 11, "video_id": 1, "frame_index": 0, "file_name": "v/11.jpg"}],
        "annotations": [{
            "video_id": 1,
            "track_id": 7,
            "image_id": 11,
            "bbox": [1, 2, 3, 4],
        }],
    }


def test_structural_records_do_not_require_test_category_ids():
    rows, categories = _records(_annotation_without_semantic_fields(), include_categories=False)
    assert len(rows) == 1
    assert categories == {}
    assert "category_id" not in rows[0]


def test_labelled_records_require_category_ids():
    with pytest.raises(KeyError):
        _records(_annotation_without_semantic_fields(), include_categories=True)


def test_trackeval_gt_adapter_is_single_class_and_clears_category_exclusions():
    converted = _class_agnostic({
        "info": {"year": 2026},
        "licenses": [],
        "videos": [{"id": 1, "neg_category_ids": [7], "not_exhaustive_category_ids": [8]}],
        "images": [{"id": 11, "video_id": 1}],
        "annotations": [{"id": 1, "category_id": 7, "track_id": 3, "video_id": 1, "image_id": 11}],
        "tracks": [{"id": 3, "category_id": 7, "video_id": 1}],
        "categories": [{"id": 7, "name": "source"}],
    })
    assert converted["categories"] == [{"id": 1, "name": "object"}]
    assert converted["annotations"][0]["category_id"] == 1
    assert converted["tracks"][0]["category_id"] == 1
    assert converted["videos"][0]["neg_category_ids"] == []
    assert converted["videos"][0]["not_exhaustive_category_ids"] == []
