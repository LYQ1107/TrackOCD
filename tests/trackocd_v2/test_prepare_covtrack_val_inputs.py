import json
from pathlib import Path

from scripts.trackocd_v2.prepare_covtrack_val_inputs import _derive_annotation, _load_lvis_classes


def test_derive_annotation_adds_video_local_frame_ids_without_reordering_contract():
    annotation = {
        "videos": [{"id": 9}],
        "images": [
            {"id": 3, "video_id": 9, "frame_index": 20},
            {"id": 2, "video_id": 9, "frame_index": 10},
            {"id": 4, "video_id": 8, "frame_index": 3},
        ],
        "annotations": [],
        "tracks": [],
        "categories": [],
    }
    derived, counts = _derive_annotation(annotation)
    rows = {row["id"]: row for row in derived["images"]}
    assert rows[2]["frame_id"] == 0
    assert rows[3]["frame_id"] == 1
    assert rows[4]["frame_id"] == 0
    assert counts == {"9": 2, "8": 1}
    assert [row["id"] for row in derived["images"]] == [3, 2, 4]


def test_load_lvis_classes_reads_only_pinned_source(tmp_path):
    path = tmp_path / "ovtrack/models/roi_heads/class_name.py"
    path.parent.mkdir(parents=True)
    path.write_text("LVIS_CLASSES = (\n    'a',\n)\n", encoding="utf-8")
    # The helper must reject a malformed/non-public class source rather than
    # silently creating a vocabulary with a different ontology.
    try:
        _load_lvis_classes(tmp_path)
    except ValueError as exc:
        assert "1203" in str(exc)
    else:
        raise AssertionError("malformed class source was accepted")
