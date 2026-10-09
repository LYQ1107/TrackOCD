import json
from pathlib import Path

import pytest

from src.trackocd_core.pilot import select_pilot


def fixture_annotation():
    images, annotations = [], []
    for category in range(1, 13):
        for vid_index in range(10 if category <= 4 else 3):
            video = category * 100 + vid_index
            for frame in range(20):
                image_id = len(images)
                images.append({"id": image_id, "video_id": video, "frame_index": frame})
                annotations.append({"category_id": category, "video_id": video, "track_id": 7, "image_id": image_id})
    return {"images": images, "annotations": annotations}


def configuration():
    return json.loads((Path(__file__).resolve().parents[2] / "configs/trackocd_core/gt_feasibility_pilot.json").read_text())


def test_pilot_is_small_legal_category_and_video_disjoint():
    selected, report = select_pilot(fixture_annotation(), set(range(1, 13)), configuration())
    assert report["tracks"] == 64 and report["observations"] == 1024
    assert report["partition_tracks"] == {"representation": 16, "policy_train": 24, "heldout_selection": 24}
    assert report["purpose_tracks"] == {"prototype": 4, "representation_fit": 12, "stream": 48}
    assert report["selected_known_categories"] == [1, 2, 3, 4]
    assert report["policy_train_pseudo_novel_categories"] == [5, 6, 7, 8]
    assert report["heldout_selection_pseudo_novel_categories"] == [9, 10, 11, 12]
    sets = [set(v) for v in report["partition_videos"].values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len({(r["video_id"], r["gt_track_id"]) for r in selected}) == 64
    assert all(len(r["observations"]) == 16 for r in selected)
    assert all(r["observations"][-1]["image"]["frame_index"] == 15 for r in selected)


def test_order_of_input_does_not_change_preregistered_sampling():
    annotation = fixture_annotation()
    before, summary = select_pilot(annotation, set(range(1, 13)), configuration())
    annotation["images"].reverse()
    annotation["annotations"].reverse()
    after, other = select_pilot(annotation, set(range(1, 13)), configuration())
    assert summary == other
    assert [(r["category"], r["video_id"], r["gt_track_id"]) for r in before] == [
        (r["category"], r["video_id"], r["gt_track_id"]) for r in after]


def test_unknown_train_categories_are_excluded_not_redefined():
    annotation = fixture_annotation()
    for image, ob in zip(annotation["images"], annotation["annotations"]):
        if ob["category_id"] == 12:
            ob["category_id"] = 999
    with pytest.raises(RuntimeError, match="INSUFFICIENT_TRAIN_SUPPORT"):
        select_pilot(annotation, set(range(1, 13)), configuration())


def test_duplicate_frame_or_inconsistent_track_category_rejected():
    annotation = fixture_annotation()
    annotation["annotations"].append(dict(annotation["annotations"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        select_pilot(annotation, set(range(1, 13)), configuration())
    annotation = fixture_annotation()
    annotation["annotations"][0]["category_id"] = 2
    with pytest.raises(ValueError, match="Inconsistent"):
        select_pilot(annotation, set(range(1, 13)), configuration())
