import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from scripts.trackocd_core.smoke_predicted_features import select_observed_prefixes, verify_prefixes


def fixture():
    video = {"video_id": 4, "images": [{"frame_index": i, "image_id": i} for i in range(4)]}
    arrays = {"frame_offsets": np.asarray([0, 2, 4, 6, 7]),
              "track_id": np.asarray([9, 2, 9, 3, 9, 2, 100]),
              "boxes": np.tile([0., 0., 4., 4.], (7, 1)), "score": np.full(7, .4)}
    return video, arrays


def test_first_frame_selection_not_future_longer_tracks_and_causal_stop():
    video, arrays = fixture()
    class NoLabels(dict):
        def __getitem__(self, key):
            assert key not in {"category_id", "roles", "gt", "total_track_length"}
            return super().__getitem__(key)
    rows, scanned = select_observed_prefixes(NoLabels(video), NoLabels(arrays))
    assert [r["track_id"] for r in rows] == [2, 9]
    assert scanned == 3
    assert [[ob["image"]["frame_index"] for ob in r["observations"]] for r in rows] == [[0, 2], [0, 1]]
    arrays["boxes"][6:] = np.nan
    assert select_observed_prefixes(video, arrays) == (rows, scanned)


def test_short_track_is_not_replaced_by_longer_later_track():
    video, arrays = fixture()
    arrays["track_id"][5] = 3
    rows, scanned = select_observed_prefixes(video, arrays)
    assert [r["track_id"] for r in rows] == [2, 9]
    assert [len(r["observations"]) for r in rows] == [1, 2]
    assert scanned == 4
    _, tested = verify_prefixes(np.ones((3, 768), dtype=np.float16), np.ones((3, 4)), np.ones(3), np.asarray([0, 0, 1]), [1, 2])
    assert tested == [1, 1, 2]


def test_empty_first_frame_is_not_replaced():
    video, arrays = fixture()
    arrays["frame_offsets"][1] = 0
    with pytest.raises(ValueError, match="Empty first image"):
        select_observed_prefixes(video, arrays)


@pytest.mark.parametrize("field,value", [("tracks", 5), ("frames", 9), ("observations", 3)])
def test_smoke_cannot_expand_to_unbounded_cache(field, value):
    with pytest.raises(ValueError, match="smoke bounds"):
        select_observed_prefixes(*fixture(), **{field: value})


def test_registered_engineering_boundary_and_same_encoder_not_gt_promotion():
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "configs/trackocd_core/masa_predicted_feature_smoke.json").read_text())
    assert plan["maximum_observations"] == 8
    assert plan["maximum_tracks"] == 4 and plan["batch_size"] == 4
    assert plan["model_view_fields"] == ["visual", "boxes_normalized_xyxy", "quality", "elapsed_frames"]
    for field in ("training", "gt_or_role_annotation_access", "test_access", "formal_cache_started",
                  "primary_freeze_permitted", "scientific_pass_permitted", "new_image_or_checkpoint_download"):
        assert plan[field] is False


def test_common_config_and_inherited_crop_interface_are_byte_identical():
    from src.trackocd_v2.io import sha256_file
    root = Path(__file__).resolve().parents[2]
    plan = json.loads((root / "configs/trackocd_core/masa_predicted_feature_smoke.json").read_text())
    assert sha256_file(root / plan["common_config"]) == plan["common_config_sha256"]
    for name, expected in plan["unchanged_source_sha256"].items():
        assert sha256_file(root / name) == expected


def test_actual_input_barrier_denies_annotations_roles_test_and_network():
    code = """
from scripts.trackocd_core.smoke_predicted_features import install_input_barrier
import socket
install_input_barrier()
for path in ('/data3/liuyeqiang/TAO-Amodal/annotations/validation.json',
             '/data3/liuyeqiang/TrackOCD/configs/trackocd_core/roles.json',
             '/data3/liuyeqiang/TAO-Amodal/frames/test/forbidden.jpg'):
    try: open(path)
    except PermissionError: pass
    else: raise AssertionError('Prohibited path opened')
try: socket.getaddrinfo('example.com', 443)
except PermissionError: pass
else: raise AssertionError('Network allowed')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
