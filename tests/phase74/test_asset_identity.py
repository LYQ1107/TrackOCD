from src.iclr27_phase74.asset_identity import canonical_image_key, canonical_video_key, norm_rel


def test_canonical_keys_ignore_numeric_track_identity():
    v = canonical_video_key("YFCC100M", "train", "train/YFCC100M/v_a")
    assert canonical_image_key(v, 3, "anything.jpg") == "yfcc100m|train|train/YFCC100M/v_a|frame=3"


def test_path_normalization():
    assert norm_rel("./train\\YFCC100M//v_a") == "train/YFCC100M/v_a"
