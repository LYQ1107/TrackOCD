from src.iclr27_phase74.mapping import assess_mapping


def test_verified_direct_mapping_is_allowed():
    assert assess_mapping(mapping_type="DIRECT_NUMERIC", provenance_verified=True, one_to_one=True, frame_identity_verified=True, bbox_space_verified=True)["legal"]


def test_unverified_numeric_overlap_is_rejected():
    assert not assess_mapping(mapping_type="DIRECT_NUMERIC", provenance_verified=True, one_to_one=True, frame_identity_verified=True, bbox_space_verified=True, numeric_only=True)["legal"]


def test_asset_bijection_is_allowed():
    assert assess_mapping(mapping_type="CANONICAL_ASSET", provenance_verified=True, one_to_one=True, frame_identity_verified=True, bbox_space_verified=True)["legal"]


def test_one_to_many_mapping_is_blocked():
    assert not assess_mapping(mapping_type="CANONICAL_ASSET", provenance_verified=True, one_to_one=False, frame_identity_verified=True, bbox_space_verified=True)["legal"]


def test_category_assisted_mapping_is_blocked():
    assert not assess_mapping(mapping_type="CANONICAL_ASSET", provenance_verified=True, one_to_one=True, category_used=True, frame_identity_verified=True, bbox_space_verified=True)["legal"]
