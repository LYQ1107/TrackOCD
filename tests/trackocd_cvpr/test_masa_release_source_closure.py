from scripts.trackocd_cvpr.audit_masa_release_source_closure import build_receipt


def test_masa_release_and_image_only_routes_are_not_conflated():
    receipt = build_receipt()

    assert receipt["status"] == "PASS_SOURCE_CLOSURE_RELEASE_ROUTE_MISMATCH_QUALIFICATION_OPEN"
    assert all(receipt["checks"].values())
    route = receipt["release_route"]
    assert route["official_load_public_dets"] is True
    assert route["current_trackocd_candidate_load_public_dets"] is False
    assert route["current_trackocd_candidate_public_det_path"] is None
    assert receipt["qualification_boundary"]["official_release_route_reproduced"] is False
    assert receipt["qualification_boundary"]["current_image_only_candidate_is_official_tao_route"] is False
    assert receipt["boundary"]["tao_test_read"] is False
