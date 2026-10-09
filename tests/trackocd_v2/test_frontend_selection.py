from scripts.trackocd_v2.register_frontend_metrics import _teta_status_allowed
from scripts.trackocd_v2.select_frontend import _is_final_frontend_candidate


def test_vocabulary_assisted_ovtr_is_excluded_from_clean_final_selection():
    assert _is_final_frontend_candidate(
        "OVTR-native",
        {"route_contract": {"frontend_role": "VOCABULARY_ASSISTED_OVMOT_REFERENCE"}},
    ) is False


def test_clean_candidate_remains_eligible():
    assert _is_final_frontend_candidate(
        "COVTrack-native",
        {"route_contract": {"frontend_role": "CLEAN_TRACKOCD_PHYSICAL_CANDIDATE"}},
    ) is True


def test_ovtr_teta_failure_is_nonblocking_but_other_routes_remain_strict():
    assert _teta_status_allowed("ovtr", "OVTR_NATIVE_TETA_REFERENCE_EVAL_FAILED") is True
    assert _teta_status_allowed("simowt", "OVTR_NATIVE_TETA_REFERENCE_EVAL_FAILED") is False
