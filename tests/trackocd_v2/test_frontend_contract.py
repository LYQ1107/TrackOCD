import sys

from src.trackocd_v2.frontend_contract import FRONTENDS, PHYSICAL_FIELDS, evaluator_contract, route_specs
from scripts.trackocd_v2.build_common_features import main as feature_builder_main
from scripts.trackocd_v2.build_sharded_feature_cache import main as sharded_builder_main


def test_frontends_use_independent_streams_but_one_evaluator_contract():
    contract = evaluator_contract()
    assert contract["physical_streams_are_frontend_specific"] is True
    assert contract["same_v2_predicted_stream"] is False
    assert contract["same_v2_evaluator_contract"] is True
    assert set(PHYSICAL_FIELDS).isdisjoint({"category_id", "category_name", "text"})
    assert route_specs()["COVTrack-NoSemantic"]["fixed_overrides"]["model.tracker.confused_features"] is False
    assert route_specs()["OVTR-native"]["frontend_role"] == "VOCABULARY_ASSISTED_OVMOT_REFERENCE"
    assert route_specs()["OVTR-native"]["candidate_for_final_frontend"] is False
    assert route_specs()["COVTrack-native"]["candidate_for_final_frontend"] is True
    assert set(route_specs()) == set(FRONTENDS)


def test_legacy_predicted_feature_builder_is_disabled(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_common_features.py", "--split", "pred"])
    assert feature_builder_main() == 3
    assert "PREDICTED_LEGACY_FEATURE_BUILD_DISABLED" in capsys.readouterr().err


def test_formal_sharded_cache_requires_final_frontend(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["build_sharded_feature_cache.py", "--frontend", "simowt", "--preflight-only"])
    assert sharded_builder_main() == 3
    assert "FORMAL_CACHE_NOT_AUTHORIZED" in capsys.readouterr().out
