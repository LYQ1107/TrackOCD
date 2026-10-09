"""Synthetic contract checks; no private NAS assets required for tests."""

import json
from pathlib import Path

import pytest

from scripts.trackocd_core.audit_simowt_score_contract import EXPECTED_EMPTY_PREFIX, diagnose_source
from scripts.trackocd_core.audit_frontend import refresh_simowt_candidate, simowt_candidate

ROOT = Path(__file__).resolve().parents[2]


def synthetic_tracker(prefix=EXPECTED_EMPTY_PREFIX):
    statements = "\n".join("            " + line for line in prefix.splitlines())
    return "class IDOL_Tracker:\n    def match(self):\n        if self.empty:\n" + statements + "\n"


def test_inspected_cpu_score_view_alias_cannot_be_mislabeled_probability():
    result = diagnose_source(synthetic_tracker())
    assert result["device"] == "cpu"
    assert result["score_column_and_conf_list_share_storage"]
    assert result["scores_after_empty_branch"] == pytest.approx([0.7501] * 3)
    assert result["initial_candidates_passing_0_2"] == 1
    assert result["candidates_passing_0_2_after_source_prefix"] == 3
    assert result["second_sigmoid_range_on_unit_interval"] == pytest.approx([0.5, 0.73105857863])
    assert result["all_examples_exceed_registered_addnew_0_2"]
    assert not result["historical_stream_scores_or_fragmentation_cause_verified"]


def test_unknown_source_ast_is_not_executed():
    altered = EXPECTED_EMPTY_PREFIX.replace("bboxes[:, 4]", "bboxes[:, 4].clone()", 1)
    with pytest.raises(ValueError, match="refuse execution"):
        diagnose_source(synthetic_tracker(altered))


def test_simowt_receipt_preserves_unmatched_denominators_and_uncertainty():
    receipt = json.loads((ROOT / "outputs/trackocd_core/audit/nas_simowt_provenance.json").read_text())
    physical = receipt["current_source_stream"]
    histogram = physical["length"]["exact_histogram"]
    assert sum(histogram.values()) == physical["tracks"] == 649378
    assert sum(int(n) * c for n, c in histogram.items()) == physical["observations"] == 1853369
    assert histogram["1"] / physical["tracks"] == physical["length"]["single_frame_ratio"]
    coverage = receipt["historical_coverage_with_current_key_recount"]
    assert coverage["known"]["denominator"] == 4413
    assert coverage["novel"]["denominator"] == 819
    assert not coverage["geometry_matches_freshly_recomputed"]
    gate = receipt["qualification"]
    assert gate["selected_as_primary"] is False
    assert gate["no_novel_vocabulary_on_inspected_path"] == "SUPPORTED_STATIC"
    assert gate["checkpoint_supervision_exclusion"] == "UNVERIFIED"
    assert gate["legal_contract_status"] != "FAIL_NO_NOVEL_VOCABULARY"
    assert not receipt["historical_raw_clean_flags_treated_as_proof"]


def test_candidate_metadata_keeps_provenance_distinct_from_proven_leakage():
    candidate = simowt_candidate(ROOT, verify_private=False)
    assert candidate["qualification"]["legal_contract_status"] == "BLOCKED_PROVENANCE_NOT_PROVEN_LEAKAGE"
    assert not candidate["metrics_recomputed_on_a100"]
    assert not candidate["stream_present_on_a100"]
    assert candidate["historical_clean_declarations_are_not_proof"]


def test_candidate_refresh_preserves_physical_metrics_and_rejects_source_mismatch(tmp_path):
    base = tmp_path / "outputs/trackocd_core/audit"
    private = base / "nas_simowt_source"
    private.mkdir(parents=True)
    (private / "tracker.py").write_text("wrong source\n")
    (base / "nas_simowt_provenance.json").write_text(json.dumps({"private_source_copies_restored": [
        {"file": "tracker.py", "bytes": 13, "sha256": "0" * 64}
    ]}))
    snapshot = {"selected_frontend": None, "FROZEN_PHYSICAL_FRONTEND_written": False,
                "pandas_bytetrack_frozen_reference": {"unchanged": True}, "other_candidates": {}}
    with pytest.raises(ValueError, match="source copy byte mismatch"):
        refresh_simowt_candidate(snapshot, tmp_path)
    assert snapshot["pandas_bytetrack_frozen_reference"] == {"unchanged": True}
