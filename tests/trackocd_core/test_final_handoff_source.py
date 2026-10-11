"""Posthoc handoff safeguards do not alter any frozen scientific execution."""
import json
from pathlib import Path
import pytest
from scripts.trackocd_core.report_final_handoff import (
    mean_std, grouped_diagnostics, stage_audit, validate, p16_primary, score_table,
)

OUT=Path(__file__).resolve().parents[2]/"outputs/trackocd_core"


@pytest.fixture(scope="module")
def actual():
    def read(n):return json.loads((OUT/(n+".json")).read_text())
    return (read("CORE_TRAINING_FINAL_RESULT"),read("TRAIN_FIRST_REPRESENTATION_HELDOUT_RESULT"),
            read("TRAIN_FIRST_POLICY_HELDOUT_RESULT"),read("LIMITED_MASA_EVALUATION_RESULT"))


def test_statistic_is_actual_mean_not_h_of_other_means():
    assert mean_std([0.,1.])==(.5,.5)
    with pytest.raises(ValueError):mean_std([])
    with pytest.raises(ValueError):mean_std([float("nan")])


def test_all_original_requirements_have_truthful_disposition_not_strong_frontend():
    audit=stage_audit()
    assert [r["stage"] for r in audit]==[f"M{i}" for i in range(11)]
    assert audit[1]["scientific_status"]=="BLOCKED"
    assert all(audit[i]["scientific_status"]=="FAIL" for i in (5,7,8,9,10))
    assert "INCOMPARABLE" in audit[4]["scientific_status"]
    assert "255" in audit[0]["finding"] and "partial/unvalidated" in audit[0]["finding"]


def test_completed_input_and_frozen_hashes_are_required(actual):
    validate(*actual)
    wrong=dict(actual[0],actual_fits=32)
    with pytest.raises(ValueError):validate(wrong,*actual[1:])


def test_matched_only_runtime_is_full_matrix_aux_not_new_score(actual):
    rows=grouped_diagnostics(actual[-1]);assert len(rows)==len(actual[-1]["aggregate"])
    p16=p16_primary(rows);assert len(p16)==16
    full=next(r for r in p16 if r["method"]=="FULL")
    assert full["matched_h_score_mean"]==0
    assert (full["matched_known_denominator"],full["matched_novel_denominator"])==(1400,189)
    assert full["predicted_ID_commitment_coverage_mean"]==1
    assert all(r["matched_only_not_primary"] and r["same_global_mapping_no_second_Hungarian"] for r in rows)


def test_missing_order_is_not_accepted_as_complete_diagnostic(actual):
    v=dict(actual[-1],cases=actual[-1]["cases"][1:])
    with pytest.raises(ValueError):grouped_diagnostics(v)


def test_table_uses_actual_h_mean_and_order_seed_sd(actual):
    rows=p16_primary(actual[-1]["aggregate"])
    full=next(r for r in rows if r["method"]=="FULL")
    text=score_table([full]);assert "0.0000 ± 0.0000" in text
    assert full["old_acc_mean"]>0 and full["new_acc_mean"]>0
