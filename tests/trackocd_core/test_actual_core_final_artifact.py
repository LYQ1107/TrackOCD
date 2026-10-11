"""Actual final synthesis bindings, not an assertion of algorithmic success."""
import json
from pathlib import Path

from src.trackocd_v2.io import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/trackocd_core"


def test_actual_core_final_documents_and_all_public_artifacts_are_hash_bound():
    d = json.loads((OUT / "CORE_TRAINING_FINAL_RESULT.json").read_text())
    assert d["status"] == "COMPLETE_REAL_TRAIN_AND_LIMITED_PREDICTED_EVALUATION_SCIENTIFIC_NEGATIVE"
    assert (d["actual_fits"], len(d["selected_checkpoints"]), len(d["retained_checkpoints"])) == (33, 33, 99)
    assert (d["limited_MASA_unique_executions"], d["limited_MASA_logical_cases"], d["canonical_per_order_rows"]) == (840, 2040, 4260)
    assert len(d["final_document_sha256"]) == 5
    for name, sha in d["final_document_sha256"].items():
        assert sha256_file(ROOT / "docs/trackocd_core" / name) == sha
    for name, sha in d["artifact_sha256"].items():
        assert sha256_file(OUT / name) == sha
    assert d["GT_in_live_memory"] is d["Val_training_or_tuning"] is d["TAO_Test_access"] is False
    assert d["historical_M1_status"] == "BLOCKED_FRONTEND_QUALITY"
    assert not d["scientific_conclusions"]["new_TrackOCD_method_paper_claim_supported"]
    assert d["scientific_conclusions"]["complete_credible_negative_result"]
    assert len(d["limited_MASA_p16_fixed_comparison_sign_counts"]) == 7


def test_actual_final_publication_keeps_uncertainty_aliases_and_real_mean_h():
    d = json.loads((OUT / "CORE_TRAINING_FINAL_RESULT.json").read_text())
    provenance = d["original_registration_and_operational_recovery_provenance"]
    assert provenance["original_frozen_inference_registration_commits"] == ["eddbc50435e87b85b7141decfb2a5dc32290ef24"]
    assert provenance["scheduling_recovery_registration_commit"] == "c662f76606e23ee2cd054b507db7c0da700dcece"
    assert len(provenance["actual_successful_terminal_receipts_sha256"]) == 4
    full = next(r for r in d["limited_MASA_p16_aggregate"] if r["method"] == "FULL")
    assert full["old_acc_mean"] > 0 and full["new_acc_mean"] > 0 and full["h_score_mean"] == 0
    dp = next(r for r in d["limited_MASA_p16_aggregate"] if r["method"] == "B2_track_dpmeans")
    assert dp["correct_commit_ct_mean"] > 0
    assert d["resources"]["conservative_task_private_file_bytes"] < d["resources"]["storage_soft_bytes"]
    text = (ROOT / "docs/trackocd_core/CORE_TRAINING_FINAL_REPORT.md").read_text()
    assert "ANNOTATED-CADENCE CAUSAL REPLAY" in text and "PHE INCOMPARABLE" in text
    assert "unavailable and never fabricated as zero" in text
    assert "Original frozen inference registration" in text
