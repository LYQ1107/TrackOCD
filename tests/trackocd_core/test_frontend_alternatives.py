"""Source-only candidate contract checks, not a scientific frontend PASS."""
import json
from pathlib import Path

import pytest

from scripts.trackocd_core.audit_frontend_alternatives import (
    centered_first_box_coordinate, class_methods, config_keyword,
)

ROOT = Path(__file__).resolve().parents[2]


def test_literal_keyword_does_not_execute_config_or_base_reference():
    source = "raise RuntimeError('must not execute')\nmodel = dict(detector=unknown_base, load_public_dets=True)\n"
    assert config_keyword(source, "model", "load_public_dets") is True


def test_class_method_listing_does_not_import_runtime():
    source = "import nonexistent\nclass SamMasa(BaseModule):\n    def forward(self): pass\n    @property\n    def device(self): pass\n"
    assert class_methods(source, "SamMasa") == ["forward", "device"]


def test_centered_smoothing_depends_on_inaccessible_future_coordinate():
    assert centered_first_box_coordinate([0.] * 5) == 0.
    assert centered_first_box_coordinate([0., 0., 10., 0., 0.]) == 2.


def test_smaller_segment_is_not_silently_treated_as_upstream_smoothing():
    with pytest.raises(ValueError):
        centered_first_box_coordinate([0., 1.])


def test_source_receipt_preserves_unverified_frontend_and_aed_training_boundary():
    receipt = json.loads((ROOT / "outputs/trackocd_core/audit/frontend_alternatives_preflight.json").read_text())
    assert len(receipt["source_files"]) == 15
    assert sum(r["bytes"] for r in receipt["source_files"]) == receipt["source_payload_bytes"] == 96708
    assert all(r["git_blob_matches"] for r in receipt["source_files"])
    assert receipt["masa"]["sam_class_bases"] == ["BaseModule"]
    assert "predict" not in receipt["masa"]["sam_class_methods"]
    assert receipt["masa"]["demo_offline_post"]["causal_candidate_must_bypass_all_three"]
    assert not receipt["masa"]["weights_downloaded"]
    assert not receipt["aed"]["forbidden_supervision_proven_for_release"]
    assert not receipt["qualification"]["primary_frontend_selected"]
    assert not receipt["qualification"]["m1_complete"]
    assert not any(receipt["boundary"].values())


def test_conditional_smoke_is_bounded_frozen_train_image_only_and_not_an_r2():
    plan = json.loads((ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json").read_text())
    assert plan["status"] == "PREREGISTERED_CONDITIONAL_NOT_RUN"
    assert sum(w["bytes"] for w in plan["weights"]) == plan["conditional_weight_ceiling_bytes"] == 933925642
    assert not any(w["downloaded"] for w in plan["weights"])
    assert not plan["runtime"]["base_trackocd_environment_mutation_permitted"]
    assert plan["runtime"]["dependency_resolver_and_exact_wheel_lock_required_before_install"]
    limits = plan["execution_limits"]
    assert limits["max_images"] == 8 and limits["max_videos"] == 1
    assert limits["max_new_environment_source_weights_and_output_bytes"] == 8 * 1024**3
    assert not limits["full_val_or_cache_job_permitted"]
    assert not limits["tao_test_access_permitted"]
    assert not limits["unsafe_pickle_fallback_permitted"]
    assert not plan["proposal_contract"]["demo_offline_postprocessing_permitted"]
    assert not plan["proposal_contract"]["gt_boxes_labels_tracks_or_category_names_as_runtime_input"]
    assert not plan["proposal_contract"]["threshold_search_or_physical_training_permitted"]
