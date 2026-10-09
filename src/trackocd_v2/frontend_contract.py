"""The shared contract for TrackOCD v2 physical frontend comparisons.

The four frontends are allowed to produce different physical tracks.  What
must be shared is the TAO Val universe, the temporal ordering, the evaluator
join boundary, and the metric definitions.  Keeping this contract in one
small module prevents a native stream from being mistaken for the public
SimOWT stream merely because both are called ``predicted``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FRONTENDS = ("SimOWT/Q0", "OVTR-native", "COVTrack-native", "COVTrack-NoSemantic")
OVTR_REFERENCE_ROLE = "VOCABULARY_ASSISTED_OVMOT_REFERENCE"
CLEAN_PHYSICAL_CANDIDATE_ROLE = "CLEAN_TRACKOCD_PHYSICAL_CANDIDATE"
PHYSICAL_FIELDS = (
    "sample_key",
    "source_sample_id",
    "video_id",
    "physical_track_id",
    "frame_ids",
    "boxes_xyxy",
    "image_paths",
    "quality",
    "stream_order",
    "source_split",
)
NATIVE_ROW_FIELDS = (
    "video_id",
    "image_id",
    "frame_index",
    "physical_track_id",
    "bbox_xyxy",
    "score",
    "category_id",
)


def _asset(path: Path) -> dict[str, str]:
    return {"path": str(path)}


def route_specs(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    """Return immutable route metadata, without probing or launching a job."""

    return {
        "SimOWT/Q0": {
            "slug": "simowt",
            "frontend_role": CLEAN_PHYSICAL_CANDIDATE_ROLE,
            "candidate_for_final_frontend": True,
            "clean_trackocd_final_frontend": True,
            "detector_ontology": "category_agnostic_foreground",
            "semantic_association_mode": "native_class_agnostic_tracking",
            "trackocd_backend_category_free": True,
            "novel_ontology_leakage": False,
            "execution_entrypoint": _asset(root / "scripts/run_arch1_blocking.sh"),
            "test_execution_entrypoint": _asset(root / "scripts/trackocd_v2/run_simowt_native.py"),
            "checkpoint": _asset(root / "data/iclr27_phase14b/checkpoints/simowt_weight.pth"),
            "fixed_overrides": {},
            "native_teta_role": "not_applicable_category_agnostic_route",
            "notes": "The existing public export is reusable as a physical stream, but does not export a legal appearance embedding.",
        },
        "OVTR-native": {
            "slug": "ovtr",
            "frontend_role": OVTR_REFERENCE_ROLE,
            "candidate_for_final_frontend": False,
            "clean_trackocd_final_frontend": False,
            "detector_ontology": "open_vocabulary",
            "semantic_association_mode": "native_ovmot_association",
            "trackocd_backend_category_free": True,
            "novel_ontology_leakage": False,
            "execution_entrypoint": _asset(root / "scripts/iclr27_phase75a/ovtr_native_eval.py"),
            "test_execution_entrypoint": _asset(root / "scripts/trackocd_v2/run_ovtr_native.py"),
            "checkpoint": _asset(root / "data/iclr27_phase14b/checkpoints/ovtr_5_frame.pth"),
            "fixed_overrides": {"score_mode": "base"},
            "native_teta_role": "reference_only",
            "notes": "Open-vocabulary detector outputs remain available to the native evaluator; TrackOCD receives only the normalized category-free physical stream.",
        },
        "COVTrack-native": {
            "slug": "covtrack_native",
            "frontend_role": CLEAN_PHYSICAL_CANDIDATE_ROLE,
            "candidate_for_final_frontend": True,
            "clean_trackocd_final_frontend": True,
            "detector_ontology": "open_vocabulary",
            "semantic_association_mode": "native_semantic_association",
            "trackocd_backend_category_free": True,
            "novel_ontology_leakage": False,
            "execution_entrypoint": _asset(root / "third_party/research_refs_phase4n/COVTrack/tools/test.py"),
            "test_execution_entrypoint": _asset(root / "scripts/trackocd_v2/run_covtrack_native.py"),
            "checkpoint": _asset(root / "data/iclr27_phase14b/checkpoints/covtrack_ctao_public.pth"),
            "fixed_overrides": {"model.tracker.confused_features": True},
            "native_teta_role": "reference_only",
            "notes": "Native COVTrack association is retained for the reference route; no category or text field enters the TrackOCD backend.",
        },
        "COVTrack-NoSemantic": {
            "slug": "covtrack_nosem",
            "frontend_role": CLEAN_PHYSICAL_CANDIDATE_ROLE,
            "candidate_for_final_frontend": True,
            "clean_trackocd_final_frontend": True,
            "detector_ontology": "open_vocabulary",
            "semantic_association_mode": "disabled",
            "trackocd_backend_category_free": True,
            "novel_ontology_leakage": False,
            "execution_entrypoint": _asset(root / "third_party/research_refs_phase4n/COVTrack/tools/test.py"),
            "test_execution_entrypoint": _asset(root / "scripts/trackocd_v2/run_covtrack_native.py"),
            "checkpoint": _asset(root / "data/iclr27_phase14b/checkpoints/covtrack_ctao_public.pth"),
            "fixed_overrides": {"model.tracker.confused_features": False},
            "native_teta_role": "reference_only",
            "notes": "Same frozen COVTrack checkpoint and detector as the native route; only the semantic association fusion cue is disabled while appearance and motion remain active.",
        },
    }


def evaluator_contract(root: Path = ROOT) -> dict[str, Any]:
    """Describe the one evaluator contract shared by all native streams."""

    return {
        "schema_version": "trackocd.v2.frontend_evaluator_contract.v1",
        "canonical_val_annotation": str((root / "data/raw/tao/annotations/validation.json").resolve()),
        "val_role": "development_and_frontend_selection",
        "physical_streams_are_frontend_specific": True,
        "same_v2_predicted_stream": False,
        "same_v2_evaluator_contract": True,
        "normalization": {
            "physical_stream_fields": list(PHYSICAL_FIELDS),
            "native_evaluator_row_fields": list(NATIVE_ROW_FIELDS),
            "category_free_backend_fields": [field for field in PHYSICAL_FIELDS if "category" not in field],
            "observation_order": "canonical TAO Val frame_index, then image_id; never source JSON order",
            "track_identity": "(video_id, frontend physical_track_id); local IDs are not cross-video identities",
            "invalid_box_policy": "strict mode fails; an explicit allow-invalid-boxes audit route retains every observation and records the count; never silently drops an observation",
        },
        "evaluation_boundary": {
            "model_and_causal_decisions_before_gt_join": True,
            "gt_join_for_scoring_only": True,
            "test_semantic_accessed": False,
        },
        "metrics": {
            "category_free_physical": ["OWTA", "AssA", "DetRe", "LocA"],
            "native_category_aware_reference": ["TETA"],
            "observability": ["Novel Observability", "Persistent Observability"],
            "ocd_safety": ["Commit-CT", "False Assignment"],
        },
        "selection_priority": [
            "no novel ontology leakage into TrackOCD backend",
            "Persistent Observability",
            "AssA",
            "simple-backend Commit-CT",
            "reproducibility_and_resource_cost",
        ],
    }


def route_manifest(root: Path = ROOT) -> dict[str, Any]:
    return {
        "schema_version": "trackocd.v2.frontend_bakeoff_route.v1",
        "status": "ROUTE_PLAN_READY",
        "frontends": list(FRONTENDS),
        "evaluator_contract": evaluator_contract(root),
        "routes": route_specs(root),
        "formal_common_feature_cache_authorized": False,
        "authorization_rule": "freeze FINAL_PHYSICAL_FRONTEND only after all eligible native routes have normalized streams and the shared Val metrics",
        "test_semantic_accessed": False,
    }
