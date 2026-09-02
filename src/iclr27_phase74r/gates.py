"""Evidence-derived mandatory Phase74R gates."""
from __future__ import annotations

from typing import Any, Mapping


MANDATORY = (
    "INPUT_INTEGRITY", "MODEL_EVENT_ORDER", "PREFIX_CONTRACT", "ASSET_IDENTITY",
    "BRANCH_A_INTEGRATION", "RELIABILITY_CONTRACT", "FRAGMENTATION_CONTRACT",
    "NO_LEAKAGE", "CAUSALITY", "EVALUATOR_DENOMINATOR", "REAL_METAMORPHIC_TESTS",
    "REPRODUCIBILITY", "ARTIFACT_FORMAT", "RESOURCE_SAFETY",
)


def gate(passed: bool, reason: str, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"pass": bool(passed), "reason": reason, "evidence": dict(evidence or {})}


def compute_gates(*, input_ok: bool, model_order: Mapping[str, Any], prefix_ok: bool,
                  asset_identity: Mapping[str, Any], branch_fixture: Mapping[str, Any],
                  reliability: Mapping[str, Any], fragmentation: Mapping[str, Any],
                  metamorphic: Mapping[str, Any], reproducibility: bool,
                  artifact_format: Mapping[str, Any], resource: Mapping[str, Any],
                  metadata_count: int, metadata_key_count: int) -> dict[str, dict[str, Any]]:
    return {
        "INPUT_INTEGRITY": gate(input_ok, "registered input hashes/counts", {"input_ok": input_ok}),
        "MODEL_EVENT_ORDER": gate(bool(model_order.get("model_order_exactly_reproducible")), "actual manifest/fallback order reproduced", {"source": model_order.get("source"), "count": model_order.get("count"), "order_sha256": model_order.get("order_sha256")}),
        "PREFIX_CONTRACT": gate(prefix_ok, "source complete-before-target and target causal prefix", {"prefix_ok": prefix_ok}),
        "ASSET_IDENTITY": gate(bool(asset_identity.get("schema_complete") and asset_identity.get("duplicates_preserved")), "protocol/content/file identity records are complete and duplicate-preserving", asset_identity),
        "BRANCH_A_INTEGRATION": gate(bool(branch_fixture.get("mapped_images", 0) == 1 and branch_fixture.get("q0_candidate_count", 0) > 0 and branch_fixture.get("classification") == "UNIQUE_MAPPING"), "synthetic content-key Branch-A pipeline", branch_fixture),
        "RELIABILITY_CONTRACT": gate(bool(reliability.get("null_before_replay") and reliability.get("no_zero_for_unreplayed")), "unreplayed Q0 observations remain null", reliability),
        "FRAGMENTATION_CONTRACT": gate(bool(fragmentation.get("ambiguous_overlap") and fragmentation.get("physical_fragmentation")), "overlap and non-overlap physical tracks classified separately", fragmentation),
        "NO_LEAKAGE": gate(bool(metamorphic.get("category_shuffle") and metamorphic.get("event_label_swap") and metamorphic.get("physical_id_renumber")), "category/label/ID transformations do not affect physical evidence", {k: metamorphic.get(k) for k in ("category_shuffle", "event_label_swap", "physical_id_renumber")}),
        "CAUSALITY": gate(bool(metamorphic.get("future_append") and metamorphic.get("source_before_target")), "future append and source-before-target invariance", {k: metamorphic.get(k) for k in ("future_append", "source_before_target")}),
        "EVALUATOR_DENOMINATOR": gate(bool(metadata_count == 152 and metadata_key_count == 152 and model_order.get("model_key_set_equals_metadata")), "76+76 metadata denominator and model/evaluator event universe", {"metadata_count": metadata_count, "metadata_key_count": metadata_key_count, "model_count": model_order.get("count"), "model_metadata_matched": model_order.get("model_metadata_matched"), "model_key_set_equals_metadata": model_order.get("model_key_set_equals_metadata")}),
        "REAL_METAMORPHIC_TESTS": gate(all(bool(v) for k, v in metamorphic.items() if k not in {"details"}), "all registered metamorphic tests executed from comparisons", {k: v for k, v in metamorphic.items() if k != "details"}),
        "REPRODUCIBILITY": gate(bool(reproducibility), "independent fixture pipeline hashes agree", {"reproducibility": reproducibility}),
        "ARTIFACT_FORMAT": gate(bool(artifact_format.get("json_parse") and artifact_format.get("jsonl_parse") and artifact_format.get("timeline_json_array")), "JSON/JSONL/command log formats parse", artifact_format),
        "RESOURCE_SAFETY": gate(bool(resource.get("preflight_recorded") and resource.get("postflight_recorded") and resource.get("no_external_kill") and resource.get("no_duplicate_supervisor")), "resource and process records are real postflight evidence", resource),
    }


def blocked_status(gates: Mapping[str, Mapping[str, Any]]) -> str | None:
    for name in MANDATORY:
        if not bool(gates.get(name, {}).get("pass")):
            suffix = "DENOMINATOR" if name == "EVALUATOR_DENOMINATOR" else name
            return f"PHASE74R_BLOCKED_{suffix}"
    return None
