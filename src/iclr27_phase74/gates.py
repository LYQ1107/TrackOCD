"""Evidence-derived Phase74 gates."""
from __future__ import annotations

from typing import Any, Mapping


def _check(value: bool | None, reason: str) -> dict[str, Any]:
    return {"pass": value, "reason": reason}


def compute_gates(*, input_verification: Mapping[str, Any], manifest: Mapping[str, Any], prefix: Mapping[str, Any], assets: Mapping[str, Any], lineage: Mapping[str, Any], replay: Mapping[str, Any], dependency: Mapping[str, Any], tests: Mapping[str, Any], observability: Mapping[str, Any], resource: Mapping[str, Any]) -> dict[str, Any]:
    matches = input_verification.get("matches", {})
    i = all(bool(v) for v in matches.values()) if matches else False
    o = bool(manifest.get("positive_line_order_preserved") and manifest.get("negative_line_order_preserved") and manifest.get("event_key_unique") and manifest.get("positive_count") == 76 and manifest.get("negative_count") == 76 and not manifest.get("implicit_sort_used", True))
    pfx = bool(prefix.get("source_before_target") and prefix.get("source_rows_not_concatenated") and prefix.get("target_position_monotonic") and prefix.get("contract_status") == "PROVEN_FROM_RUNNER_AND_STREAM")
    a = bool(assets.get("required_images", 0) > 0 and assets.get("missing_images", 0) == 0 and assets.get("ambiguous_images", 0) == 0 and assets.get("mapping_method_legal", False))
    f = bool(lineage.get("five_field_roundtrip", False) and lineage.get("frame_id_source") and lineage.get("proposal_local_id_source"))
    q = bool(replay.get("control_replay_complete") and replay.get("control_equivalence_pass") and replay.get("repeat_determinism_pass")) if replay.get("required") else True
    t = dependency.get("classification") in {"NO_TEXT_CATEGORY_FORWARD_PATH", "TEXT_PATH_PRESENT_OUTPUT_INVARIANT"}
    c = bool(tests.get("future_append", False) and pfx)
    n = bool(tests.get("category_shuffle", False) and tests.get("event_label_swap", False) and tests.get("physical_id_renumber", False))
    e = bool(manifest.get("positive_count") == 76 and manifest.get("negative_count") == 76 and manifest.get("event_key_unique") and set(manifest.get("prefixes", [1,2,4,8,16])) == {1,2,4,8,16})
    r = bool(tests.get("repeat_determinism", False) and tests.get("atomic_crash", False))
    s = bool(resource.get("ram_safety", False) and resource.get("no_external_kill", True) and resource.get("no_duplicate_supervisor", True))
    return {"input_integrity": _check(i, "all registered hashes/counts match" if i else "one or more registered hashes/counts mismatch"),
            "original_order": _check(o, "raw manifest order and complete set"), "prefix_contract": _check(pfx, "runner-derived source/target visibility"),
            "asset_lineage": _check(a, "canonical event asset inventory"), "five_field_lineage": _check(f, "upstream frame/proposal local IDs and roundtrip"),
            "q0_replay_equivalence": _check(q, "control replay equivalence" if replay.get("required") else "not required because no replay selected"),
            "text_category_dependency": {"classification": dependency.get("classification"), "qualified": t},
            "causality": _check(c, "future append and source-before-target tests"), "no_leakage": _check(n, "metamorphic leakage tests"),
            "evaluator_contract": _check(e, "fixed 76+76 denominator/prefix"), "reproducibility": _check(r, "independent canonical repeat and atomic crash"),
            "resource_safety": _check(s, "preflight and process ownership")}
